"""CLAP embedding + FAISS indexing for long-form audio retrieval.

Supports:
- Offline bulk preprocessing of an entire dataset with multi-GPU sharding
- Incremental ingestion: cached per-audio .npz files are skipped on re-runs
- Inference-time scoped search: retrieval is restricted to a single audio_id

Cache layout (under cache_dir):
    embeddings/
        {audio_id}.npz          -- (N, D) embeddings + start/end timestamp arrays
    faiss/
        {audio_id}.index        -- per-audio IndexFlatIP (N vectors for that audio only)
        {audio_id}.jsonl        -- per-audio segment metadata (start, end per row)

Search is always scoped to one audio_id, so results can never bleed across recordings.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import faiss
import numpy as np
import torch
import torchaudio

from areal.utils.logging import getLogger

logger = getLogger("CLAPIndexer")

# ──────────────────────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────────────────────
CLAP_MODEL_ID = "laion/larger_clap_general"
CLAP_SAMPLE_RATE = 48_000  # Hz – CLAP models expect 48 kHz input
EMBED_DIM = 512  # laion/larger_clap_general projection dim


# ──────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────
@dataclass
class CLAPConfig:
    """Configuration for the CLAP preprocessing pipeline."""

    # Paths
    data_root: str  # root of the prepared dataset (contains manifest.json + split dirs)
    cache_dir: str  # directory where embeddings and FAISS indexes are stored

    # Model
    model_id: str = CLAP_MODEL_ID

    # Segmentation
    chunk_duration: float = 9.0  # seconds per chunk
    chunk_stride: float = 3.0  # stride between chunks (seconds); use == chunk_duration for no overlap

    # Inference
    batch_size: int = 32  # audio chunks per CLAP forward pass
    device: str = "cuda"

    # Dataset scope
    splits: list = field(default_factory=lambda: ["train", "val", "test"])


# ──────────────────────────────────────────────────────────────
# Data discovery helpers
# ──────────────────────────────────────────────────────────────
def discover_audio_files(data_root: str, splits: list[str]) -> list[dict]:
    """Return a list of dicts with keys: audio_id, wav_path, split.

    Scans each split directory for (audio_N.wav, audio_N.json) pairs and reads
    the audio_id from the JSON sidecar.
    """
    root = Path(data_root)
    entries = []
    for split in splits:
        split_dir = root / split
        if not split_dir.exists():
            logger.warning(f"Split directory not found, skipping: {split_dir}")
            continue
        for wav_path in sorted(split_dir.glob("audio_*.wav")):
            json_path = wav_path.with_suffix(".json")
            if not json_path.exists():
                logger.warning(f"No JSON sidecar for {wav_path}, skipping.")
                continue
            meta = json.loads(json_path.read_text())
            entries.append(
                {
                    "audio_id": meta["audio_id"],
                    "wav_path": str(wav_path),
                    "split": split,
                    "duration_seconds": meta.get("duration_seconds", 0.0),
                }
            )
    return entries


# ──────────────────────────────────────────────────────────────
# Core indexer
# ──────────────────────────────────────────────────────────────
class CLAPIndexer:
    """Manages CLAP embedding and per-audio FAISS indexing.

    Each audio file gets its own FAISS index so that retrieval is always
    scoped to the recording being queried — results can never bleed across
    different audio_ids.

    Usage (offline bulk):
        indexer = CLAPIndexer(config)
        indexer.run_embedding_phase(worker_id=0, num_workers=2)  # per GPU
        # after all workers done:
        indexer.build_faiss_indexes()

    Usage (inference-time, scoped to one recording):
        indexer = CLAPIndexer(config)
        results = indexer.search_by_text("applause", audio_id="meetingbank_train_0", k=10)
        results = indexer.search_by_audio(waveform, audio_id="meetingbank_train_0", k=10)
    """

    def __init__(self, config: CLAPConfig):
        self.config = config
        self.cache_dir = Path(config.cache_dir)
        self.embed_dir = self.cache_dir / "embeddings"
        self.faiss_dir = self.cache_dir / "faiss"
        self.embed_dir.mkdir(parents=True, exist_ok=True)
        self.faiss_dir.mkdir(parents=True, exist_ok=True)

        self._model = None
        self._processor = None
        # In-memory cache of loaded per-audio FAISS indexes
        self._index_cache: dict[str, tuple] = {}  # audio_id → (index, segments)

    # ── model loading ──────────────────────────────────────────

    def _load_model(self):
        """Lazily load CLAP model + processor onto the configured device.

        Bypasses ``from_pretrained`` for model weights to avoid FSDP / meta-device
        init contexts. Loads the config, creates an empty model on the target
        device, then loads the state dict with ``assign=True``.
        """
        if self._model is not None:
            return
        from transformers import ClapConfig, ClapModel, ClapProcessor

        logger.info(f"Loading CLAP model: {self.config.model_id}")
        self._processor = ClapProcessor.from_pretrained(self.config.model_id)

        # Load config and resolve the cached checkpoint path
        clap_config = ClapConfig.from_pretrained(self.config.model_id)
        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(
            self.config.model_id, filename="pytorch_model.bin"
        )
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        # Build model on meta (fast, no memory), move to real device, load weights
        with torch.device("meta"):
            model = ClapModel(clap_config)
        model = model.to_empty(device=self.config.device)
        model.load_state_dict(state_dict, assign=True)

        self._model = model
        self._model.eval()
        logger.info(f"CLAP model ready on {self.config.device}")

    def _unload_model(self):
        """Release model memory (useful after the embedding phase)."""
        if self._model is not None:
            del self._model, self._processor
            self._model = None
            self._processor = None
            if self.config.device == "cuda":
                torch.cuda.empty_cache()

    # ── embedding helpers ──────────────────────────────────────

    def _segment_waveform(
        self, waveform: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Split a mono waveform (at CLAP_SAMPLE_RATE) into fixed-length chunks.

        Returns:
            chunks:  (N, chunk_samples) float32 array (last chunk zero-padded)
            starts:  (N,) float32 start times in seconds
            ends:    (N,) float32 end times in seconds
        """
        chunk_samples = int(self.config.chunk_duration * CLAP_SAMPLE_RATE)
        stride_samples = int(self.config.chunk_stride * CLAP_SAMPLE_RATE)
        total = len(waveform)

        chunk_list, starts, ends = [], [], []
        offset = 0
        while offset < total:
            end_offset = min(offset + chunk_samples, total)
            chunk = waveform[offset:end_offset]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            chunk_list.append(chunk)
            starts.append(offset / CLAP_SAMPLE_RATE)
            ends.append(min(end_offset, total) / CLAP_SAMPLE_RATE)
            if end_offset >= total:
                break
            offset += stride_samples

        chunks = np.stack(chunk_list, axis=0)  # (N, chunk_samples)
        return chunks, np.array(starts, dtype=np.float32), np.array(ends, dtype=np.float32)

    def _batch_embed_audio(self, chunks: np.ndarray) -> np.ndarray:
        """Run CLAP audio encoder on a (N, chunk_samples) array.

        Returns L2-normalised embeddings of shape (N, EMBED_DIM).
        """
        self._load_model()
        all_embs = []
        for i in range(0, len(chunks), self.config.batch_size):
            batch = list(chunks[i : i + self.config.batch_size])
            inputs = self._processor(
                audios=batch,
                sampling_rate=CLAP_SAMPLE_RATE,
                return_tensors="pt",
            )
            inputs = {k: v.to(self.config.device) for k, v in inputs.items()}
            with torch.no_grad():
                audio_features = self._model.get_audio_features(**inputs)
            audio_features = torch.nn.functional.normalize(audio_features, dim=-1)
            all_embs.append(audio_features.cpu().float().numpy())
        return np.concatenate(all_embs, axis=0)

    # ── per-file embedding ──────────────────────────────────────

    def embed_audio_file(
        self, wav_path: str, audio_id: str, force: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embed one audio file, returning (embeddings, starts, ends).

        Loads from cache if available (skip recompute unless force=True).

        Args:
            wav_path:  Path to the .wav file (any sample rate, any channels).
            audio_id:  Unique identifier used for the cache filename.
            force:     If True, recompute even if cache exists.

        Returns:
            embeddings: (N, EMBED_DIM) float32 — L2-normalised CLAP embeddings
            starts:     (N,) float32 — segment start times in seconds
            ends:       (N,) float32 — segment end times in seconds
        """
        cache_path = self.embed_dir / f"{audio_id}.npz"

        if cache_path.exists() and not force:
            logger.debug(f"Cache hit: {audio_id}")
            data = np.load(cache_path)
            return data["embeddings"], data["starts"], data["ends"]

        logger.info(f"Embedding {audio_id}  ({wav_path})")

        # Load and resample to CLAP_SAMPLE_RATE
        waveform, sr = torchaudio.load(wav_path)  # (C, T)
        if sr != CLAP_SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=CLAP_SAMPLE_RATE)
            waveform = resampler(waveform)
        # Downmix to mono
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        waveform_np = waveform.squeeze(0).numpy()  # (T,) float32

        chunks, starts, ends = self._segment_waveform(waveform_np)
        embeddings = self._batch_embed_audio(chunks)

        np.savez(cache_path, embeddings=embeddings, starts=starts, ends=ends)
        logger.info(f"  → {len(embeddings)} segments cached at {cache_path}")
        return embeddings, starts, ends

    # ── offline bulk embedding ─────────────────────────────────

    def run_embedding_phase(
        self,
        worker_id: int = 0,
        num_workers: int = 1,
        force: bool = False,
    ) -> int:
        """Embed all audio files in the dataset, sharded across workers.

        Each worker processes audio files at indices where index % num_workers == worker_id.
        Already-cached files are skipped unless force=True.

        Returns:
            Number of files newly embedded (cache misses).
        """
        all_files = discover_audio_files(self.config.data_root, self.config.splits)
        shard = [f for i, f in enumerate(all_files) if i % num_workers == worker_id]
        logger.info(
            f"Worker {worker_id}/{num_workers}: {len(shard)} files to process "
            f"({len(all_files)} total)"
        )

        newly_embedded = 0
        for idx, entry in enumerate(shard):
            audio_id = entry["audio_id"]
            cache_path = self.embed_dir / f"{audio_id}.npz"
            if cache_path.exists() and not force:
                logger.debug(f"[{idx+1}/{len(shard)}] Skip (cached): {audio_id}")
                continue
            self.embed_audio_file(entry["wav_path"], audio_id, force=force)
            newly_embedded += 1
            logger.info(f"[{idx+1}/{len(shard)}] Embedded: {audio_id}")

        logger.info(
            f"Worker {worker_id}: done. Newly embedded: {newly_embedded}, "
            f"cached: {len(shard) - newly_embedded}"
        )
        return newly_embedded

    # ── per-audio FAISS index build ────────────────────────────

    def build_faiss_indexes(self, force: bool = False) -> int:
        """Build one FAISS index per audio_id from cached .npz embedding files.

        Should be called after all embedding workers have finished.
        For each {audio_id}.npz writes:
            {cache_dir}/faiss/{audio_id}.index   -- IndexFlatIP (cosine via L2-norm)
            {cache_dir}/faiss/{audio_id}.jsonl   -- segment list [{start, end}, ...]

        Skips audio_ids whose .index file already exists (unless force=True).

        Returns:
            Number of per-audio indexes newly built.
        """
        npz_files = sorted(self.embed_dir.glob("*.npz"))
        if not npz_files:
            raise RuntimeError(f"No cached embeddings found in {self.embed_dir}")

        logger.info(f"Building per-audio FAISS indexes from {len(npz_files)} cached files...")
        built = 0

        for npz_path in npz_files:
            audio_id = npz_path.stem
            index_path = self.faiss_dir / f"{audio_id}.index"
            meta_path = self.faiss_dir / f"{audio_id}.jsonl"

            if index_path.exists() and not force:
                logger.debug(f"Index exists, skipping: {audio_id}")
                continue

            data = np.load(npz_path)
            embs = data["embeddings"].astype(np.float32)  # (N, D)
            starts = data["starts"].tolist()
            ends = data["ends"].tolist()

            embed_dim = embs.shape[1]
            index = faiss.IndexFlatIP(embed_dim)
            index.add(embs)
            faiss.write_index(index, str(index_path))

            with open(meta_path, "w") as f:
                for start, end in zip(starts, ends):
                    f.write(json.dumps({"start": start, "end": end}) + "\n")

            logger.info(f"  {audio_id}: {index.ntotal} segments indexed")
            built += 1

        logger.info(
            f"FAISS build complete. Built: {built}, "
            f"already existed: {len(npz_files) - built}"
        )
        return built

    # ── per-audio FAISS index loading ──────────────────────────

    def load_audio_index(self, audio_id: str) -> tuple[faiss.Index, list[dict]]:
        """Load the FAISS index and segment list for a single audio_id.

        Results are cached in memory so repeated calls are free.

        Returns:
            (faiss_index, segments) where segments is a list of {start, end} dicts.
        """
        if audio_id in self._index_cache:
            return self._index_cache[audio_id]

        index_path = self.faiss_dir / f"{audio_id}.index"
        meta_path = self.faiss_dir / f"{audio_id}.jsonl"

        if not index_path.exists():
            raise FileNotFoundError(
                f"No FAISS index for audio_id '{audio_id}' at {index_path}. "
                "Run build_faiss_indexes() first."
            )

        index = faiss.read_index(str(index_path))
        with open(meta_path) as f:
            segments = [json.loads(line) for line in f]

        self._index_cache[audio_id] = (index, segments)
        return index, segments

    # ── inference-time retrieval ───────────────────────────────

    def _search(self, query_emb: np.ndarray, audio_id: str, k: int) -> list[dict]:
        """Search the index for audio_id and return top-k results."""
        index, segments = self.load_audio_index(audio_id)
        k = min(k, index.ntotal)

        query_emb = query_emb.astype(np.float32).reshape(1, -1)
        query_emb = query_emb / (np.linalg.norm(query_emb) + 1e-8)

        scores, indices = index.search(query_emb, k)
        results = []
        for score, idx in zip(scores[0].tolist(), indices[0].tolist()):
            if idx < 0:
                continue
            row = dict(segments[idx])
            row["audio_id"] = audio_id
            row["score"] = score
            results.append(row)
        return results

    def search_by_audio(
        self,
        query_audio: np.ndarray,
        audio_id: str,
        k: int = 10,
        sampling_rate: int = CLAP_SAMPLE_RATE,
    ) -> list[dict]:
        """Retrieve segments of audio_id most similar to a query audio clip.

        Args:
            query_audio:   1-D float32 waveform of the query clip.
            audio_id:      Which recording to search within.
            k:             Number of results to return.
            sampling_rate: Sampling rate of query_audio (resampled if needed).

        Returns:
            List of dicts: {audio_id, start, end, score}, sorted by descending score.
        """
        self._load_model()

        if sampling_rate != CLAP_SAMPLE_RATE:
            waveform = torch.from_numpy(query_audio).unsqueeze(0)
            resampler = torchaudio.transforms.Resample(
                orig_freq=sampling_rate, new_freq=CLAP_SAMPLE_RATE
            )
            query_audio = resampler(waveform).squeeze(0).numpy()

        inputs = self._processor(
            audios=[query_audio],
            sampling_rate=CLAP_SAMPLE_RATE,
            return_tensors="pt",
        )
        inputs = {kk: v.to(self.config.device) for kk, v in inputs.items()}
        with torch.no_grad():
            query_emb = self._model.get_audio_features(**inputs)
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1).cpu().float().numpy()

        return self._search(query_emb[0], audio_id, k)

    def search_by_text(
        self,
        query_text: str,
        audio_id: str,
        k: int = 10,
    ) -> list[dict]:
        """Retrieve segments of audio_id most similar to a text description.

        Args:
            query_text: Natural-language description of the desired audio content.
            audio_id:   Which recording to search within.
            k:          Number of results to return.

        Returns:
            List of dicts: {audio_id, start, end, score}, sorted by descending score.
        """
        self._load_model()

        inputs = self._processor(
            text=[query_text],
            return_tensors="pt",
            padding=True,
        )
        inputs = {kk: v.to(self.config.device) for kk, v in inputs.items()}
        with torch.no_grad():
            query_emb = self._model.get_text_features(**inputs)
        query_emb = torch.nn.functional.normalize(query_emb, dim=-1).cpu().float().numpy()

        return self._search(query_emb[0], audio_id, k)
