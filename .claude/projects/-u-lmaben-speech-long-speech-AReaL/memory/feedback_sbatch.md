---
name: feedback_sbatch
description: Delta cluster sbatch scripts need ffmpeg module and LD_LIBRARY_PATH for torchaudio
type: feedback
---

On the Delta cluster, sbatch scripts for audio processing need:
```bash
module load ffmpeg
export LD_LIBRARY_PATH=/sw/rh9.4/spack/v1.0.0/sw/linux-x86_64_v2/ffmpeg-7.1-3avnbo4/lib:$LD_LIBRARY_PATH
```

**Why:** torchaudio requires ffmpeg shared libraries at runtime; they're not in the default environment.

**How to apply:** Always include these lines in sbatch scripts that process audio files (before the Python command). The user already added this to `prepare_clap_index.sbatch`.
