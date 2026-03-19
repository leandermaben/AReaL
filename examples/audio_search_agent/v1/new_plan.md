# Audio Search Agent

Long story short - Audio -> Search Agent Searches for relevant snippets and submits -> Answering system

The Search Agent will be powered by Qwen3 family (atleast initially). Let's start with Qwen3-4B-Instruct-2507 and then try to move to smaller models.

## Planned Tools

1. CLAP over the audios -> FAISS store (Should be done as pre-processing) ->Retriver
2. Transcript Probe -> An ASR model that will be run on the selected stretch  of Audio (Qwen3-ASR model likely)
3. Omni Probe -> Selected Snippet of audio can be passed to Qwen Omni with a Specific Query
4. Non Model based tools (still not decided) but here are some examples:
    speech_activity(span): VAD % / speech ratio (simple energy-based)
    silence_peaks() / change_points(): find long pauses / energy shifts to propose segment boundaries
    loudness_peaks() / event_peaks(): spikes that often correspond to salient events
    repeat_detector(): near-duplicate detection using existing CLAP embeddings (no new model) to find repeated segments
    temporal_expand(span, left, right): deterministic expand/contract windows
    window_sampler(strategy): uniform / stratified sampling (early-mid-late, high-speech-density, high-change)
    cluster_summary(): cluster CLAP embeddings (k-means offline) and return “top clusters / timestamps” for navigation
5. The PostGres based ASR+Emotion+Diarized+Event Detected
    Notes about this tool:
        a. There are other people working to improve this and replace it with dense retrieval and other things
        b. We will fork the research topics to have 2 separate research papers. 1. where there is complete pre-processing with multiple models and one without where the agent uses the CLAP based soft matching and probes mentioned above to find the snippet.
6. Potentially a memory compress to manage context.
6. Submit Tool: Actually Submit the final list snippets-
    Each snippet should have the start and end times specified, it can potentially also have the transcript and feedback from the Omni tool if available.
    Beyond this there can be a metadata field which can have the metadata or relevant rows from tool 5 if it is enabled.


## RL
The primary reward will be precision and recall.
Secondary rewards can be related to number of turns, different tools can have costs, the length of snippets passed to tools can have cost,
the rows requested from database can have cost etc.
(Again there will be a fork in research questions so we can make each fork more detailed accordingly later)

Let's start solving phase by phase, I will update this document with new phases as we move on

## Phase 1
Let us implement CLAP as a pre-processing pipeline. Then do clap for all audios.
Ingest this into FAISS all of this can be done in the same ipeline if approprite plan accordingly.

Things to keep in mind:
1. Make sure to cache it (with appropriate audio id) so that whenever we re-run this pre-processing step, we can reuse the cache and only ingest the new data only.
2. The python script should support both offline pre-processing of all data as well as "inference time style" single input processing too (provide functionality that can be called later if needed).
3. If you can parallelize with multi-GPU, enable that function
4. Write sbatch script to launch bulk pre-processing. For reference see this: /u/lmaben/speech/long_speech/LongAudioUnderstandingSystem/agent_search/data_prep/meeting_bank/launch/prepare_meetingbank.sbatch. Decide how much GPU power is needed,.I think it is faster to get allocated 2 A100:40GB if that's enough. 


Other details: 
The data for preprocessing is here:/work/hdd/bbjs/lmaben/speech/long_speech/meeting_bank_prepared 
This is the meeting bank data. Let us start with this, in future there will be other datasets to ingest too. You can assume those will also have a similar format of (wav,json) pairs.
Save your cache in an appropriately created dir here: /work/hdd/bbjs/lmaben/speech/long_speech/
For info on the v0 version see the parent dir and /u/lmaben/speech/long_speech/AReaL/examples/audio_search_agent/desc_v0.md
For info on the dataset see: /u/lmaben/speech/long_speech/LongAudioUnderstandingSystem/agent_search/data_prep/meeting_bank/DATASET.md
I think this stage can go in /u/lmaben/speech/long_speech/AReaL/examples/audio_search_agent/v1/preprocessing but if you have a cleaner structure go for it.


#Update 1:
- I have done Proof of concept that Qwen ASR works, but for now to keep things simple let us assume there is one module powered by Qwen-Omni for audio understanding.
- So for the first version, we can have 2 main tools -> CLAP based soft matching and Omni powered audio understanding (the agent can sen file path, start time, end time and a question).
- Qwen-Omni3 installation is currently going on so let us build the interface for the soft grep tool first.

I was planning to have a sort of registry design and we register tools. But if there's a cleaner design please do that. Also since there will only be a max of 5 tools probably, if this is ennecessary, let me know.
I was thinking tools can go here: /u/lmaben/speech/long_speech/AReaL/examples/audio_search_agent/v1/agent/tools
So first why don't you create the clap tool ( we have already created the embeddings and index last time, now make it available for an agent to use)
Once the Omni-installation is sorted,we can move to that.
