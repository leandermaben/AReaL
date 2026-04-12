# Agent Safety

This is an Agent Safety Project.

There will be multiple stages and TODOs.

The first phase is this:
1. Setup Agent
2. Setup Scenarios
3. Setup RL training to optimize for scenarios.

## Setting up Scenarios:
1. We may potentially set up other scenarios but the first one we will setup is the werewolf scenario.

## Agent
1. We will have an agent defined (independent of AReaL) usinng Asunc Openai calls.
2. This agent will participate in a particular scenario (mostly multi-agent environment).

## RL 
1. Atleast in the first phase of RL we will optimize for the particular rewards for the scenario.


Notes:
1. Werewolf implementation will be according to the werewolf flow defined in /u/lmaben/agent_safety/MARBLE
    - Main differenece is that we do not want to have separate prompts for a given role - We can have different prompts for werewolf, villager, seer etc. but we do not want to have different prompts when a werewolf is voting for sherrif, vs voting to kill etc. I think this is better for RL training and as a general agent.
    - A good idea would be to have a full description of rules goals objectives of the game etc. Then role specific prompt.
    - Then the agent can have a multi-turn concatenated trajectory.. the user prompts can specify what happens in the environment wrt other players. The agent can perform actions as tool calls etc.
2. A prior implementation of rl for werewolf is here /u/lmaben/agent_safety/AReaL/examples/agent_safety/rl_v0 but let us have a clean refactored version now.
3. A working rl implementation is here for a different application for reference: /u/lmaben/speech/long_speech/AReaL/examples/audio_search_agent/v1 and an example working config for 4 GPU with GSPO is here: /u/lmaben/agent_safety/AReaL/examples/agent_safety/sample_config_4gpu.yaml


Let us begin by setting up the werewolf scenario and the agent to play this game. As a futuere reference for RL (no need to implement now), one of these agents will be traininable and other will potentially use a frontier model. But later maybe more than one agent may be defined as traininable in the workflow.

Determine the best way to organise scenario and agent given this info information, feel free to ask clarifying questions. There may be multiple scenarios that may be added later but for now its only werewolf (maybe keep this in mind). 

Write test cases and run them when designing scenarios. Organise things on well structured directiories in /u/lmaben/agent_safety/AReaL/examples/agent_safety.

For testing purposes you can use a low cost llm like gpt-4.1-mini as defined here: /u/lmaben/agent_safety/AReaL/examples/agent_safety/llm

For now focus on Scenario and Agent Setup, we can do RL in next steps.