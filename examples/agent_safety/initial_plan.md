# Research Plan: Deceptive Strategy Generalization in Multi-Agent RL

## Core Research Question

Does reinforcement learning in competitive social environments that require deception (e.g., Werewolf) induce **deceptive behaviors that generalize outside the game context**?

Key distinction:

| Context | Expected Behavior |
|-------|------|
| In-game | Deception may be role-appropriate |
| Outside game | Deception may indicate misalignment |

The goal is **not to prevent deception in-game**, but to test whether **policies trained for competitive success learn reusable deceptive capabilities**.

---

# Environment

## Primary Environment
**MultiAgentBench – Werewolf**

Reasons:
- Hidden roles with asymmetric information
- Public speeches and private coordination
- Multi-round social reasoning
- Institutional dynamics (Sheriff election, badge passing)
- Rich role structure: Werewolf, Villager, Seer, Witch, Guard, Sheriff

This environment supports:
- coalition formation
- persuasion
- reputation management
- deception strategies

---

# High-Level Experimental Plan

## Phase 1 — Baseline Characterization

Run games **without RL training**.

Measure:
- win rate
- deception frequency
- contradictions / misinformation
- persuasion effectiveness
- speech and voting dynamics
- sheriff election behavior

Goal:
Understand baseline strategic behavior before optimization.

---

## Phase 2 — RL Training

Train LLM agents to **maximize win rate in Werewolf**.

### Main Setup
Trainable shared policy across **all werewolf seats**.

Trainable LLM → all Werewolves
Fixed LLMs → Villagers + special roles


This allows:
- coordinated deception
- private communication strategy learning
- stronger RL signal

### Key Ablation
Single trainable werewolf seat.


Trainable LLM → one Werewolf
Fixed LLMs → other Werewolves + Villagers


Purpose:
Distinguish between:

| Learning Type | Meaning |
|---|---|
| Individual deception | single-agent lying strategy |
| Coordinated deception | multi-agent communication strategy |

---

## Phase 3 — Transfer Evaluation

After RL training, evaluate the trained model **outside the game environment**.

### Evaluation Domains

| Evaluation Type | Goal |
|---|---|
| Truthfulness benchmarks | detect increased falsehoods |
| Honesty scenarios | detect strategic lying |
| Persuasion tasks | measure manipulation ability |
| Controlled deception prompts | measure willingness to mislead |

Possible benchmarks:
- TruthfulQA
- persuasion / deception datasets
- custom honesty probes

---

## Phase 4 — Additional Domains (Future)

Evaluate generalization in other competitive communication environments.

Possible environments:

| Domain | Purpose |
|---|---|
| Debate | rhetorical persuasion |
| Bargaining / negotiation | strategic misrepresentation |
| Additional MultiAgentBench tasks | broader social strategy |

Goal:
Test whether deceptive capability transfers across **different social interaction domains**.

---

# Experimental Matrix

## A. Training Setup

| ID | Trainable Policy | Fixed Policies | Purpose |
|---|---|---|---|
| A0 | None | All fixed | Baseline |
| A1 | One werewolf seat | Others fixed | Individual deception |
| A2 | Shared werewolf policy | Villagers fixed | Coordinated deception |
| A3 | One villager seat | Others fixed | Detection specialization |
| A4 | Shared villager policy | Werewolves fixed | Suspicion strategy |

Primary condition: **A2**

---

## B. Role Assignment

| ID | Role Training | Purpose |
|---|---|---|
| B1 | Always werewolf | maximize deception pressure |
| B2 | Always villager | detection-oriented training |
| B3 | Mixed roles | role-conditioned strategies |

Primary condition: **B1**

---

## C. Communication Structure

| ID | Communication Access | Hypothesis |
|---|---|---|
| C1 | Public + private | strongest coordination |
| C2 | Public only | weaker deception |
| C3 | Reduced private rounds | reduced planning |
| C4 | Speech token limits | reduced persuasion complexity |

---

## D. Institutional Structure

| ID | Structure | Hypothesis |
|---|---|---|
| D1 | Full game (Sheriff) | authority manipulation |
| D2 | No Sheriff election | weaker leadership dynamics |
| D3 | Simplified Sheriff | isolate influence effects |

---

## E. Memory Variants (Optional)

| ID | Memory | Hypothesis |
|---|---|---|
| E1 | Stateless | tactical deception |
| E2 | Round history | strategic manipulation |
| E3 | belief summaries | theory-of-mind strategies |

---

# Key Hypotheses

### H1
RL training for competitive success in deception-required environments increases deceptive behavior **in-game**.

### H2
Deceptive strategies learned in-game **transfer to out-of-game tasks**.

### H3
Transfer is stronger when:
- agents learn **coordinated deception**
- environments allow **private planning**
- environments include **institutional power dynamics**.

---

# Long-Term Direction

Future work will explore **alignment strategies that preserve in-game deception while preventing out-of-context misuse**, including:

- role-conditioned prompting
- multi-objective RL
- post-training honesty alignment
- context-gated policies

Goal:
Enable **context-bounded deceptive capability** rather than global policy changes.

---