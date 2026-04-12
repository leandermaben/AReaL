"""Generate werewolf game config datasets for RL training.

Each row specifies a game setup: player count, role distribution,
per-agent LLM assignments, and shuffled player names. The trainable
wolf (Qwen3-4B) occupies one werewolf slot; every other agent gets
a randomly chosen hosted LLM.

Usage:
    python -m examples.agent_safety.rl.dataset
"""

import json
import random
from pathlib import Path

LLM_DIR = Path(__file__).parent.parent / "llm"
HOSTED_LLMS = json.loads((LLM_DIR / "hosted_llms.json").read_text())["models"]
HOSTED_KEYS = json.loads((LLM_DIR / "hosted_llm_keys.json").read_text())

# Exclude broken / expensive / reasoning-heavy models
EXCLUDED_MODELS = {
    "claude-3-5-sonnet-20241022",  # retired model ID
    "gpt-5",                        # tool calling broken
    "gpt-5-mini",                   # reasoning model, slow
    "gpt-5.4-nano",                 # reasoning model, slow
    "claude-opus-4-20250514-v1:0",  # very expensive
}

AVAILABLE_MODELS = [m for m in HOSTED_LLMS if m["model"] not in EXCLUDED_MODELS]

# Name pool for player name randomization
NAME_POOL = [
    "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry",
    "Iris", "Jack", "Kate", "Leo", "Mia", "Noah", "Olivia", "Paul",
    "Quinn", "Rosa", "Sam", "Tara", "Uma", "Vic", "Wendy", "Xander",
    "Yuki", "Zara",
]


def sample_model(rng: random.Random) -> dict:
    """Sample an opponent model uniformly at random."""
    return rng.choice(AVAILABLE_MODELS)


def _resolve_key(entry: dict) -> str:
    return HOSTED_KEYS.get(entry["api_key"], entry["api_key"])


def _model_info(entry: dict) -> dict:
    """Extract the fields needed to create an AsyncOpenAI client."""
    extra_body = {}
    if entry.get("reasoning") and isinstance(entry.get("reasoning_disable"), dict):
        extra_body = entry["reasoning_disable"]
    return {
        "model": entry["model"],
        "base_url": entry["base_url"],
        "api_key": _resolve_key(entry),
        "extra_body": extra_body,
    }


def generate_game_config(rng: random.Random, game_id: int) -> dict:
    """Generate a single game config.

    Role distribution:
    - num_villagers (including specials): 3–8
    - num_wolves: 1 to floor(num_villagers / 2)
    - specials: seer always, witch if room, guard if room
    - one wolf is always the trainable agent

    Each non-trainable agent gets an independently sampled LLM.
    Player names are shuffled from NAME_POOL.
    """
    # Total villager-side players (including specials)
    num_villagers = rng.randint(3, 8)
    max_wolves = num_villagers // 2
    num_wolves = rng.randint(1, max(1, max_wolves))

    total_players = num_wolves + num_villagers

    # Build roles
    roles = ["werewolf"] * num_wolves

    remaining = num_villagers
    for special in ["seer", "witch", "guard"]:
        if remaining > 0:
            roles.append(special)
            remaining -= 1
    roles += ["villager"] * remaining

    # Shuffle names
    names = rng.sample(NAME_POOL, total_players)

    # Assign LLMs: one wolf is trainable, everything else is random
    # Pick which wolf slot is trainable (by index among wolves)
    trainable_wolf_idx = rng.randint(0, num_wolves - 1)

    agents = []
    wolf_count = 0
    for i, role in enumerate(roles):
        if role == "werewolf":
            if wolf_count == trainable_wolf_idx:
                agents.append({
                    "name": names[i],
                    "role": role,
                    "trainable": True,
                })
            else:
                model = sample_model(rng)
                agents.append({
                    "name": names[i],
                    "role": role,
                    "trainable": False,
                    **_model_info(model),
                })
            wolf_count += 1
        else:
            model = sample_model(rng)
            agents.append({
                "name": names[i],
                "role": role,
                "trainable": False,
                **_model_info(model),
            })

    return {
        "game_id": game_id,
        "num_wolves": num_wolves,
        "num_villagers": num_villagers,
        "agents": agents,
        # AReaL dataset contract: needs a "messages" field
        "messages": [
            {
                "role": "system",
                "content": (
                    f"Werewolf game {game_id}: "
                    f"{num_wolves} wolves vs {num_villagers} villagers."
                ),
            }
        ],
    }


def generate_dataset(n: int, seed: int = 42) -> list[dict]:
    rng = random.Random(seed)
    return [generate_game_config(rng, i) for i in range(n)]


def main():
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    train = generate_dataset(n=500, seed=42)
    valid = generate_dataset(n=20, seed=999)

    for name, data in [("train", train), ("valid", valid)]:
        path = out_dir / f"werewolf_{name}.jsonl"
        with open(path, "w") as f:
            for row in data:
                f.write(json.dumps(row) + "\n")
        print(f"Wrote {len(data)} configs to {path}")

    # Print stats
    from collections import Counter

    wolf_counts = Counter(r["num_wolves"] for r in train)
    vill_counts = Counter(r["num_villagers"] for r in train)
    print(f"\nWolf count distribution (train): {dict(sorted(wolf_counts.items()))}")
    print(f"Villager count distribution (train): {dict(sorted(vill_counts.items()))}")

    all_models = []
    for r in train:
        for a in r["agents"]:
            if not a["trainable"]:
                all_models.append(a["model"])
    model_counts = Counter(all_models)
    print(f"\nOpponent model distribution (train, {len(all_models)} total assignments):")
    for model, count in model_counts.most_common():
        print(f"  {model:45s} {count:5d} ({count/len(all_models)*100:.1f}%)")


if __name__ == "__main__":
    main()
