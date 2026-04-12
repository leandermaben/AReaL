"""OpenAI function-calling tool definitions for werewolf game phases.

Each function returns a list of tool dicts in OpenAI's tool format.
All tools include a 'reasoning' field so the agent explains its thinking.
"""


def werewolf_kill_tool(alive_players: list[str]) -> list[dict]:
    """Tool for werewolf night kill action."""
    return [
        {
            "type": "function",
            "function": {
                "name": "werewolf_kill",
                "description": (
                    "Choose a player to target for elimination tonight. "
                    "Analyze high-value roles, guard protection likelihood, "
                    "and risks before deciding. Include a message to your "
                    "fellow werewolves explaining your choice."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your analysis: who might be high-value roles, "
                                "who the guard might protect, risks of each target."
                            ),
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "A message to your fellow werewolves explaining "
                                "your choice and reasoning (they will see this)."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                f"Player to target. Must be one of: "
                                f"{', '.join(alive_players)}. "
                                f"Or 'skip' to not attack."
                            ),
                        },
                    },
                    "required": ["reasoning", "message", "target"],
                },
            },
        }
    ]


def werewolf_discuss_tool(alive_players: list[str]) -> list[dict]:
    """Tool for werewolf consensus discussion rounds."""
    return [
        {
            "type": "function",
            "function": {
                "name": "werewolf_discuss",
                "description": (
                    "After seeing other werewolves' choices and messages, "
                    "revise your target to reach consensus. Include a message "
                    "to your fellow werewolves."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your analysis of allies' choices and your "
                                "updated reasoning for target selection."
                            ),
                        },
                        "message": {
                            "type": "string",
                            "description": (
                                "A message to your fellow werewolves about your "
                                "revised choice and reasoning (they will see this)."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                f"Your revised target. Must be one of: "
                                f"{', '.join(alive_players)}. "
                                f"Or 'skip' to not attack."
                            ),
                        },
                    },
                    "required": ["reasoning", "message", "target"],
                },
            },
        }
    ]


def guard_protect_tool(
    alive_players: list[str], last_protected: str | None
) -> list[dict]:
    """Tool for guard night protection."""
    cannot_protect = (
        f" You cannot protect {last_protected} again (protected last night)."
        if last_protected
        else ""
    )
    return [
        {
            "type": "function",
            "function": {
                "name": "guard_protect",
                "description": (
                    "Choose a player to protect from werewolf attack tonight."
                    + cannot_protect
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your reasoning on who the werewolves will likely "
                                "target and why you chose this player to protect."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                f"Player to protect. Must be one of: "
                                f"{', '.join(alive_players)}."
                            ),
                        },
                    },
                    "required": ["reasoning", "target"],
                },
            },
        }
    ]


def seer_check_tool(alive_players: list[str]) -> list[dict]:
    """Tool for seer night check."""
    return [
        {
            "type": "function",
            "function": {
                "name": "seer_check",
                "description": (
                    "Choose a player to check tonight. You will learn if "
                    "they are a werewolf or not."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your reasoning for checking this player based on "
                                "their behavior and your current knowledge."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                f"Player to check. Must be one of: "
                                f"{', '.join(alive_players)}."
                            ),
                        },
                    },
                    "required": ["reasoning", "target"],
                },
            },
        }
    ]


def witch_action_tool(
    alive_players: list[str],
    killed_player: str | None,
    has_antidote: bool,
    has_poison: bool,
) -> list[dict]:
    """Tool for witch night action."""
    desc_parts = ["Decide whether to use your potions tonight."]
    if killed_player:
        desc_parts.append(f"{killed_player} was killed by werewolves tonight.")
    else:
        desc_parts.append("No one was killed by werewolves tonight.")
    if has_antidote:
        desc_parts.append("You have 1 antidote (can save the killed player).")
    else:
        desc_parts.append("You have no antidote remaining.")
    if has_poison:
        desc_parts.append("You have 1 poison (can kill a player).")
    else:
        desc_parts.append("You have no poison remaining.")
    desc_parts.append("You cannot use both antidote and poison in the same night.")

    return [
        {
            "type": "function",
            "function": {
                "name": "witch_action",
                "description": " ".join(desc_parts),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your reasoning about whether to use antidote "
                                "or poison, and on whom."
                            ),
                        },
                        "use_antidote": {
                            "type": "boolean",
                            "description": (
                                "Whether to use the antidote to save the killed player."
                            ),
                        },
                        "use_poison": {
                            "type": "boolean",
                            "description": "Whether to use the poison to kill a player.",
                        },
                        "poison_target": {
                            "type": "string",
                            "description": (
                                f"If using poison, the target player. "
                                f"Must be one of: {', '.join(alive_players)}. "
                                f"Set to 'none' if not using poison."
                            ),
                        },
                    },
                    "required": [
                        "reasoning",
                        "use_antidote",
                        "use_poison",
                        "poison_target",
                    ],
                },
            },
        }
    ]


def run_for_sheriff_tool() -> list[dict]:
    """Tool for sheriff election decision."""
    return [
        {
            "type": "function",
            "function": {
                "name": "run_for_sheriff",
                "description": (
                    "Decide whether to run for sheriff. The sheriff has 1.5x "
                    "vote weight and decides speaking order."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your reasoning about whether running for sheriff "
                                "aligns with your role and strategy."
                            ),
                        },
                        "run": {
                            "type": "boolean",
                            "description": "True to run for sheriff, False to abstain.",
                        },
                    },
                    "required": ["reasoning", "run"],
                },
            },
        }
    ]


def sheriff_speech_tool() -> list[dict]:
    """Tool for sheriff campaign speech."""
    return [
        {
            "type": "function",
            "function": {
                "name": "sheriff_speech",
                "description": (
                    "Give your campaign speech for the sheriff election. "
                    "You may also choose to withdraw."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": "Your analysis of other candidates and your strategy.",
                        },
                        "continue_running": {
                            "type": "boolean",
                            "description": "True to stay in the election, False to withdraw.",
                        },
                        "speech_content": {
                            "type": "string",
                            "description": "Your campaign speech content.",
                        },
                    },
                    "required": ["reasoning", "continue_running", "speech_content"],
                },
            },
        }
    ]


def vote_for_sheriff_tool(candidates: list[str]) -> list[dict]:
    """Tool for voting in sheriff election."""
    return [
        {
            "type": "function",
            "function": {
                "name": "vote_for_sheriff",
                "description": "Vote for a candidate in the sheriff election.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your analysis of the candidates and why you're "
                                "voting for your choice."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                f"The candidate you vote for. Must be one of: "
                                f"{', '.join(candidates)}. "
                                f"Or 'abstain'."
                            ),
                        },
                    },
                    "required": ["reasoning", "target"],
                },
            },
        }
    ]


def speech_tool() -> list[dict]:
    """Tool for day phase speech."""
    return [
        {
            "type": "function",
            "function": {
                "name": "speak",
                "description": (
                    "Give your speech during the day phase. Share observations, "
                    "suspicions, defend yourself, or strategize."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your private analysis: role objectives, who you "
                                "suspect, your strategy for this speech."
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "Your public speech content that all players will hear.",
                        },
                    },
                    "required": ["reasoning", "content"],
                },
            },
        }
    ]


def vote_tool(alive_players: list[str]) -> list[dict]:
    """Tool for day phase banishment vote."""
    return [
        {
            "type": "function",
            "function": {
                "name": "vote",
                "description": "Vote to banish a player from the game.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your analysis of suspects, public perception, "
                                "and strategic considerations for your vote."
                            ),
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                f"Player to vote to banish. Must be one of: "
                                f"{', '.join(alive_players)}. "
                                f"Or 'abstain'."
                            ),
                        },
                    },
                    "required": ["reasoning", "target"],
                },
            },
        }
    ]


def last_words_tool() -> list[dict]:
    """Tool for eliminated player's last words."""
    return [
        {
            "type": "function",
            "function": {
                "name": "last_words",
                "description": (
                    "Give your final words after being eliminated. "
                    "Share suspicions or advice to help your team."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your private analysis of who the werewolves are "
                                "and what message will best help your team."
                            ),
                        },
                        "content": {
                            "type": "string",
                            "description": "Your final words that all players will hear.",
                        },
                    },
                    "required": ["reasoning", "content"],
                },
            },
        }
    ]


def badge_flow_tool(alive_players: list[str]) -> list[dict]:
    """Tool for sheriff badge transfer when sheriff is eliminated."""
    return [
        {
            "type": "function",
            "function": {
                "name": "badge_flow",
                "description": (
                    "As the eliminated sheriff, decide whether to pass "
                    "the badge to another player or destroy it."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your analysis of who you trust and whether "
                                "passing or destroying the badge is better."
                            ),
                        },
                        "pass_badge": {
                            "type": "boolean",
                            "description": "True to pass the badge, False to destroy it.",
                        },
                        "target": {
                            "type": "string",
                            "description": (
                                f"Player to pass the badge to (if passing). "
                                f"Must be one of: {', '.join(alive_players)}. "
                                f"Set to 'none' if destroying."
                            ),
                        },
                    },
                    "required": ["reasoning", "pass_badge", "target"],
                },
            },
        }
    ]


def decide_speech_order_tool(alive_players: list[str]) -> list[dict]:
    """Tool for sheriff to decide speaking order."""
    return [
        {
            "type": "function",
            "function": {
                "name": "decide_speech_order",
                "description": (
                    "As sheriff, decide the speaking order for this round. "
                    "Choose a starting player and direction."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "reasoning": {
                            "type": "string",
                            "description": (
                                "Your strategy for the speaking order — who should "
                                "speak first/last and why."
                            ),
                        },
                        "starting_player": {
                            "type": "string",
                            "description": (
                                f"Player to start speaking from. "
                                f"Must be one of: {', '.join(alive_players)}."
                            ),
                        },
                        "reverse": {
                            "type": "boolean",
                            "description": (
                                "True for reverse order (right to left), "
                                "False for normal order (left to right)."
                            ),
                        },
                    },
                    "required": ["reasoning", "starting_player", "reverse"],
                },
            },
        }
    ]
