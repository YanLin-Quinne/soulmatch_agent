#!/usr/bin/env python3
"""Generate evaluation dataset for baseline experiments.

Uses bot_personas.json as ground truth personality profiles.
For each bot, generates synthetic conversations via LLMs,
where the persona features act as ground truth labels.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.agents.llm_router import AgentRole, router


DEFAULT_OUTPUT_PATH = Path(__file__).parent.parent / "data" / "evaluation" / "eval_dataset.json"

USER_PERSONAS = [
    {
        "style": "curious_explorer",
        "system": (
            "You are a curious, open-minded person who asks lots of questions "
            "about the other person's interests and experiences. You share your "
            "own adventures too."
        ),
    },
    {
        "style": "reserved_intellectual",
        "system": (
            "You are a thoughtful, reserved person. You prefer deep "
            "conversations over small talk. You share opinions on books, "
            "ideas, and values."
        ),
    },
    {
        "style": "playful_flirt",
        "system": (
            "You are a playful, witty person. You use humor and light teasing. "
            "You're confident but not aggressive."
        ),
    },
    {
        "style": "anxious_sincere",
        "system": (
            "You are a sincere person who sometimes overthinks. You're genuine "
            "and caring but occasionally self-deprecating. You value emotional honesty."
        ),
    },
    {
        "style": "direct_practical",
        "system": (
            "You are a direct, no-nonsense person. You value efficiency and honesty. "
            "You talk about goals, plans, and practical matters."
        ),
    },
]


def load_bot_personas(path: Optional[str] = None, max_bots: Optional[int] = None) -> List[Dict]:
    """Load bot personas with ground truth features."""
    if path is None:
        path = str(Path(__file__).parent.parent / "data" / "processed" / "bot_personas.json")

    with open(path) as f:
        personas = json.load(f)

    if max_bots is not None:
        return personas[:max_bots]
    return personas


def parse_turn_counts(raw: str) -> List[int]:
    """Parse comma-separated turn counts into a validated list."""
    turn_counts = []
    for chunk in raw.split(","):
        value = chunk.strip()
        if not value:
            continue
        turns = int(value)
        if turns <= 0:
            raise ValueError("Turn counts must be positive integers.")
        turn_counts.append(turns)

    if not turn_counts:
        raise ValueError("At least one turn count is required.")
    return turn_counts


def _build_user_prompt(user_persona: Dict, conversation: List[Dict], is_opener: bool) -> str:
    """Build the next-user-message prompt from recent conversation context."""
    if is_opener:
        return (
            "You are using a dating app. Start a natural conversation with one message "
            "(1-3 sentences). Match your persona naturally and avoid generic filler."
        )

    conv_context = "\n".join(
        f"{'User' if message['speaker'] == 'user' else 'Bot'}: {message['message']}"
        for message in conversation[-6:]
    )
    return (
        "Continue this dating-app conversation with exactly one natural user message "
        "(1-3 sentences). Stay consistent with your persona.\n\n"
        f"Persona style: {user_persona['style']}\n"
        f"Previous messages:\n{conv_context}\n\n"
        "Write the next User message only."
    )


def _build_bot_prompt(conversation: List[Dict]) -> str:
    """Build the next-bot-message prompt from recent conversation context."""
    conv_context = "\n".join(
        f"{'User' if message['speaker'] == 'user' else 'You'}: {message['message']}"
        for message in conversation[-6:]
    )
    return (
        "Continue this dating-app conversation naturally with exactly one reply "
        "(1-3 sentences).\n\n"
        f"Previous messages:\n{conv_context}\n\n"
        "Write your next message only."
    )


def generate_conversation(
    bot_persona: Dict,
    user_persona: Dict,
    n_turns: int = 10,
) -> List[Dict]:
    """Generate a synthetic conversation between a user and the bot.

    Args:
        bot_persona: Persona definition for the bot.
        user_persona: Persona definition for the synthetic user.
        n_turns: Total number of dialogue entries to generate.
    """
    conversation: List[Dict] = []

    for turn_index in range(n_turns):
        if turn_index % 2 == 0:
            user_msg = router.chat(
                role=AgentRole.GENERAL,
                system=(
                    f"{user_persona['system']} "
                    "Write casual, natural dating-app messages. Return only one message."
                ),
                messages=[{"role": "user", "content": _build_user_prompt(user_persona, conversation, not conversation)}],
                temperature=0.8,
                max_tokens=100,
            )
            conversation.append({"speaker": "user", "message": user_msg.strip()})
            continue

        bot_msg = router.chat(
            role=AgentRole.PERSONA,
            system=bot_persona["system_prompt"],
            messages=[{"role": "user", "content": _build_bot_prompt(conversation)}],
            temperature=0.7,
            max_tokens=150,
        )
        conversation.append({"speaker": "bot", "message": bot_msg.strip()})

    return conversation


def extract_ground_truth(bot_persona: Dict) -> Dict:
    """Extract ground truth labels from bot persona features."""
    features = bot_persona.get("features", {})
    profile = bot_persona.get("original_profile", {})

    return {
        "personality": {
            "big_five": {
                "openness": features.get("big_five_openness"),
                "conscientiousness": features.get("big_five_conscientiousness"),
                "extraversion": features.get("big_five_extraversion"),
                "agreeableness": features.get("big_five_agreeableness"),
                "neuroticism": features.get("big_five_neuroticism"),
            },
            "mbti": features.get("mbti"),
        },
        "demographics": {
            "age": profile.get("age"),
            "sex": profile.get("sex"),
            "job": profile.get("job"),
        },
        "relationship": {
            "rel_type": "love",
            "rel_status": "acquaintance",
        },
    }


def generate_eval_dataset_for_personas(
    personas: List[Dict],
    n_conversations_per_bot: int,
    turn_counts: List[int],
    output_path: Optional[str],
) -> List[Dict]:
    """Internal dataset generation helper for a concrete persona list."""
    if output_path is None:
        output_path = str(DEFAULT_OUTPUT_PATH)

    dataset: List[Dict] = []
    total_jobs = (
        len(personas)
        * len(USER_PERSONAS)
        * len(turn_counts)
        * n_conversations_per_bot
    )
    completed_jobs = 0

    for persona in personas:
        ground_truth = extract_ground_truth(persona)

        for user_persona in USER_PERSONAS:
            for target_turns in turn_counts:
                for conv_idx in range(n_conversations_per_bot):
                    completed_jobs += 1
                    percent_complete = (completed_jobs / total_jobs) * 100 if total_jobs else 100.0
                    print(
                        f"[{completed_jobs}/{total_jobs}] {percent_complete:6.2f}% "
                        f"{persona['profile_id']} | {user_persona['style']} | "
                        f"target_turns={target_turns} | sample={conv_idx + 1}/{n_conversations_per_bot}"
                    )

                    try:
                        conversation = generate_conversation(
                            persona,
                            user_persona=user_persona,
                            n_turns=target_turns,
                        )
                    except Exception as exc:
                        print(f"  Error generating conversation: {exc}")
                        continue

                    sample = {
                        "conversation_id": (
                            f"{persona['profile_id']}_"
                            f"{user_persona['style']}_"
                            f"t{target_turns}_"
                            f"conv{conv_idx}"
                        ),
                        "bot_id": persona["profile_id"],
                        "user_persona": user_persona["style"],
                        "dialogue": conversation,
                        "ground_truth": ground_truth,
                        "n_turns": len(conversation),
                        "target_turns": target_turns,
                    }
                    dataset.append(sample)

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to {output_file}")
    print(f"Total samples: {len(dataset)}")
    return dataset


def generate_eval_dataset(
    n_conversations_per_bot: int = 3,
    turn_counts: List[int] = [10, 20, 30],
    output_path: Optional[str] = None,
) -> List[Dict]:
    """Generate the full evaluation dataset.

    Args:
        n_conversations_per_bot: replicate conversations per bot/persona/turn-count cell
        turn_counts: target dialogue lengths measured in total dialogue entries
        output_path: where to save the dataset JSON
    """
    personas = load_bot_personas()
    return generate_eval_dataset_for_personas(
        personas=personas,
        n_conversations_per_bot=n_conversations_per_bot,
        turn_counts=turn_counts,
        output_path=output_path,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate evaluation dataset")
    parser.add_argument(
        "--n-per-bot",
        type=int,
        default=3,
        help="Conversation replicates per bot/persona/turn-count combination",
    )
    parser.add_argument(
        "--turns",
        type=str,
        default="10,20,30",
        help="Comma-separated target dialogue lengths measured in total dialogue entries",
    )
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument(
        "--max-bots",
        type=int,
        default=None,
        help="Limit number of bot personas for quick testing",
    )
    args = parser.parse_args()

    personas = load_bot_personas(max_bots=args.max_bots)
    turn_counts = parse_turn_counts(args.turns)

    generate_eval_dataset_for_personas(
        personas=personas,
        n_conversations_per_bot=args.n_per_bot,
        turn_counts=turn_counts,
        output_path=args.output,
    )
