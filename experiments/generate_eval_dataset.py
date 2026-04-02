"""Generate evaluation dataset for baseline experiments.

Uses bot_personas.json as ground truth personality profiles.
For each bot, generates synthetic conversations via LLM,
where we know the true personality traits (from the persona features).
"""
import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.agents.llm_router import router, AgentRole


def load_bot_personas(path: Optional[str] = None) -> List[Dict]:
    """Load bot personas with ground truth features."""
    if path is None:
        path = str(Path(__file__).parent.parent / "data" / "processed" / "bot_personas.json")
    with open(path) as f:
        return json.load(f)


def generate_conversation(bot_persona: Dict, n_turns: int = 10) -> List[Dict]:
    """Generate a synthetic conversation between a user and the bot.

    Uses the bot's system_prompt to generate realistic bot responses,
    and a generic user personality to generate user messages.
    """
    conversation = []

    # User starts the conversation
    user_opener_prompt = """You are a person using a dating app. Start a casual conversation.
    Just write one message (1-3 sentences). Be natural and friendly."""

    for turn in range(n_turns):
        # Generate user message
        if turn == 0:
            user_msg = router.chat(
                role=AgentRole.GENERAL,
                system="You are a friendly person on a dating app. Write casual, natural messages.",
                messages=[{"role": "user", "content": user_opener_prompt}],
                temperature=0.8,
                max_tokens=100,
            )
        else:
            # User responds to bot's last message
            conv_context = "\n".join([
                f"{'User' if m['speaker'] == 'user' else 'Bot'}: {m['message']}"
                for m in conversation[-6:]  # last 3 exchanges
            ])
            user_msg = router.chat(
                role=AgentRole.GENERAL,
                system="You are a friendly person on a dating app. Write casual, natural messages. Just write one message (1-3 sentences).",
                messages=[{"role": "user", "content": f"Continue this conversation naturally. Previous messages:\n{conv_context}\n\nWrite your next message as the User:"}],
                temperature=0.8,
                max_tokens=100,
            )

        conversation.append({"speaker": "user", "message": user_msg.strip()})

        # Generate bot response using bot's persona
        conv_context = "\n".join([
            f"{'User' if m['speaker'] == 'user' else 'You'}: {m['message']}"
            for m in conversation[-6:]
        ])
        bot_msg = router.chat(
            role=AgentRole.PERSONA,
            system=bot_persona["system_prompt"],
            messages=[{"role": "user", "content": f"Continue this conversation naturally. Previous messages:\n{conv_context}\n\nWrite your next message:"}],
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
        # For relationship prediction, the ground truth is based on conversation dynamics
        # Since these are first conversations, expected: stranger -> acquaintance
        "relationship": {
            "rel_type": "love",  # dating app context
            "rel_status": "acquaintance",  # after 10 turns of casual chat
        },
    }


def generate_eval_dataset(
    n_conversations_per_bot: int = 3,
    n_turns: int = 10,
    output_path: Optional[str] = None,
) -> List[Dict]:
    """Generate the full evaluation dataset.

    Args:
        n_conversations_per_bot: conversations to generate per bot persona
        n_turns: turns per conversation (each turn = 1 user msg + 1 bot msg)
        output_path: where to save the dataset JSON
    """
    if output_path is None:
        output_path = str(Path(__file__).parent.parent / "data" / "evaluation" / "eval_dataset.json")

    personas = load_bot_personas()
    dataset = []

    for persona in personas:
        ground_truth = extract_ground_truth(persona)

        for conv_idx in range(n_conversations_per_bot):
            print(f"Generating conversation {conv_idx+1}/{n_conversations_per_bot} for {persona['profile_id']}...")

            try:
                conversation = generate_conversation(persona, n_turns=n_turns)

                sample = {
                    "conversation_id": f"{persona['profile_id']}_conv{conv_idx}",
                    "bot_id": persona["profile_id"],
                    "dialogue": conversation,
                    "ground_truth": ground_truth,
                    "n_turns": len(conversation),
                }
                dataset.append(sample)
            except Exception as e:
                print(f"  Error generating conversation: {e}")
                continue

    # Save dataset
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"\nDataset saved to {output_path}")
    print(f"Total samples: {len(dataset)}")
    return dataset


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate evaluation dataset")
    parser.add_argument("--n-conv", type=int, default=3, help="Conversations per bot")
    parser.add_argument("--n-turns", type=int, default=10, help="Turns per conversation")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    args = parser.parse_args()

    generate_eval_dataset(
        n_conversations_per_bot=args.n_conv,
        n_turns=args.n_turns,
        output_path=args.output,
    )
