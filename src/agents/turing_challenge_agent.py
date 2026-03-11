"""Turing Challenge Agent - Manages Social Turing Test logic"""

from typing import Dict, Optional
from loguru import logger


class TuringChallengeAgent:
    """
    Manages the Social Turing Challenge (Stage 2).
    After 30 turns, users guess if their conversation partner is AI or human.
    """

    def __init__(self):
        # All bots are AI in this implementation
        self.bot_labels = {f"bot_{i}": True for i in range(15)}
        logger.info("TuringChallengeAgent initialized")

    def start_challenge(self) -> Dict:
        """Start the Turing challenge after 30 turns."""
        return {
            "message": "Based on your 30-turn conversation, guess: Is this a real person or AI?",
            "options": ["Real Person", "AI Bot"]
        }

    def submit_guess(self, bot_id: str, guess: str) -> Dict:
        """
        Evaluate user's guess.

        Args:
            bot_id: Bot identifier (e.g., "bot_0")
            guess: User's guess ("Real Person" or "AI Bot")

        Returns:
            Result dict with correctness and score
        """
        is_ai = self.bot_labels.get(bot_id, True)
        correct_answer = "AI Bot" if is_ai else "Real Person"
        is_correct = (guess == correct_answer)

        logger.info(f"Turing test: bot={bot_id}, guess={guess}, correct={is_correct}")

        return {
            "correct": is_correct,
            "your_guess": guess,
            "actual": correct_answer,
            "score": 100 if is_correct else 0,
            "message": "Correct! You identified the AI." if is_correct else "Incorrect. This was an AI bot.",
        }
