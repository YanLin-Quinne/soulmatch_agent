"""Conversation simulator for bot-to-bot dialogue generation — uses LLMRouter."""

import random
from typing import Optional
from loguru import logger

from src.agents.persona_agent import PersonaAgent
from src.agents.prompt_generator import ConversationHistory
from src.agents.llm_router import router, AgentRole


class ConversationSimulator:
    """Simulates natural conversations between two persona agents."""

    ENDING_SIGNALS = [
        "bye", "goodbye", "see you", "gotta go", "talk later",
        "have a good", "take care", "catch you later", "nice chatting",
        "gtg", "ttyl", "cya"
    ]

    def __init__(self, temperature: float = 0.8, max_tokens: int = 200, ending_probability: float = 0.15):
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ending_probability = ending_probability

    def simulate_conversation(
        self,
        bot1: PersonaAgent,
        bot2: PersonaAgent,
        num_turns: int = 10,
        min_turns: int = 6,
        topic_prompt: Optional[str] = None,
    ) -> list[dict]:
        bot1.reset_conversation()
        bot2.reset_conversation()

        conversation_turns = []
        turn_count = 0

        # Bot1 opens
        if topic_prompt:
            first_message = self._generate_topic_opener(bot1, topic_prompt)
        else:
            first_message = bot1.generate_greeting()

        conversation_turns.append({
            "speaker": bot1.persona.profile_id,
            "message": first_message,
            "turn": turn_count,
        })
        turn_count += 1

        current_speaker = bot2
        current_listener = bot1

        while turn_count < num_turns * 2:
            last_message = conversation_turns[-1]["message"]

            if turn_count >= min_turns * 2:
                if self._should_end(last_message, turn_count, num_turns * 2):
                    break

            try:
                response = current_speaker.generate_response(message=last_message)
            except Exception as e:
                logger.error(f"Turn {turn_count} failed: {e}")
                response = current_speaker._get_fallback_response()

            conversation_turns.append({
                "speaker": current_speaker.persona.profile_id,
                "message": response,
                "turn": turn_count,
            })
            turn_count += 1
            current_speaker, current_listener = current_listener, current_speaker

        bot1.reset_conversation()
        bot2.reset_conversation()
        return conversation_turns

    def _generate_topic_opener(self, bot: PersonaAgent, topic: str) -> str:
        prompt = (
            f"Start a conversation naturally on a dating app. "
            f"Your first message should be warm and authentic, and casually mention: {topic}. "
            f"Keep it short (2-3 sentences)."
        )
        try:
            return router.chat(
                role=AgentRole.PERSONA,
                system=bot.persona.system_prompt,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=150,
            )
        except Exception as e:
            logger.error(f"Topic opener failed: {e}")
            return bot.generate_greeting()

    def _should_end(self, last_message: str, current_turn: int, max_turns: int) -> bool:
        lower = last_message.lower()
        for signal in self.ENDING_SIGNALS:
            if signal in lower:
                return True
        if current_turn >= max_turns * 0.7:
            progress = (current_turn - max_turns * 0.7) / (max_turns * 0.3)
            if random.random() < self.ending_probability * (1 + progress):
                return True
        return current_turn >= max_turns
