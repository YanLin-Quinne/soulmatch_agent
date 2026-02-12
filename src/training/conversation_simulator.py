"""Conversation simulator for bot-to-bot dialogue generation"""

import random
from typing import Optional
from loguru import logger

from src.agents.persona_agent import PersonaAgent
from src.agents.prompt_generator import ConversationHistory


class ConversationSimulator:
    """Simulates natural conversations between two persona agents"""
    
    # Conversation ending signals
    ENDING_SIGNALS = [
        "bye", "goodbye", "see you", "gotta go", "talk later",
        "have a good", "take care", "catch you later", "nice chatting",
        "gtg", "ttyl", "cya"
    ]
    
    def __init__(
        self,
        temperature: float = 0.8,
        max_tokens: int = 200,
        ending_probability: float = 0.15  # Chance of natural ending after min_turns
    ):
        """
        Initialize conversation simulator
        
        Args:
            temperature: Response creativity (higher = more varied)
            max_tokens: Max tokens per message
            ending_probability: Probability of conversation ending naturally after min turns
        """
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.ending_probability = ending_probability
        
        logger.info("ConversationSimulator initialized")
    
    def simulate_conversation(
        self,
        bot1: PersonaAgent,
        bot2: PersonaAgent,
        num_turns: int = 10,
        min_turns: int = 6,
        topic_prompt: Optional[str] = None
    ) -> list[dict]:
        """
        Simulate a conversation between two bots
        
        Args:
            bot1: First persona agent (initiates conversation)
            bot2: Second persona agent (responds)
            num_turns: Maximum number of turns (messages per bot)
            min_turns: Minimum turns before natural ending can occur
            topic_prompt: Optional topic to guide conversation start
        
        Returns:
            List of conversation turns with format:
            [
                {"speaker": "bot_0", "message": "Hi!", "turn": 0},
                {"speaker": "bot_1", "message": "Hello!", "turn": 1}
            ]
        """
        # Reset both agents' conversation history
        bot1.reset_conversation()
        bot2.reset_conversation()
        
        # Create shared conversation history for both bots
        shared_history = ConversationHistory(max_messages=num_turns * 2)
        
        conversation_turns = []
        turn_count = 0
        
        logger.info(
            f"Starting conversation: {bot1.persona.profile_id} <-> {bot2.persona.profile_id}"
        )
        
        # Bot1 starts with greeting or topic-based opener
        if topic_prompt:
            # Use topic to generate contextual first message
            first_message = self._generate_topic_opener(bot1, topic_prompt, shared_history)
        else:
            first_message = bot1.generate_greeting()
        
        # Add to conversation
        conversation_turns.append({
            "speaker": bot1.persona.profile_id,
            "message": first_message,
            "turn": turn_count
        })
        
        # Add to shared history (as "user" from bot2's perspective)
        shared_history.add_user_message(first_message)
        turn_count += 1
        
        logger.debug(f"[Turn {turn_count-1}] {bot1.persona.profile_id}: {first_message[:50]}...")
        
        # Alternate conversation
        current_speaker = bot2
        current_listener = bot1
        
        while turn_count < num_turns * 2:
            # Get last message (from the other bot)
            last_message = conversation_turns[-1]["message"]
            
            # Check for natural ending
            if turn_count >= min_turns * 2:
                if self._should_end_conversation(last_message, turn_count, num_turns * 2):
                    logger.info(f"Conversation ended naturally at turn {turn_count}")
                    break
            
            # Generate response
            try:
                response = current_speaker.generate_response(
                    message=last_message,
                    conversation_history=shared_history
                )
            except Exception as e:
                logger.error(f"Failed to generate response at turn {turn_count}: {e}")
                # Try to recover with fallback
                response = current_speaker._get_fallback_response()
            
            # Add to conversation
            conversation_turns.append({
                "speaker": current_speaker.persona.profile_id,
                "message": response,
                "turn": turn_count
            })
            
            turn_count += 1
            logger.debug(f"[Turn {turn_count-1}] {current_speaker.persona.profile_id}: {response[:50]}...")
            
            # Swap speakers
            current_speaker, current_listener = current_listener, current_speaker
        
        logger.info(f"Conversation completed with {turn_count} turns")
        
        # Reset agents after conversation
        bot1.reset_conversation()
        bot2.reset_conversation()
        
        return conversation_turns
    
    def _generate_topic_opener(
        self,
        bot: PersonaAgent,
        topic: str,
        history: ConversationHistory
    ) -> str:
        """
        Generate a first message that naturally incorporates a topic
        
        Args:
            bot: Bot generating the opener
            topic: Topic to incorporate (e.g., "travel", "music", "cooking")
            history: Shared conversation history
        
        Returns:
            Opening message
        """
        # Create a prompt that weaves topic into greeting
        prompt = (
            f"Start a conversation naturally on a dating app. "
            f"Your first message should be warm and authentic, and casually mention or relate to: {topic}. "
            f"Keep it short (2-3 sentences), friendly, and don't be too direct or forced."
        )
        
        temp_history = ConversationHistory(max_messages=2)
        temp_history.add_user_message(prompt)
        
        try:
            if bot.use_claude:
                response = bot.client.messages.create(
                    model=bot.model,
                    max_tokens=150,
                    temperature=0.9,
                    system=bot.persona.system_prompt,
                    messages=temp_history.to_api_format()
                )
                opener = response.content[0].text
            else:
                messages = [{"role": "system", "content": bot.persona.system_prompt}]
                messages.extend(temp_history.to_api_format())
                
                response = bot.client.chat.completions.create(
                    model=bot.model,
                    messages=messages,
                    temperature=0.9,
                    max_tokens=150
                )
                opener = response.choices[0].message.content
            
            return opener
            
        except Exception as e:
            logger.error(f"Failed to generate topic opener: {e}")
            # Fallback to generic greeting
            return bot.generate_greeting()
    
    def _should_end_conversation(
        self,
        last_message: str,
        current_turn: int,
        max_turns: int
    ) -> bool:
        """
        Determine if conversation should end naturally
        
        Args:
            last_message: The most recent message
            current_turn: Current turn number
            max_turns: Maximum allowed turns
        
        Returns:
            True if conversation should end
        """
        # Check for explicit ending signals
        last_message_lower = last_message.lower()
        for signal in self.ENDING_SIGNALS:
            if signal in last_message_lower:
                logger.debug(f"Detected ending signal: '{signal}'")
                return True
        
        # Natural probabilistic ending (increases as we approach max_turns)
        # This creates varied conversation lengths
        if current_turn >= max_turns * 0.7:  # After 70% of max turns
            progress = (current_turn - max_turns * 0.7) / (max_turns * 0.3)
            # Gradually increase ending probability
            adjusted_probability = self.ending_probability * (1 + progress)
            
            if random.random() < adjusted_probability:
                logger.debug(f"Natural probabilistic ending at turn {current_turn}")
                return True
        
        # Force end at max turns
        if current_turn >= max_turns:
            logger.debug(f"Reached max turns: {max_turns}")
            return True
        
        return False
    
    def simulate_multi_topic_conversation(
        self,
        bot1: PersonaAgent,
        bot2: PersonaAgent,
        topics: list[str],
        turns_per_topic: int = 4
    ) -> list[dict]:
        """
        Simulate a conversation that covers multiple topics sequentially
        
        Args:
            bot1: First persona agent
            bot2: Second persona agent
            topics: List of topics to cover
            turns_per_topic: Approximate turns to spend on each topic
        
        Returns:
            Full conversation covering all topics
        """
        all_turns = []
        
        for i, topic in enumerate(topics):
            logger.info(f"Topic {i+1}/{len(topics)}: {topic}")
            
            # For first topic, use topic opener; for subsequent, use None (natural flow)
            topic_prompt = topic if i == 0 else None
            
            # Simulate conversation segment
            segment = self.simulate_conversation(
                bot1=bot1,
                bot2=bot2,
                num_turns=turns_per_topic,
                min_turns=turns_per_topic - 1,
                topic_prompt=topic_prompt
            )
            
            # Adjust turn numbers to be continuous
            turn_offset = len(all_turns)
            for turn in segment:
                turn["turn"] += turn_offset
                turn["topic"] = topic  # Tag with topic
            
            all_turns.extend(segment)
        
        logger.info(f"Multi-topic conversation completed: {len(all_turns)} total turns")
        return all_turns
