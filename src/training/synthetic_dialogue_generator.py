"""Synthetic dialogue dataset generator for SFT training"""

import json
import random
import uuid
from pathlib import Path
from typing import Optional, Union
from itertools import combinations
from loguru import logger

try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    logger.warning("tqdm not installed, progress bars disabled")

from src.agents.persona_agent import PersonaAgent, PersonaAgentPool
from src.data.schema import PersonaProfile, ExtractedFeatures
from src.training.conversation_simulator import ConversationSimulator


class SyntheticDialogueGenerator:
    """Generates synthetic training data from bot-to-bot conversations"""
    
    # Common dating app conversation topics
    CONVERSATION_TOPICS = [
        "weekend plans", "favorite music", "travel experiences", "food preferences",
        "hobbies and interests", "work and career", "movies and TV shows", 
        "fitness and health", "books and reading", "pets and animals",
        "art and creativity", "sports", "technology", "cooking",
        "outdoor activities", "coffee or tea", "favorite restaurants",
        "life goals", "childhood memories", "recent adventures"
    ]
    
    def __init__(
        self,
        agent_pool: PersonaAgentPool,
        simulator: Optional[ConversationSimulator] = None,
        output_dir: Union[str, Path] = "data/training"
    ):
        """
        Initialize synthetic dialogue generator
        
        Args:
            agent_pool: Pool of persona agents to use
            simulator: Conversation simulator (creates default if None)
            output_dir: Directory to save generated data
        """
        self.agent_pool = agent_pool
        self.simulator = simulator or ConversationSimulator()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(
            f"SyntheticDialogueGenerator initialized with {len(agent_pool)} agents"
        )
    
    def generate_dataset(
        self,
        num_conversations: int = 100,
        min_turns: int = 6,
        max_turns: int = 12,
        output_filename: str = "synthetic_dialogues.jsonl",
        resume: bool = True
    ) -> Path:
        """
        Generate synthetic dialogue dataset
        
        Args:
            num_conversations: Total number of conversations to generate
            min_turns: Minimum conversation turns
            max_turns: Maximum conversation turns
            output_filename: Output file name (JSONL format)
            resume: Resume from existing file if present
        
        Returns:
            Path to generated dataset file
        """
        output_path = self.output_dir / output_filename
        
        # Check for existing data to resume
        existing_dialogues = []
        if resume and output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_dialogues = [json.loads(line) for line in f]
                logger.info(
                    f"Resuming from {output_path}: {len(existing_dialogues)} dialogues already generated"
                )
            except Exception as e:
                logger.warning(f"Could not load existing dialogues: {e}")
                existing_dialogues = []
        
        # Calculate how many more to generate
        already_generated = len(existing_dialogues)
        remaining = num_conversations - already_generated
        
        if remaining <= 0:
            logger.info(f"Dataset already complete with {already_generated} dialogues")
            return output_path
        
        logger.info(f"Generating {remaining} more dialogues...")
        
        # Get all agents
        agents = self.agent_pool.get_all_agents()
        
        if len(agents) < 2:
            raise ValueError(f"Need at least 2 agents, but only have {len(agents)}")
        
        # Generate bot pairs (all combinations)
        all_pairs = list(combinations(agents, 2))
        logger.info(f"Total possible bot pairs: {len(all_pairs)}")
        
        # Open file in append mode
        mode = 'a' if resume and already_generated > 0 else 'w'
        
        with open(output_path, mode, encoding='utf-8') as f:
            # Setup progress bar
            iterator = range(remaining)
            if TQDM_AVAILABLE:
                iterator = tqdm(iterator, desc="Generating dialogues", initial=already_generated, total=num_conversations)
            
            for i in iterator:
                try:
                    # Select random bot pair
                    bot1, bot2 = random.choice(all_pairs)
                    
                    # Randomize turn count for variety
                    num_turns = random.randint(min_turns, max_turns)
                    
                    # Randomly pick 1-2 topics
                    num_topics = random.randint(1, 2)
                    topics = random.sample(self.CONVERSATION_TOPICS, num_topics)
                    topic_prompt = topics[0] if topics else None
                    
                    # Simulate conversation
                    turns = self.simulator.simulate_conversation(
                        bot1=bot1,
                        bot2=bot2,
                        num_turns=num_turns,
                        min_turns=min_turns,
                        topic_prompt=topic_prompt
                    )
                    
                    # Create dialogue record
                    dialogue = self._create_dialogue_record(
                        bot1=bot1,
                        bot2=bot2,
                        turns=turns,
                        topics=topics
                    )
                    
                    # Write to file (JSONL format: one JSON object per line)
                    f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
                    f.flush()  # Ensure data is written (for resume support)
                    
                    if not TQDM_AVAILABLE and (i + 1) % 10 == 0:
                        logger.info(f"Progress: {already_generated + i + 1}/{num_conversations}")
                
                except Exception as e:
                    logger.error(f"Failed to generate dialogue {i}: {e}")
                    continue
        
        logger.info(f"Dataset generation complete: {output_path}")
        logger.info(f"Total dialogues: {num_conversations}")
        
        return output_path
    
    def _create_dialogue_record(
        self,
        bot1: PersonaAgent,
        bot2: PersonaAgent,
        turns: list[dict],
        topics: list[str]
    ) -> dict:
        """
        Create a structured dialogue record with ground truth features
        
        Args:
            bot1: First bot
            bot2: Second bot
            turns: Conversation turns
            topics: Conversation topics
        
        Returns:
            Dialogue record with metadata and ground truth
        """
        return {
            "dialogue_id": str(uuid.uuid4()),
            "bot1_id": bot1.persona.profile_id,
            "bot2_id": bot2.persona.profile_id,
            "topics": topics,
            "num_turns": len(turns),
            "turns": turns,
            "ground_truth_features": {
                bot1.persona.profile_id: self._extract_ground_truth(bot1.persona),
                bot2.persona.profile_id: self._extract_ground_truth(bot2.persona)
            },
            "metadata": {
                "bot1_summary": bot1.get_persona_summary(),
                "bot2_summary": bot2.get_persona_summary(),
                "generation_timestamp": self._get_timestamp()
            }
        }
    
    def _extract_ground_truth(self, persona: PersonaProfile) -> dict:
        """
        Extract ground truth features from PersonaProfile
        
        This extracts the features in a format compatible with
        FeaturePredictionAgent output for training
        
        Args:
            persona: PersonaProfile to extract from
        
        Returns:
            Ground truth features dictionary
        """
        features = persona.features
        
        return {
            # Communication style
            "communication_style": features.communication_style,
            "communication_confidence": features.communication_confidence,
            
            # Values
            "core_values": features.core_values,
            "values_confidence": features.values_confidence,
            
            # Interests (top categories)
            "interests": features.interest_categories,
            
            # Big Five personality traits
            "personality": {
                "openness": features.openness,
                "conscientiousness": features.conscientiousness,
                "extraversion": features.extraversion,
                "agreeableness": features.agreeableness,
                "neuroticism": features.neuroticism
            },
            
            # Relationship goals
            "relationship_goals": features.relationship_goals,
            "goals_confidence": features.goals_confidence,
            
            # Summary
            "personality_summary": features.personality_summary,
            
            # Original profile metadata (useful for training)
            "profile_metadata": {
                "age": persona.original_profile.age,
                "sex": persona.original_profile.sex,
                "location": persona.original_profile.location,
                "education": persona.original_profile.education,
                "orientation": persona.original_profile.orientation
            }
        }
    
    def _get_timestamp(self) -> str:
        """Get current timestamp in ISO format"""
        from datetime import datetime
        return datetime.utcnow().isoformat() + 'Z'
    
    def generate_balanced_dataset(
        self,
        conversations_per_pair: int = 3,
        min_turns: int = 6,
        max_turns: int = 12,
        output_filename: str = "synthetic_dialogues_balanced.jsonl",
        resume: bool = True
    ) -> Path:
        """
        Generate a balanced dataset with equal representation of all bot pairs
        
        This ensures each unique bot pair gets equal number of conversations,
        which is better for training than random sampling
        
        Args:
            conversations_per_pair: Number of conversations per bot pair
            min_turns: Minimum conversation turns
            max_turns: Maximum conversation turns
            output_filename: Output file name
            resume: Resume from existing file
        
        Returns:
            Path to generated dataset
        """
        output_path = self.output_dir / output_filename
        
        # Get all agents and create pairs
        agents = self.agent_pool.get_all_agents()
        all_pairs = list(combinations(agents, 2))
        
        total_conversations = len(all_pairs) * conversations_per_pair
        logger.info(
            f"Generating balanced dataset: {len(all_pairs)} pairs × {conversations_per_pair} = "
            f"{total_conversations} conversations"
        )
        
        # Check for existing data
        existing_count = 0
        if resume and output_path.exists():
            try:
                with open(output_path, 'r', encoding='utf-8') as f:
                    existing_count = sum(1 for _ in f)
                logger.info(f"Resuming: {existing_count} dialogues already exist")
            except Exception as e:
                logger.warning(f"Could not count existing dialogues: {e}")
        
        # Generate conversations
        mode = 'a' if resume and existing_count > 0 else 'w'
        generated = 0
        
        with open(output_path, mode, encoding='utf-8') as f:
            # Create iterator for all pair-conversation combinations
            total_items = len(all_pairs) * conversations_per_pair
            
            if TQDM_AVAILABLE:
                pair_iterator = tqdm(all_pairs, desc="Bot pairs")
            else:
                pair_iterator = all_pairs
            
            for pair_idx, (bot1, bot2) in enumerate(pair_iterator):
                for conv_idx in range(conversations_per_pair):
                    # Skip if already generated (for resume)
                    global_idx = pair_idx * conversations_per_pair + conv_idx
                    if global_idx < existing_count:
                        continue
                    
                    try:
                        # Vary conversation parameters
                        num_turns = random.randint(min_turns, max_turns)
                        num_topics = random.randint(1, 2)
                        topics = random.sample(self.CONVERSATION_TOPICS, num_topics)
                        
                        # Simulate
                        turns = self.simulator.simulate_conversation(
                            bot1=bot1,
                            bot2=bot2,
                            num_turns=num_turns,
                            min_turns=min_turns,
                            topic_prompt=topics[0]
                        )
                        
                        # Create record
                        dialogue = self._create_dialogue_record(
                            bot1=bot1,
                            bot2=bot2,
                            turns=turns,
                            topics=topics
                        )
                        
                        # Write
                        f.write(json.dumps(dialogue, ensure_ascii=False) + '\n')
                        f.flush()
                        
                        generated += 1
                        
                        if not TQDM_AVAILABLE and generated % 10 == 0:
                            logger.info(f"Progress: {existing_count + generated}/{total_conversations}")
                    
                    except Exception as e:
                        logger.error(
                            f"Failed to generate dialogue for pair "
                            f"({bot1.persona.profile_id}, {bot2.persona.profile_id}): {e}"
                        )
                        continue
        
        logger.info(f"Balanced dataset complete: {output_path}")
        logger.info(f"Total dialogues: {existing_count + generated}")
        
        return output_path
    
    def load_dataset(self, filepath: Union[str, Path]) -> list[dict]:
        """
        Load a generated JSONL dataset
        
        Args:
            filepath: Path to JSONL file
        
        Returns:
            List of dialogue records
        """
        filepath = Path(filepath)
        
        if not filepath.exists():
            raise FileNotFoundError(f"Dataset not found: {filepath}")
        
        dialogues = []
        with open(filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    dialogue = json.loads(line)
                    dialogues.append(dialogue)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping invalid JSON at line {line_num}: {e}")
        
        logger.info(f"Loaded {len(dialogues)} dialogues from {filepath}")
        return dialogues
    
    def get_dataset_statistics(self, filepath: Union[str, Path]) -> dict:
        """
        Compute statistics about a generated dataset
        
        Args:
            filepath: Path to dataset file
        
        Returns:
            Statistics dictionary
        """
        dialogues = self.load_dataset(filepath)
        
        if not dialogues:
            return {"error": "No dialogues found"}
        
        # Compute stats
        total_dialogues = len(dialogues)
        total_turns = sum(d["num_turns"] for d in dialogues)
        avg_turns = total_turns / total_dialogues
        
        # Bot participation
        bot_counts = {}
        for d in dialogues:
            for bot_id in [d["bot1_id"], d["bot2_id"]]:
                bot_counts[bot_id] = bot_counts.get(bot_id, 0) + 1
        
        # Topic distribution
        topic_counts = {}
        for d in dialogues:
            for topic in d.get("topics", []):
                topic_counts[topic] = topic_counts.get(topic, 0) + 1
        
        return {
            "total_dialogues": total_dialogues,
            "total_turns": total_turns,
            "avg_turns_per_dialogue": round(avg_turns, 2),
            "min_turns": min(d["num_turns"] for d in dialogues),
            "max_turns": max(d["num_turns"] for d in dialogues),
            "unique_bots": len(bot_counts),
            "bot_participation": bot_counts,
            "topic_distribution": topic_counts,
            "total_topics_used": len(topic_counts)
        }


# Convenience function
def create_synthetic_dataset(
    personas_path: Union[str, Path] = "data/processed/bot_personas.json",
    output_dir: Union[str, Path] = "data/training",
    num_conversations: int = 100,
    balanced: bool = True,
    conversations_per_pair: int = 3,
    resume: bool = True,
) -> Path:
    """
    Quick setup: Generate synthetic dialogue dataset from bot personas.

    Returns:
        Path to generated dataset file
    """
    from src.agents.persona_agent import PersonaAgentPool

    logger.info(f"Loading personas from {personas_path}")
    pool = PersonaAgentPool(temperature=0.8)
    pool.load_from_file(personas_path)
    
    # Create generator
    generator = SyntheticDialogueGenerator(
        agent_pool=pool,
        output_dir=output_dir
    )
    
    # Generate dataset
    if balanced:
        output_path = generator.generate_balanced_dataset(
            conversations_per_pair=conversations_per_pair,
            resume=resume
        )
    else:
        output_path = generator.generate_dataset(
            num_conversations=num_conversations,
            resume=resume
        )
    
    # Print statistics
    stats = generator.get_dataset_statistics(output_path)
    logger.info(f"Dataset statistics: {json.dumps(stats, indent=2)}")
    
    return output_path
