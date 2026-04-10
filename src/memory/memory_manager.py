"""Memory Manager — LLM-powered memory decisions via LLMRouter with Three-Layer Memory."""

import json
import uuid
from typing import List, Optional, TYPE_CHECKING
from loguru import logger

from src.agents.memory_privacy_manager import MemoryPrivacyManager
from src.memory.memory_operations import Memory, MemoryAction, MemoryOperation
from src.memory.chromadb_client import ChromaDBClient

if TYPE_CHECKING:
    from src.memory.three_layer_memory import ThreeLayerMemory
    from src.agents.llm_router import AgentRole


class MemoryManager:
    """Memory Manager Agent using LLMRouter with Three-Layer Memory System."""

    def __init__(
        self,
        user_id: str,
        db_client: Optional[ChromaDBClient] = None,
        use_llm: bool = True,
        use_three_layer: bool = True,
        session_store=None,
        privacy_manager: Optional[MemoryPrivacyManager] = None,
    ):
        self.user_id = user_id
        self.db_client = db_client or ChromaDBClient()
        self.use_llm = use_llm
        self.use_three_layer = use_three_layer
        self.privacy_manager = privacy_manager
        self.conversation_history: List[dict] = []
        self.current_turn: int = 0

        # Import router and AgentRole at runtime to avoid circular dependency
        from src.agents.llm_router import router, AgentRole
        self.router = router
        self.AgentRole = AgentRole

        # Three-layer memory system (lazy import to avoid circular dependency)
        self.three_layer_memory: Optional['ThreeLayerMemory'] = None
        if use_three_layer:
            from src.memory.three_layer_memory import ThreeLayerMemory
            self.three_layer_memory = ThreeLayerMemory(llm_router=router, user_id=user_id, session_store=session_store)

    def add_conversation_turn(self, speaker: str, message: str):
        self.conversation_history.append({"turn": self.current_turn, "speaker": speaker, "message": message})
        self.current_turn += 1

        # Add to three-layer memory system
        if self.three_layer_memory:
            self.three_layer_memory.add_to_working_memory(speaker, message)

    def decide_memory_action(self, recent_messages: List[dict], current_features: Optional[dict] = None) -> MemoryAction:
        if not self.use_llm:
            return self._heuristic_decide(recent_messages)
        return self._llm_decide(recent_messages, current_features)

    def retrieve_relevant_memories(self, query: str, n: int = 5) -> List[str]:
        """Retrieve memory texts relevant to a query."""
        if self.three_layer_memory:
            # Use three-layer memory system for retrieval
            filters = self._detect_query_topic(query)
            full_context = self.three_layer_memory.get_full_context(query, **filters)
            return [full_context] if full_context else []
        else:
            # Fallback to ChromaDB
            memories = self.db_client.retrieve_memories(user_id=self.user_id, query_text=query, n_results=n)
            memories = self._filter_forgotten_memories(memories)
            return [m.content for m in memories] if memories else []

    def get_memory_stats(self) -> dict:
        """Get three-layer memory statistics."""
        if self.three_layer_memory:
            return self.three_layer_memory.get_memory_stats()
        return {
            "current_turn": self.current_turn,
            "total_memories": 0
        }

    def _detect_query_topic(self, query: str) -> dict:
        """Auto-detect metadata filters from query context."""
        query_lower = query.lower()
        filters = {}

        if any(word in query_lower for word in ["feel", "emotion", "mood", "happy", "sad"]):
            filters["topic_filter"] = "emotion"
        elif any(word in query_lower for word in ["personality", "trait", "character"]):
            filters["topic_filter"] = "personality"
        elif any(word in query_lower for word in ["hobby", "interest", "like", "enjoy"]):
            filters["topic_filter"] = "interest"
        elif any(word in query_lower for word in ["relationship", "trust", "bond"]):
            filters["topic_filter"] = "relationship"
        elif any(word in query_lower for word in ["scam", "suspicious", "warning", "risk", "fraud", "money", "transfer"]):
            filters["topic_filter"] = "safety"

        if any(word in query_lower for word in ["big five", "big5", "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"]):
            filters["feature_group_filter"] = "big5"
        elif any(word in query_lower for word in ["mbti", "introvert", "extrovert", "sensing", "intuition", "thinking", "feeling", "judging", "perceiving"]):
            filters["feature_group_filter"] = "mbti"
        elif any(word in query_lower for word in ["attachment", "secure", "anxious", "avoidant", "disorganized"]):
            filters["feature_group_filter"] = "attachment"
        elif any(word in query_lower for word in ["trust", "trustworthy", "trust score"]):
            filters["feature_group_filter"] = "trust"
        elif any(word in query_lower for word in ["interest", "hobby", "passion", "enjoy"]):
            filters["feature_group_filter"] = "interest"
        elif any(word in query_lower for word in ["lifestyle", "diet", "exercise", "routine", "habit"]):
            filters["feature_group_filter"] = "lifestyle"
        elif any(word in query_lower for word in ["education", "job", "career", "family", "hometown"]):
            filters["feature_group_filter"] = "background"

        if any(word in query_lower for word in ["improving", "better", "positive", "progress"]):
            filters["emotion_valence_filter"] = "positive"
        elif any(word in query_lower for word in ["declining", "worse", "negative", "conflict"]):
            filters["emotion_valence_filter"] = "negative"
        elif any(word in query_lower for word in ["stable", "neutral", "steady"]):
            filters["emotion_valence_filter"] = "neutral"

        return filters

    # ------------------------------------------------------------------

    def _heuristic_decide(self, recent_messages: List[dict]) -> MemoryAction:
        if self.current_turn % 5 == 0 and recent_messages:
            return MemoryAction(
                operation=MemoryOperation.ADD,
                content=recent_messages[-1]["message"],
                memory_type="conversation",
                importance=0.5,
                reasoning="Periodic save",
            )
        return MemoryAction(operation=MemoryOperation.NOOP)

    def _llm_decide(self, recent_messages: List[dict], current_features: Optional[dict]) -> MemoryAction:
        context = "Recent conversation:\n"
        for msg in recent_messages[-5:]:
            context += f"{msg['speaker']}: {msg['message']}\n"

        memories = self.db_client.retrieve_memories(user_id=self.user_id, n_results=5)
        memories = self._filter_forgotten_memories(memories)
        mem_ctx = "\nExisting memories:\n"
        if memories:
            for m in memories:
                mem_ctx += f"- [{m.memory_id}] {m.content}\n"
        else:
            mem_ctx += "- (no memories yet)\n"

        prompt = (
            f"{context}\n{mem_ctx}\n"
            f"Current features: {current_features or 'unknown'}\n\n"
            "Available operations: ADD, UPDATE, DELETE, CALLBACK, NOOP.\n"
            "Respond with JSON:\n"
            '{"operation":"<OP>","content":"<for ADD>","memory_type":"<conversation/feature/fact/event>",'
            '"importance":0.0-1.0,"memory_id":"<for UPDATE/DELETE>",'
            '"new_content":"<for UPDATE>","callback_query":"<for CALLBACK>","reasoning":"<why>"}'
        )

        try:
            text = self.router.chat(
                role=self.AgentRole.MEMORY,
                system="You are a memory management agent for a dating app. Decide what to remember.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=512,
                json_mode=True,
            )
            text = text.strip()
            if text.startswith("```json"):
                text = text[7:]
            if text.startswith("```"):
                text = text[3:]
            if text.endswith("```"):
                text = text[:-3]
            action = MemoryAction(**json.loads(text.strip()))
            return action if action.validate_action() else MemoryAction(operation=MemoryOperation.NOOP)
        except Exception as e:
            logger.error(f"LLM memory decision failed: {e}")
            return MemoryAction(operation=MemoryOperation.NOOP)

    # ------------------------------------------------------------------
    # Execute
    # ------------------------------------------------------------------

    def execute_action(self, action: MemoryAction) -> Optional[List[Memory]]:
        if action.operation == MemoryOperation.ADD:
            return self._add(action)
        elif action.operation == MemoryOperation.UPDATE:
            return self._update(action)
        elif action.operation == MemoryOperation.DELETE:
            return self._delete(action)
        elif action.operation == MemoryOperation.CALLBACK:
            return self._callback(action)
        return None

    def _add(self, action: MemoryAction) -> List[Memory]:
        memory = Memory(
            memory_id=str(uuid.uuid4()),
            content=action.content,
            memory_type=action.memory_type or "conversation",
            importance=action.importance or 0.5,
            conversation_turn=self.current_turn,
            tags=action.tags or [],
        )
        self.db_client.add_memory(self.user_id, memory)
        logger.info(f"Added memory: {memory.content[:50]}...")
        return [memory]

    def _update(self, action: MemoryAction) -> List[Memory]:
        self.db_client.update_memory(self.user_id, action.memory_id, action.new_content)
        return []

    def _delete(self, action: MemoryAction) -> List[Memory]:
        self.db_client.delete_memory(self.user_id, action.memory_id)
        return []

    def _callback(self, action: MemoryAction) -> List[Memory]:
        memories = self.db_client.retrieve_memories(
            self.user_id,
            query_text=action.callback_query,
            n_results=5,
        )
        return self._filter_forgotten_memories(memories)

    def get_all_memories(self) -> List[Memory]:
        memories = self.db_client.retrieve_memories(self.user_id, n_results=100)
        return self._filter_forgotten_memories(memories)

    def _filter_forgotten_memories(self, memories: Optional[List[Memory]]) -> List[Memory]:
        """Drop memories the user has asked the system to forget."""
        if not memories:
            return []
        if not self.privacy_manager:
            return list(memories)

        forgotten_ids = self.privacy_manager.get_forgotten_ids()
        if not forgotten_ids:
            return list(memories)

        return [memory for memory in memories if memory.memory_id not in forgotten_ids]
