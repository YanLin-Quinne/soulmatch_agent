"""Memory Manager — LLM-powered memory decisions via LLMRouter."""

import json
import uuid
from typing import Optional, List
from loguru import logger

from src.memory.memory_operations import Memory, MemoryAction, MemoryOperation, MemoryReward
from src.memory.chromadb_client import ChromaDBClient
from src.agents.llm_router import router, AgentRole


class MemoryManager:
    """Memory Manager Agent using LLMRouter for multi-provider support."""

    def __init__(self, user_id: str, db_client: Optional[ChromaDBClient] = None, use_llm: bool = True):
        self.user_id = user_id
        self.db_client = db_client or ChromaDBClient()
        self.use_llm = use_llm
        self.conversation_history: List[dict] = []
        self.current_turn: int = 0

    def add_conversation_turn(self, speaker: str, message: str):
        self.conversation_history.append({"turn": self.current_turn, "speaker": speaker, "message": message})
        self.current_turn += 1

    def decide_memory_action(self, recent_messages: List[dict], current_features: Optional[dict] = None) -> MemoryAction:
        if not self.use_llm:
            return self._heuristic_decide(recent_messages)
        return self._llm_decide(recent_messages, current_features)

    def retrieve_relevant_memories(self, query: str, n: int = 5) -> List[str]:
        """Retrieve memory texts relevant to a query."""
        memories = self.db_client.retrieve_memories(user_id=self.user_id, query_text=query, n_results=n)
        return [m.content for m in memories] if memories else []

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
            text = router.chat(
                role=AgentRole.MEMORY,
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
        return self.db_client.retrieve_memories(self.user_id, query_text=action.callback_query, n_results=5)

    def get_all_memories(self) -> List[Memory]:
        return self.db_client.retrieve_memories(self.user_id, n_results=100)
