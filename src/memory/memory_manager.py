"""Memory Manager Agent with ADD/UPDATE/DELETE/NOOP/CALLBACK operations"""

import uuid
from typing import Optional, List
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from src.memory.memory_operations import (
    Memory, MemoryAction, MemoryOperation, MemoryReward
)
from src.memory.chromadb_client import ChromaDBClient
from src.config import settings


class MemoryManager:
    """Memory Manager Agent (Memory-R1 + ReMemR1)"""
    
    def __init__(
        self, 
        user_id: str,
        db_client: Optional[ChromaDBClient] = None,
        use_llm: bool = True
    ):
        self.user_id = user_id
        self.db_client = db_client or ChromaDBClient()
        self.use_llm = use_llm
        
        # LLM for memory decision-making (optional, for inference)
        self.llm_client = None
        if use_llm and ANTHROPIC_AVAILABLE and settings.anthropic_api_key:
            self.llm_client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        
        # Conversation history (for context)
        self.conversation_history: List[dict] = []
        self.current_turn: int = 0
    
    def add_conversation_turn(self, speaker: str, message: str):
        """Add a conversation turn"""
        self.conversation_history.append({
            "turn": self.current_turn,
            "speaker": speaker,
            "message": message
        })
        self.current_turn += 1
    
    def decide_memory_action(
        self, 
        recent_messages: List[dict],
        current_features: Optional[dict] = None
    ) -> MemoryAction:
        """Decide what memory operation to perform"""
        
        if not self.use_llm or not self.llm_client:
            # Simple heuristic: add memory every few turns
            if self.current_turn % 5 == 0 and recent_messages:
                last_msg = recent_messages[-1]
                return MemoryAction(
                    operation=MemoryOperation.ADD,
                    content=last_msg["message"],
                    memory_type="conversation",
                    importance=0.5,
                    reasoning="Periodic save"
                )
            else:
                return MemoryAction(operation=MemoryOperation.NOOP)
        
        # Use LLM to decide
        return self._llm_decide_action(recent_messages, current_features)
    
    def _llm_decide_action(
        self,
        recent_messages: List[dict],
        current_features: Optional[dict] = None
    ) -> MemoryAction:
        """Use LLM to decide memory action"""
        
        # Build context
        context = "Recent conversation:\n"
        for msg in recent_messages[-5:]:  # Last 5 messages
            context += f"{msg['speaker']}: {msg['message']}\n"
        
        # Get existing memories
        memories = self.db_client.retrieve_memories(
            user_id=self.user_id,
            n_results=5
        )
        
        memory_context = "\nExisting memories:\n"
        if memories:
            for mem in memories:
                memory_context += f"- [{mem.memory_id}] {mem.content}\n"
        else:
            memory_context += "- (no memories yet)\n"
        
        prompt = f"""You are a memory management agent. Decide what memory operation to perform.

{context}

{memory_context}

Current features: {current_features or 'unknown'}

Available operations:
- ADD: Create a new memory (important facts, preferences, events)
- UPDATE: Update an existing memory with new information
- DELETE: Remove outdated or incorrect memory
- CALLBACK: Retrieve relevant memories for context
- NOOP: No action needed

Respond with JSON:
{{
  "operation": "<ADD/UPDATE/DELETE/CALLBACK/NOOP>",
  "content": "<memory content for ADD>",
  "memory_type": "<conversation/feature/fact/event>",
  "importance": <0.0-1.0>,
  "memory_id": "<id for UPDATE/DELETE>",
  "new_content": "<updated content for UPDATE>",
  "callback_query": "<query for CALLBACK>",
  "reasoning": "<why this operation>"
}}

Only include fields relevant to the chosen operation."""

        try:
            response = self.llm_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}]
            )
            
            import json
            content = response.content[0].text.strip()
            
            # Remove markdown
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()
            
            action_dict = json.loads(content)
            action = MemoryAction(**action_dict)
            
            if action.validate_action():
                return action
            else:
                logger.warning("Invalid action from LLM, defaulting to NOOP")
                return MemoryAction(operation=MemoryOperation.NOOP)
                
        except Exception as e:
            logger.error(f"LLM decision failed: {e}")
            return MemoryAction(operation=MemoryOperation.NOOP)
    
    def execute_action(self, action: MemoryAction) -> Optional[List[Memory]]:
        """Execute a memory action"""
        
        if action.operation == MemoryOperation.ADD:
            return self._execute_add(action)
        
        elif action.operation == MemoryOperation.UPDATE:
            return self._execute_update(action)
        
        elif action.operation == MemoryOperation.DELETE:
            return self._execute_delete(action)
        
        elif action.operation == MemoryOperation.CALLBACK:
            return self._execute_callback(action)
        
        elif action.operation == MemoryOperation.NOOP:
            return None
        
        return None
    
    def _execute_add(self, action: MemoryAction) -> List[Memory]:
        """Execute ADD operation"""
        
        memory = Memory(
            memory_id=str(uuid.uuid4()),
            content=action.content,
            memory_type=action.memory_type or "conversation",
            importance=action.importance or 0.5,
            conversation_turn=self.current_turn,
            tags=action.tags or []
        )
        
        self.db_client.add_memory(self.user_id, memory)
        
        logger.info(f"Added memory: {memory.content[:50]}...")
        return [memory]
    
    def _execute_update(self, action: MemoryAction) -> List[Memory]:
        """Execute UPDATE operation"""
        
        self.db_client.update_memory(
            user_id=self.user_id,
            memory_id=action.memory_id,
            new_content=action.new_content
        )
        
        logger.info(f"Updated memory {action.memory_id}")
        return []
    
    def _execute_delete(self, action: MemoryAction) -> List[Memory]:
        """Execute DELETE operation"""
        
        self.db_client.delete_memory(
            user_id=self.user_id,
            memory_id=action.memory_id
        )
        
        logger.info(f"Deleted memory {action.memory_id}")
        return []
    
    def _execute_callback(self, action: MemoryAction) -> List[Memory]:
        """Execute CALLBACK operation (retrieve memories)"""
        
        memories = self.db_client.retrieve_memories(
            user_id=self.user_id,
            query_text=action.callback_query,
            n_results=5
        )
        
        logger.info(f"Retrieved {len(memories)} memories for: {action.callback_query}")
        return memories
    
    def get_all_memories(self) -> List[Memory]:
        """Get all memories for the user"""
        return self.db_client.retrieve_memories(
            user_id=self.user_id,
            n_results=100
        )
    
    def compute_reward(
        self,
        predicted_features: dict,
        ground_truth_features: dict,
        memory_action: MemoryAction
    ) -> MemoryReward:
        """Compute reward for RL training"""
        
        reward = MemoryReward()
        
        # Final accuracy (feature prediction vs ground truth)
        if predicted_features and ground_truth_features:
            # Simple accuracy: proportion of matching features
            matches = sum(
                1 for k in predicted_features 
                if k in ground_truth_features and predicted_features[k] == ground_truth_features[k]
            )
            total = len(ground_truth_features)
            reward.final_accuracy = matches / total if total > 0 else 0.0
        
        # Information gain (did we learn something new?)
        if memory_action.operation == MemoryOperation.ADD:
            reward.information_gain = memory_action.importance or 0.5
        
        # Memory quality (importance score)
        if memory_action.operation in [MemoryOperation.ADD, MemoryOperation.UPDATE]:
            reward.memory_quality = memory_action.importance or 0.5
        
        # Retrieval relevance (for CALLBACK)
        if memory_action.operation == MemoryOperation.CALLBACK:
            reward.retrieval_relevance = 0.7  # Placeholder
        
        # Compute total
        reward.compute_total()
        
        return reward
