"""Memory operations (ADD/UPDATE/DELETE/NOOP/CALLBACK) based on Memory-R1"""

from enum import Enum
from typing import Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime


class MemoryOperation(str, Enum):
    """Memory operation types (Memory-R1)"""
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NOOP = "NOOP"
    CALLBACK = "CALLBACK"  # ReMemR1 enhancement


class Memory(BaseModel):
    """Memory item"""
    
    memory_id: str
    content: str
    memory_type: str = "conversation"  # conversation, feature, fact, event
    importance: float = Field(ge=0.0, le=1.0, default=0.5)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    access_count: int = 0
    
    # Context
    conversation_turn: Optional[int] = None
    related_feature: Optional[str] = None  # e.g., "extraversion", "core_values"
    
    # Embedding
    embedding: Optional[list[float]] = None
    
    # Tags for retrieval
    tags: list[str] = Field(default_factory=list)
    
    def update_content(self, new_content: str):
        """Update memory content"""
        self.content = new_content
        self.updated_at = datetime.now()
    
    def increment_access(self):
        """Increment access count"""
        self.access_count += 1


class MemoryAction(BaseModel):
    """Memory action returned by agent"""
    
    operation: MemoryOperation
    
    # For ADD
    content: Optional[str] = None
    memory_type: Optional[str] = None
    importance: Optional[float] = None
    tags: Optional[list[str]] = None
    
    # For UPDATE
    memory_id: Optional[str] = None
    new_content: Optional[str] = None
    
    # For DELETE
    # memory_id is reused
    
    # For CALLBACK
    callback_query: Optional[str] = None  # Query to retrieve relevant memories
    
    # Reasoning
    reasoning: Optional[str] = None  # Why this operation was chosen
    
    def validate_action(self) -> bool:
        """Validate action has required fields"""
        
        if self.operation == MemoryOperation.ADD:
            return bool(self.content and self.memory_type)
        
        elif self.operation == MemoryOperation.UPDATE:
            return bool(self.memory_id and self.new_content)
        
        elif self.operation == MemoryOperation.DELETE:
            return bool(self.memory_id)
        
        elif self.operation == MemoryOperation.CALLBACK:
            return bool(self.callback_query)
        
        elif self.operation == MemoryOperation.NOOP:
            return True
        
        return False


class MemoryReward(BaseModel):
    """Reward signal for RL training (Memory-R1)"""
    
    # Final reward (end of conversation)
    final_accuracy: float = 0.0  # Feature prediction accuracy vs ground truth
    
    # Step-level rewards
    information_gain: float = 0.0  # How much new info was gained
    memory_quality: float = 0.0  # Quality of memory content
    retrieval_relevance: float = 0.0  # Relevance of recalled memories
    
    # Combined reward
    total_reward: float = 0.0
    
    def compute_total(
        self,
        alpha: float = 0.5,  # Weight for final accuracy
        beta: float = 0.3,   # Weight for information gain
        gamma: float = 0.2   # Weight for memory quality
    ):
        """Compute total reward"""
        self.total_reward = (
            alpha * self.final_accuracy +
            beta * self.information_gain +
            gamma * (self.memory_quality + self.retrieval_relevance) / 2
        )
        return self.total_reward
