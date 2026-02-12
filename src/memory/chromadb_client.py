"""ChromaDB client for vector memory storage"""

from typing import Optional, List
from pathlib import Path
from loguru import logger

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("chromadb not installed. Install with: pip install chromadb")

from src.memory.memory_operations import Memory
from src.config import settings


class ChromaDBClient:
    """ChromaDB client for memory storage"""
    
    def __init__(self, db_path: Optional[Path] = None):
        if not CHROMADB_AVAILABLE:
            raise ImportError("chromadb not installed")
        
        self.db_path = db_path or Path(settings.chroma_db_path)
        self.db_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=Settings(anonymized_telemetry=False)
        )
        
        logger.info(f"ChromaDB initialized at {self.db_path}")
    
    def get_or_create_collection(self, user_id: str):
        """Get or create collection for a user"""
        
        collection_name = f"user_{user_id}_memories"
        
        # Create or get collection
        collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"user_id": user_id}
        )
        
        return collection
    
    def add_memory(
        self, 
        user_id: str, 
        memory: Memory,
        embedding: Optional[list[float]] = None
    ):
        """Add a memory to the collection"""
        
        collection = self.get_or_create_collection(user_id)
        
        # Prepare metadata
        metadata = {
            "memory_type": memory.memory_type,
            "importance": memory.importance,
            "created_at": memory.created_at.isoformat(),
            "conversation_turn": memory.conversation_turn or -1,
            "access_count": memory.access_count,
        }
        
        if memory.related_feature:
            metadata["related_feature"] = memory.related_feature
        
        if memory.tags:
            metadata["tags"] = ",".join(memory.tags)
        
        # Add to collection
        collection.add(
            ids=[memory.memory_id],
            embeddings=[embedding] if embedding else None,
            documents=[memory.content],
            metadatas=[metadata]
        )
        
        logger.debug(f"Added memory {memory.memory_id} for user {user_id}")
    
    def update_memory(
        self, 
        user_id: str, 
        memory_id: str,
        new_content: str,
        embedding: Optional[list[float]] = None
    ):
        """Update a memory"""
        
        collection = self.get_or_create_collection(user_id)
        
        # Get existing metadata
        result = collection.get(ids=[memory_id], include=["metadatas"])
        
        if not result["ids"]:
            logger.warning(f"Memory {memory_id} not found")
            return
        
        metadata = result["metadatas"][0]
        metadata["updated_at"] = Memory(
            memory_id=memory_id,
            content=""
        ).updated_at.isoformat()
        
        # Update
        collection.update(
            ids=[memory_id],
            embeddings=[embedding] if embedding else None,
            documents=[new_content],
            metadatas=[metadata]
        )
        
        logger.debug(f"Updated memory {memory_id} for user {user_id}")
    
    def delete_memory(self, user_id: str, memory_id: str):
        """Delete a memory"""
        
        collection = self.get_or_create_collection(user_id)
        collection.delete(ids=[memory_id])
        
        logger.debug(f"Deleted memory {memory_id} for user {user_id}")
    
    def retrieve_memories(
        self,
        user_id: str,
        query_embedding: Optional[list[float]] = None,
        query_text: Optional[str] = None,
        n_results: int = 5,
        filter_metadata: Optional[dict] = None
    ) -> List[Memory]:
        """Retrieve memories by similarity or filter"""
        
        collection = self.get_or_create_collection(user_id)
        
        # Query
        if query_embedding:
            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                where=filter_metadata
            )
        elif query_text:
            results = collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where=filter_metadata
            )
        else:
            # Get all
            results = collection.get(
                limit=n_results,
                where=filter_metadata
            )
        
        # Convert to Memory objects
        memories = []
        
        ids = results.get("ids", [[]])[0] if "ids" in results else results.get("ids", [])
        documents = results.get("documents", [[]])[0] if "documents" in results else results.get("documents", [])
        metadatas = results.get("metadatas", [[]])[0] if "metadatas" in results else results.get("metadatas", [])
        
        for memory_id, content, metadata in zip(ids, documents, metadatas):
            # Parse tags
            tags = metadata.get("tags", "").split(",") if metadata.get("tags") else []
            
            memory = Memory(
                memory_id=memory_id,
                content=content,
                memory_type=metadata.get("memory_type", "conversation"),
                importance=metadata.get("importance", 0.5),
                created_at=metadata.get("created_at"),
                access_count=metadata.get("access_count", 0),
                conversation_turn=metadata.get("conversation_turn"),
                related_feature=metadata.get("related_feature"),
                tags=tags
            )
            
            memories.append(memory)
        
        return memories
    
    def get_memory_count(self, user_id: str) -> int:
        """Get total memory count for user"""
        
        collection = self.get_or_create_collection(user_id)
        return collection.count()
    
    def clear_user_memories(self, user_id: str):
        """Clear all memories for a user"""
        
        collection_name = f"user_{user_id}_memories"
        
        try:
            self.client.delete_collection(name=collection_name)
            logger.info(f"Cleared all memories for user {user_id}")
        except Exception as e:
            logger.warning(f"Failed to clear memories: {e}")
