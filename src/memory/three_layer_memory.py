"""
Three-Layer Memory System with Anti-Hallucination Mechanisms

Architecture:
  Layer 1: Working Memory (最近20轮原始对话，滑动窗口)
  Layer 2: Episodic Memory (每10轮LLM压缩摘要 + 关键事件)
  Layer 3: Semantic Memory (每50轮反思 + 特征更新)

Anti-Hallucination:
  - Strict grounding: 所有摘要必须引用轮次编号
  - Consistency check: 每20轮独立LLM验证
  - Conflict resolution: 新旧记忆冲突处理
"""

import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger

if TYPE_CHECKING:
    from src.agents.llm_router import LLMRouter, AgentRole
else:
    from src.agents.llm_router import AgentRole


@dataclass
class WorkingMemoryItem:
    """工作记忆项（原始对话）"""
    turn: int
    speaker: str
    message: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class EpisodicMemoryItem:
    """情景记忆项（压缩摘要）"""
    episode_id: str
    turn_range: tuple[int, int]  # (start_turn, end_turn)
    summary: str
    key_events: List[str]  # ["conflict", "repair", "milestone"]
    emotion_trend: str  # "improving", "declining", "stable"
    participants: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    embedding: Optional[List[float]] = None


@dataclass
class SemanticMemoryItem:
    """语义记忆项（高层反思）"""
    reflection_id: str
    turn_range: tuple[int, int]
    reflection: str
    feature_updates: Dict[str, Any]  # 特征变化
    relationship_insights: str
    timestamp: datetime = field(default_factory=datetime.now)


class ThreeLayerMemory:
    """
    三层记忆系统

    核心创新：
    1. 分层存储：工作记忆→情景记忆→语义记忆
    2. 自动压缩：每10轮压缩，每50轮反思
    3. 防幻觉：严格grounding + 一致性校验
    """

    def __init__(self, llm_router, working_memory_size: int = 20, user_id: str = "default"):
        from src.agents.llm_router import LLMRouter
        self.llm: LLMRouter = llm_router
        self.working_memory_size = working_memory_size
        self.user_id = user_id

        # Layer 1: Working Memory (FIFO队列)
        self.working_memory: List[WorkingMemoryItem] = []

        # Layer 2: Episodic Memory (压缩摘要)
        self.episodic_memory: List[EpisodicMemoryItem] = []

        # Layer 3: Semantic Memory (高层反思)
        self.semantic_memory: List[SemanticMemoryItem] = []

        # 当前轮次
        self.current_turn = 0

        # ChromaDB for embedding-based retrieval
        try:
            from src.memory.chromadb_client import ChromaDBClient
            self.chroma_client = ChromaDBClient()
            self.chroma_collection = self.chroma_client.get_or_create_collection(user_id)
            self.use_embeddings = True
            logger.info("[ThreeLayerMemory] ChromaDB enabled for semantic retrieval")
        except Exception as e:
            logger.warning(f"[ThreeLayerMemory] ChromaDB not available: {e}. Falling back to keyword matching.")
            self.chroma_client = None
            self.chroma_collection = None
            self.use_embeddings = False

        # Archive for consistency checking (stores original dialogues)
        self.dialogue_archive: Dict[tuple, List[WorkingMemoryItem]] = {}  # {(start_turn, end_turn): [items]}

    # ========== Layer 1: Working Memory ==========

    def add_to_working_memory(self, speaker: str, message: str):
        """添加到工作记忆（滑动窗口）"""
        item = WorkingMemoryItem(
            turn=self.current_turn,
            speaker=speaker,
            message=message
        )
        self.working_memory.append(item)
        self.current_turn += 1

        # FIFO: 超过容量则移除最老的
        if len(self.working_memory) > self.working_memory_size:
            self.working_memory.pop(0)

        # 触发压缩（每10轮）
        if self.current_turn % 10 == 0:
            self._compress_to_episodic()

        # 触发反思（每50轮）
        if self.current_turn % 50 == 0:
            self._reflect_to_semantic()

        # 一致性校验（每20轮）
        if self.current_turn % 20 == 0:
            self._consistency_check()

    def get_working_memory_context(self) -> str:
        """获取工作记忆上下文（注入LLM prompt）"""
        if not self.working_memory:
            return ""

        context = "Recent conversation (working memory):\n"
        for item in self.working_memory:
            context += f"[Turn {item.turn}] {item.speaker}: {item.message}\n"
        return context

    # ========== Layer 2: Episodic Memory ==========

    def _compress_to_episodic(self):
        """每10轮：压缩工作记忆到情景记忆"""
        if len(self.working_memory) < 10:
            return

        # 取最老的10轮
        oldest_10 = self.working_memory[:10]
        turn_range = (oldest_10[0].turn, oldest_10[-1].turn)

        # LLM压缩摘要
        dialogue_text = "\n".join([
            f"[Turn {item.turn}] {item.speaker}: {item.message}"
            for item in oldest_10
        ])

        prompt = f"""Summarize the following 10-turn conversation into a concise episodic memory.

Dialogue:
{dialogue_text}

Output JSON:
{{
  "summary": "Brief summary (2-3 sentences)",
  "key_events": ["conflict", "repair", "milestone", "disclosure", etc.],
  "emotion_trend": "improving|declining|stable",
  "participants": ["user", "bot"]
}}

IMPORTANT:
- Summary MUST reference specific turn numbers (e.g., "At turn 5, user disclosed...")
- Only include events that actually happened in the dialogue
- Do NOT hallucinate or infer events not present in the text
"""

        try:
            response = self.llm.chat(
                role=AgentRole.MEMORY,
                system="You are a memory compression agent. Extract key information without hallucination.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                json_mode=True
            )

            # 解析JSON
            import re
            clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
            data = json.loads(clean_response)

            # 创建情景记忆
            episode = EpisodicMemoryItem(
                episode_id=f"ep_{turn_range[0]}_{turn_range[1]}",
                turn_range=turn_range,
                summary=data["summary"],
                key_events=data.get("key_events", []),
                emotion_trend=data.get("emotion_trend", "stable"),
                participants=data.get("participants", ["user", "bot"])
            )

            self.episodic_memory.append(episode)
            logger.info(f"[ThreeLayerMemory] Compressed turns {turn_range[0]}-{turn_range[1]} to episodic memory")

            # 保存原始对话到archive（用于一致性检查）
            self.dialogue_archive[turn_range] = list(oldest_10)

            # 存储到ChromaDB（用于embedding检索）
            if self.use_embeddings and self.chroma_collection:
                try:
                    self.chroma_collection.add(
                        ids=[episode.episode_id],
                        documents=[episode.summary],
                        metadatas=[{
                            "turn_start": turn_range[0],
                            "turn_end": turn_range[1],
                            "emotion_trend": episode.emotion_trend,
                            "key_events": ",".join(episode.key_events)
                        }]
                    )
                    logger.debug(f"[ThreeLayerMemory] Stored episode {episode.episode_id} to ChromaDB")
                except Exception as e:
                    logger.warning(f"[ThreeLayerMemory] Failed to store to ChromaDB: {e}")

            # 从工作记忆中移除已压缩的10轮
            self.working_memory = self.working_memory[10:]

        except Exception as e:
            logger.error(f"[ThreeLayerMemory] Episodic compression failed: {e}")

    def retrieve_relevant_episodes(self, query: str, top_k: int = 3) -> List[EpisodicMemoryItem]:
        """语义检索相关情景（使用embedding或关键词匹配）"""

        # 优先使用embedding检索
        if self.use_embeddings and self.chroma_collection:
            try:
                results = self.chroma_collection.query(
                    query_texts=[query],
                    n_results=min(top_k, len(self.episodic_memory))
                )

                if results and results['ids'] and len(results['ids'][0]) > 0:
                    # 根据返回的episode_id找到对应的EpisodicMemoryItem
                    episode_ids = results['ids'][0]
                    relevant = []
                    for ep_id in episode_ids:
                        for episode in self.episodic_memory:
                            if episode.episode_id == ep_id:
                                relevant.append(episode)
                                break

                    logger.debug(f"[ThreeLayerMemory] Retrieved {len(relevant)} episodes via embedding search")
                    return relevant
            except Exception as e:
                logger.warning(f"[ThreeLayerMemory] Embedding retrieval failed: {e}. Falling back to keyword matching.")

        # Fallback: 关键词匹配
        relevant = []
        query_lower = query.lower()

        for episode in self.episodic_memory:
            if any(keyword in episode.summary.lower() for keyword in query_lower.split()):
                relevant.append(episode)

        logger.debug(f"[ThreeLayerMemory] Retrieved {len(relevant[:top_k])} episodes via keyword matching")
        return relevant[:top_k]

    def get_episodic_memory_context(self, query: Optional[str] = None) -> str:
        """获取情景记忆上下文"""
        if not self.episodic_memory:
            return ""

        if query:
            episodes = self.retrieve_relevant_episodes(query)
        else:
            episodes = self.episodic_memory[-3:]  # 最近3个情景

        context = "Relevant episodic memories:\n"
        for ep in episodes:
            context += f"[Turns {ep.turn_range[0]}-{ep.turn_range[1]}] {ep.summary}\n"
            if ep.key_events:
                context += f"  Key events: {', '.join(ep.key_events)}\n"
        return context

    # ========== Layer 3: Semantic Memory ==========

    def _reflect_to_semantic(self):
        """每50轮：反思情景记忆，更新语义记忆"""
        if len(self.episodic_memory) < 5:
            return

        # 取最近5个情景
        recent_episodes = self.episodic_memory[-5:]
        turn_range = (recent_episodes[0].turn_range[0], recent_episodes[-1].turn_range[1])

        # LLM反思
        episodes_text = "\n\n".join([
            f"Episode {i+1} (Turns {ep.turn_range[0]}-{ep.turn_range[1]}):\n{ep.summary}\nKey events: {', '.join(ep.key_events)}"
            for i, ep in enumerate(recent_episodes)
        ])

        prompt = f"""Reflect on the following 5 episodic memories and extract high-level insights.

Episodes:
{episodes_text}

Output JSON:
{{
  "reflection": "High-level reflection on relationship development (3-4 sentences)",
  "feature_updates": {{
    "openness": {{"change": "+0.05", "reason": "User showed curiosity..."}},
    "trust_score": {{"change": "+0.10", "reason": "Consistent positive interactions..."}}
  }},
  "relationship_insights": "Key patterns observed (e.g., 'User tends to avoid conflict but responds well to patience')"
}}

IMPORTANT:
- Reflection MUST be grounded in the episodes provided
- Feature updates MUST have specific reasons from the dialogue
- Do NOT hallucinate patterns not present in the episodes
"""

        try:
            response = self.llm.chat(
                role=AgentRole.MEMORY,
                system="You are a reflective memory agent. Extract patterns without hallucination.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=600,
                json_mode=True
            )

            # 解析JSON
            import re
            clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
            data = json.loads(clean_response)

            # 创建语义记忆
            semantic = SemanticMemoryItem(
                reflection_id=f"sem_{turn_range[0]}_{turn_range[1]}",
                turn_range=turn_range,
                reflection=data["reflection"],
                feature_updates=data.get("feature_updates", {}),
                relationship_insights=data.get("relationship_insights", "")
            )

            self.semantic_memory.append(semantic)
            logger.info(f"[ThreeLayerMemory] Reflected on turns {turn_range[0]}-{turn_range[1]} to semantic memory")

        except Exception as e:
            logger.error(f"[ThreeLayerMemory] Semantic reflection failed: {e}")

    def get_semantic_memory_context(self) -> str:
        """获取语义记忆上下文"""
        if not self.semantic_memory:
            return ""

        latest = self.semantic_memory[-1]
        context = f"High-level reflection (Turns {latest.turn_range[0]}-{latest.turn_range[1]}):\n"
        context += f"{latest.reflection}\n"
        context += f"Relationship insights: {latest.relationship_insights}\n"
        return context

    # ========== Anti-Hallucination Mechanisms ==========

    def _consistency_check(self):
        """每20轮：一致性校验（防幻觉）"""
        if not self.episodic_memory:
            return

        # 检查最近的情景记忆
        latest_episode = self.episodic_memory[-1]

        # 从archive获取原始对话
        turn_range = latest_episode.turn_range
        original_dialogue = self.dialogue_archive.get(turn_range)

        if not original_dialogue:
            logger.warning(f"[ThreeLayerMemory] No archived dialogue found for {turn_range}. Skipping consistency check.")
            return

        # 构建原始对话文本
        dialogue_text = "\n".join([
            f"[Turn {item.turn}] {item.speaker}: {item.message}"
            for item in original_dialogue
        ])

        prompt = f"""Verify if the following episodic summary is consistent with the original dialogue.

Original Dialogue:
{dialogue_text}

Episodic Summary:
{latest_episode.summary}

Task: Check for hallucinations or inconsistencies.
Output JSON:
{{
  "is_consistent": true|false,
  "hallucinations": ["list of hallucinated facts"],
  "missing_info": ["important info not captured"]
}}
"""

        try:
            response = self.llm.chat(
                role=AgentRole.MEMORY,
                system="You are a consistency checker. Detect hallucinations in memory summaries by comparing with original dialogue.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                json_mode=True
            )

            import re
            clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
            data = json.loads(clean_response)

            if not data.get("is_consistent", True):
                logger.warning(f"[ThreeLayerMemory] Inconsistency detected in episode {latest_episode.episode_id}")
                logger.warning(f"  Hallucinations: {data.get('hallucinations', [])}")
                logger.warning(f"  Missing info: {data.get('missing_info', [])}")

                # 处理不一致：重新生成摘要
                self._fix_inconsistent_episode(latest_episode, original_dialogue, data)

        except Exception as e:
            logger.error(f"[ThreeLayerMemory] Consistency check failed: {e}")

    def _fix_inconsistent_episode(self, episode: EpisodicMemoryItem, original_dialogue: List[WorkingMemoryItem], inconsistency_data: Dict):
        """修复不一致的情景记忆"""
        logger.info(f"[ThreeLayerMemory] Attempting to fix inconsistent episode {episode.episode_id}")

        # 重新生成摘要
        dialogue_text = "\n".join([
            f"[Turn {item.turn}] {item.speaker}: {item.message}"
            for item in original_dialogue
        ])

        prompt = f"""Re-summarize the following conversation. Previous summary had hallucinations: {inconsistency_data.get('hallucinations', [])}.

Dialogue:
{dialogue_text}

Output JSON:
{{
  "summary": "Accurate summary (2-3 sentences)",
  "key_events": ["event1", "event2"],
  "emotion_trend": "improving|declining|stable"
}}

CRITICAL: Only include facts explicitly present in the dialogue. Do NOT infer or hallucinate.
"""

        try:
            response = self.llm.chat(
                role=AgentRole.MEMORY,
                system="You are a memory compression agent. Extract key information without hallucination.",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=400,
                json_mode=True
            )

            import re
            clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
            data = json.loads(clean_response)

            # 更新episode
            episode.summary = data["summary"]
            episode.key_events = data.get("key_events", [])
            episode.emotion_trend = data.get("emotion_trend", "stable")

            # 更新ChromaDB
            if self.use_embeddings and self.chroma_collection:
                try:
                    self.chroma_collection.update(
                        ids=[episode.episode_id],
                        documents=[episode.summary],
                        metadatas=[{
                            "turn_start": episode.turn_range[0],
                            "turn_end": episode.turn_range[1],
                            "emotion_trend": episode.emotion_trend,
                            "key_events": ",".join(episode.key_events)
                        }]
                    )
                    logger.info(f"[ThreeLayerMemory] Fixed and updated episode {episode.episode_id}")
                except Exception as e:
                    logger.warning(f"[ThreeLayerMemory] Failed to update ChromaDB: {e}")

        except Exception as e:
            logger.error(f"[ThreeLayerMemory] Failed to fix inconsistent episode: {e}")
            # 标记为不可信
            episode.summary = f"[UNRELIABLE] {episode.summary}"
            logger.warning(f"[ThreeLayerMemory] Marked episode {episode.episode_id} as unreliable")

    # ========== Unified Context Retrieval ==========

    def get_full_context(self, query: Optional[str] = None) -> str:
        """获取完整的三层记忆上下文（注入LLM prompt）"""
        context = ""

        # Layer 1: Working Memory (最近对话)
        context += self.get_working_memory_context()
        context += "\n"

        # Layer 2: Episodic Memory (相关情景)
        context += self.get_episodic_memory_context(query)
        context += "\n"

        # Layer 3: Semantic Memory (高层反思)
        context += self.get_semantic_memory_context()

        return context

    # ========== Statistics ==========

    def get_memory_stats(self) -> Dict[str, Any]:
        """获取记忆统计信息"""
        return {
            "current_turn": self.current_turn,
            "working_memory_size": len(self.working_memory),
            "episodic_memory_count": len(self.episodic_memory),
            "semantic_memory_count": len(self.semantic_memory),
            "total_turns_covered": self.current_turn,
            "compression_ratio": len(self.episodic_memory) * 10 / max(self.current_turn, 1),
            "episodic_memories": [
                {
                    "episode_id": ep.episode_id,
                    "turn_range": list(ep.turn_range),
                    "summary": ep.summary,
                    "key_events": ep.key_events,
                    "emotion_trend": ep.emotion_trend,
                }
                for ep in self.episodic_memory
            ],
            "semantic_memories": [
                {
                    "reflection_id": sem.reflection_id,
                    "turn_range": list(sem.turn_range),
                    "reflection": sem.reflection,
                    "feature_updates": sem.feature_updates,
                    "relationship_insights": sem.relationship_insights,
                }
                for sem in self.semantic_memory
            ],
        }
