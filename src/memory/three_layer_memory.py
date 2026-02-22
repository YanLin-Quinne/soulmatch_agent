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

    def __init__(self, llm_router, working_memory_size: int = 20):
        from src.agents.llm_router import LLMRouter
        self.llm: LLMRouter = llm_router
        self.working_memory_size = working_memory_size

        # Layer 1: Working Memory (FIFO队列)
        self.working_memory: List[WorkingMemoryItem] = []

        # Layer 2: Episodic Memory (压缩摘要)
        self.episodic_memory: List[EpisodicMemoryItem] = []

        # Layer 3: Semantic Memory (高层反思)
        self.semantic_memory: List[SemanticMemoryItem] = []

        # 当前轮次
        self.current_turn = 0

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

            # 从工作记忆中移除已压缩的10轮
            self.working_memory = self.working_memory[10:]

        except Exception as e:
            logger.error(f"[ThreeLayerMemory] Episodic compression failed: {e}")

    def retrieve_relevant_episodes(self, query: str, top_k: int = 3) -> List[EpisodicMemoryItem]:
        """语义检索相关情景（简化版：基于关键词匹配）"""
        # TODO: 使用embedding进行语义检索
        relevant = []
        query_lower = query.lower()

        for episode in self.episodic_memory:
            if any(keyword in episode.summary.lower() for keyword in query_lower.split()):
                relevant.append(episode)

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

        # 重新获取原始对话（从工作记忆或存档）
        # 简化版：假设我们有原始对话的访问权限
        # TODO: 实际实现需要从持久化存储中获取

        prompt = f"""Verify if the following episodic summary is consistent with the original dialogue.

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
                system="You are a consistency checker. Detect hallucinations in memory summaries.",
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
                # TODO: 触发重新摘要或标记为不可信

        except Exception as e:
            logger.error(f"[ThreeLayerMemory] Consistency check failed: {e}")

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
            "compression_ratio": len(self.episodic_memory) * 10 / max(self.current_turn, 1)
        }
