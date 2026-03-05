"""
记忆聚合算子 - 显式聚合算法实现

核心创新：
1. 方差降低（Iterative Variance Reduction）- 多次采样+投票
2. 有限轮次收敛（Finite-Round Convergence）
3. 显式聚合算子（不依赖隐式 LLM 压缩）
"""
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from dataclasses import dataclass
import asyncio


@dataclass
class WorkingMemoryItem:
    """工作记忆项"""
    turn_number: int
    speaker: str
    content: str
    metadata: Dict[str, Any]


@dataclass
class EpisodicMemoryItem:
    """情景记忆项"""
    turn_range: Tuple[int, int]
    summary: str
    key_events: List[str]
    variance: float  # 新增：方差指标
    metadata: Dict[str, Any]


@dataclass
class SemanticMemoryItem:
    """语义记忆项"""
    feature_updates: Dict[str, float]
    patterns: List[str]
    logic_tree: Optional[Any]  # LogicTreeNode
    metadata: Dict[str, Any]


class MemoryAggregationOperator:
    """记忆聚合算子 - 显式算法实现"""

    def __init__(self, llm_router, embedding_model=None):
        self.llm_router = llm_router
        self.embedding_model = embedding_model

    async def aggregate_working_to_episodic(
        self,
        working_items: List[WorkingMemoryItem],
        method: str = "variance_reduction",
        num_samples: int = 3,
        max_refinement_rounds: int = 3
    ) -> EpisodicMemoryItem:
        """
        Working → Episodic 聚合

        方法：
        1. variance_reduction: 多次采样（默认3次）+ 投票选择最一致摘要
        2. self_refinement: 迭代提炼直到收敛（最多3轮）
        3. hybrid: 先方差降低，再自我提炼
        """
        if method == "variance_reduction":
            return await self._variance_reduction_aggregation(working_items, num_samples)
        elif method == "self_refinement":
            return await self._self_refinement_aggregation(working_items, max_refinement_rounds)
        elif method == "hybrid":
            # 先方差降低生成候选
            candidate = await self._variance_reduction_aggregation(working_items, num_samples)
            # 再自我提炼优化
            return await self._refine_episodic_memory(candidate, working_items, max_refinement_rounds)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    async def _variance_reduction_aggregation(
        self,
        working_items: List[WorkingMemoryItem],
        num_samples: int
    ) -> EpisodicMemoryItem:
        """
        方差降低聚合 - 多次采样 + 投票

        算法：
        1. 使用不同温度参数采样 N 次（temperature=0.7, 0.8, 0.9）
        2. 计算所有摘要的语义相似度矩阵
        3. 选择平均相似度最高的摘要（最一致）
        4. 计算方差指标（1 - 平均相似度）
        """
        # 构建 prompt
        conversation_text = "\n".join([
            f"Turn {item.turn_number} - {item.speaker}: {item.content}"
            for item in working_items
        ])

        prompt = f"""请总结以下对话片段，提取关键事件和情感变化。

对话内容:
{conversation_text}

请按以下格式输出：

<summary>
[1-2句话的简洁摘要]
</summary>

<key_events>
- [关键事件1]
- [关键事件2]
- [关键事件3]
</key_events>
"""

        # 多次采样（使用不同温度）
        temperatures = [0.7, 0.8, 0.9][:num_samples]
        summaries = []

        tasks = []
        for temp in temperatures:
            task = self.llm_router.chat(
                messages=[{"role": "user", "content": prompt}],
                role="memory_aggregator",
                temperature=temp
            )
            tasks.append(task)

        # 并行执行
        responses = await asyncio.gather(*tasks)
        summaries = [self._parse_summary_response(resp) for resp in responses]

        # 计算方差并选择最一致的摘要
        best_summary, variance = await self._select_best_summary(summaries)

        turn_range = (
            working_items[0].turn_number,
            working_items[-1].turn_number
        )

        return EpisodicMemoryItem(
            turn_range=turn_range,
            summary=best_summary["summary"],
            key_events=best_summary["key_events"],
            variance=variance,
            metadata={"method": "variance_reduction", "num_samples": num_samples}
        )

    async def _self_refinement_aggregation(
        self,
        working_items: List[WorkingMemoryItem],
        max_rounds: int
    ) -> EpisodicMemoryItem:
        """
        自我提炼聚合 - 迭代压缩直到收敛

        算法：
        1. 初始压缩
        2. 迭代提炼（最多 max_rounds 轮）
        3. 检测收敛（相似度 > 0.95）
        """
        conversation_text = "\n".join([
            f"Turn {item.turn_number} - {item.speaker}: {item.content}"
            for item in working_items
        ])

        # 初始压缩
        initial_prompt = f"""请总结以下对话片段：

{conversation_text}

输出格式：
<summary>[摘要]</summary>
<key_events>
- [事件1]
- [事件2]
</key_events>
"""

        response = await self.llm_router.chat(
            messages=[{"role": "user", "content": initial_prompt}],
            role="memory_aggregator",
            temperature=0.7
        )

        current_summary = self._parse_summary_response(response)

        # 迭代提炼
        for round_num in range(max_rounds):
            refined_prompt = f"""请优化以下摘要，使其更简洁准确：

原始对话:
{conversation_text}

当前摘要:
{current_summary['summary']}

关键事件:
{chr(10).join(['- ' + e for e in current_summary['key_events']])}

请提供改进版本（相同格式）。
"""

            response = await self.llm_router.chat(
                messages=[{"role": "user", "content": refined_prompt}],
                role="memory_aggregator",
                temperature=0.5  # 降低温度保证收敛
            )

            refined_summary = self._parse_summary_response(response)

            # 检测收敛
            similarity = await self._compute_similarity(
                current_summary["summary"],
                refined_summary["summary"]
            )

            if similarity > 0.95:
                # 已收敛
                break

            current_summary = refined_summary

        turn_range = (
            working_items[0].turn_number,
            working_items[-1].turn_number
        )

        return EpisodicMemoryItem(
            turn_range=turn_range,
            summary=current_summary["summary"],
            key_events=current_summary["key_events"],
            variance=1.0 - similarity,  # 使用最后一轮的相似度
            metadata={"method": "self_refinement", "rounds": round_num + 1}
        )

    async def _select_best_summary(
        self,
        summaries: List[Dict[str, Any]]
    ) -> Tuple[Dict[str, Any], float]:
        """
        选择最一致的摘要

        返回：(最佳摘要, 方差)
        """
        if len(summaries) == 1:
            return summaries[0], 0.0

        # 计算 pairwise 相似度矩阵
        n = len(summaries)
        similarity_matrix = np.zeros((n, n))

        for i in range(n):
            for j in range(i + 1, n):
                sim = await self._compute_similarity(
                    summaries[i]["summary"],
                    summaries[j]["summary"]
                )
                similarity_matrix[i, j] = sim
                similarity_matrix[j, i] = sim

        # 计算每个摘要的平均相似度
        avg_similarities = similarity_matrix.mean(axis=1)

        # 选择平均相似度最高的
        best_idx = int(np.argmax(avg_similarities))
        best_summary = summaries[best_idx]

        # 计算方差（1 - 平均相似度）
        variance = 1.0 - float(avg_similarities[best_idx])

        return best_summary, variance

    async def _compute_similarity(self, text1: str, text2: str) -> float:
        """
        计算两个文本的语义相似度

        使用 embedding 模型计算 cosine similarity
        """
        if self.embedding_model is None:
            # Fallback: 使用简单的 Jaccard 相似度
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

        # 使用 embedding 模型
        try:
            # 这里假设 embedding_model 有 encode 方法
            emb1 = self.embedding_model.encode(text1)
            emb2 = self.embedding_model.encode(text2)

            # Cosine similarity
            similarity = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
            return float(similarity)
        except Exception:
            # Fallback
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

    def _parse_summary_response(self, response: str) -> Dict[str, Any]:
        """解析 LLM 返回的摘要"""
        import re

        summary_match = re.search(r'<summary>(.*?)</summary>', response, re.DOTALL)
        summary = summary_match.group(1).strip() if summary_match else ""

        key_events_match = re.search(r'<key_events>(.*?)</key_events>', response, re.DOTALL)
        key_events_text = key_events_match.group(1).strip() if key_events_match else ""

        # 提取列表项
        key_events = []
        for line in key_events_text.split('\n'):
            line = line.strip()
            if line.startswith('-') or line.startswith('•'):
                key_events.append(line[1:].strip())

        return {
            "summary": summary,
            "key_events": key_events
        }

    async def aggregate_episodic_to_semantic(
        self,
        episodic_items: List[EpisodicMemoryItem],
        current_features: Dict[str, float]
    ) -> SemanticMemoryItem:
        """
        Episodic → Semantic 聚合

        方法：
        1. 特征变化检测（显式）
        2. 模式识别（聚类）
        3. 因果推理（LogicTree）
        """
        # 构建 prompt
        episodes_text = "\n\n".join([
            f"Episode {i+1} (Turns {item.turn_range[0]}-{item.turn_range[1]}):\n"
            f"Summary: {item.summary}\n"
            f"Key Events: {', '.join(item.key_events)}"
            for i, item in enumerate(episodic_items)
        ])

        prompt = f"""你是一个心理学专家。请分析以下情景记忆，提取高层模式和特征变化。

情景记忆:
{episodes_text}

当前特征:
{current_features}

请输出：

<feature_updates>
[特征名]: [变化值] (例如: openness: +0.05)
</feature_updates>

<patterns>
- [模式1]
- [模式2]
</patterns>
"""

        response = await self.llm_router.chat(
            messages=[{"role": "user", "content": prompt}],
            role="semantic_memory_builder",
            temperature=0.5
        )

        # 解析响应
        feature_updates = self._parse_feature_updates(response)
        patterns = self._parse_patterns(response)

        return SemanticMemoryItem(
            feature_updates=feature_updates,
            patterns=patterns,
            logic_tree=None,  # 可选：构建因果推理树
            metadata={"num_episodes": len(episodic_items)}
        )

    def _parse_feature_updates(self, response: str) -> Dict[str, float]:
        """解析特征更新"""
        import re

        updates = {}
        updates_match = re.search(r'<feature_updates>(.*?)</feature_updates>', response, re.DOTALL)

        if updates_match:
            updates_text = updates_match.group(1).strip()
            for line in updates_text.split('\n'):
                match = re.search(r'(\w+):\s*([\+\-]?[\d.]+)', line)
                if match:
                    feature_name = match.group(1)
                    change = float(match.group(2))
                    updates[feature_name] = change

        return updates

    def _parse_patterns(self, response: str) -> List[str]:
        """解析模式"""
        import re

        patterns = []
        patterns_match = re.search(r'<patterns>(.*?)</patterns>', response, re.DOTALL)

        if patterns_match:
            patterns_text = patterns_match.group(1).strip()
            for line in patterns_text.split('\n'):
                line = line.strip()
                if line.startswith('-') or line.startswith('•'):
                    patterns.append(line[1:].strip())

        return patterns

    async def _refine_episodic_memory(
        self,
        memory: EpisodicMemoryItem,
        working_items: List[WorkingMemoryItem],
        max_rounds: int
    ) -> EpisodicMemoryItem:
        """提炼情景记忆（用于 hybrid 方法）"""
        # 简化版：只做一轮提炼
        conversation_text = "\n".join([
            f"Turn {item.turn_number} - {item.speaker}: {item.content}"
            for item in working_items
        ])

        prompt = f"""请优化以下摘要：

原始对话:
{conversation_text}

当前摘要:
{memory.summary}

请提供改进版本（更简洁准确）。
"""

        response = await self.llm_router.chat(
            messages=[{"role": "user", "content": prompt}],
            role="memory_refiner",
            temperature=0.5
        )

        refined = self._parse_summary_response(response)

        return EpisodicMemoryItem(
            turn_range=memory.turn_range,
            summary=refined["summary"],
            key_events=refined["key_events"],
            variance=memory.variance,
            metadata={**memory.metadata, "refined": True}
        )
