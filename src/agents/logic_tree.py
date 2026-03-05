"""
LogicTree - 显式三段论推理树实现

核心创新：将社交匹配问题分解为结构化的三段论逻辑树，
提供可追溯性和可比较性，无需 RAG 或微调。
"""
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum


class NodeType(Enum):
    """逻辑树节点类型"""
    MAJOR_PREMISE = "major_premise"  # 大前提（普遍规律）
    MINOR_PREMISE = "minor_premise"  # 小前提（具体观察）
    CONCLUSION = "conclusion"         # 结论（推理结果）


@dataclass
class LogicTreeNode:
    """逻辑树节点 - 三段论推理单元"""
    node_type: NodeType
    content: str
    confidence: float  # 0.0-1.0
    evidence: List[str] = field(default_factory=list)  # 支持证据
    children: List['LogicTreeNode'] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_syllogism(self) -> str:
        """转换为三段论格式"""
        if self.node_type == NodeType.CONCLUSION:
            major = self.children[0] if len(self.children) > 0 else None
            minor = self.children[1] if len(self.children) > 1 else None

            return f"""
=== 三段论推理 ===
大前提 (Major Premise): {major.content if major else 'N/A'}
  置信度: {major.confidence:.2f}
  证据: {', '.join(major.evidence) if major and major.evidence else '无'}

小前提 (Minor Premise): {minor.content if minor else 'N/A'}
  置信度: {minor.confidence:.2f}
  证据: {', '.join(minor.evidence) if minor and minor.evidence else '无'}

结论 (Conclusion): {self.content}
  置信度: {self.confidence:.2f}
  证据: {', '.join(self.evidence) if self.evidence else '无'}
==================
"""
        return f"[{self.node_type.value}] {self.content} (conf={self.confidence:.2f})"

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于序列化）"""
        return {
            "node_type": self.node_type.value,
            "content": self.content,
            "confidence": self.confidence,
            "evidence": self.evidence,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata
        }


class LogicTreeBuilder:
    """逻辑树构建器 - 从对话和上下文构建三段论推理树"""

    def __init__(self, llm_router):
        self.llm_router = llm_router

    async def build_relationship_logic_tree(
        self,
        conversation_history: List[Dict[str, str]],
        current_features: Dict[str, Any],
        trust_score: float,
        turn_number: int
    ) -> LogicTreeNode:
        """
        构建关系预测的逻辑树

        示例推理：
        大前提: 当用户在连续对话中表现出高信任度（trust>0.8）且主动分享个人信息时，
               通常表明关系正在从 acquaintance 向 crush 发展
        小前提: 用户在过去5轮中 trust_score=0.85，主动分享了3次个人经历
        结论: 关系状态应从 acquaintance 升级到 crush（置信度 0.82）
        """
        # 构建 prompt
        recent_turns = conversation_history[-10:] if len(conversation_history) > 10 else conversation_history
        conversation_text = "\n".join([
            f"Turn {i+1} - {msg['speaker']}: {msg['content']}"
            for i, msg in enumerate(recent_turns)
        ])

        prompt = f"""你是一个逻辑推理专家。请基于以下信息构建一个三段论推理树，用于预测关系状态。

对话历史（最近10轮）:
{conversation_text}

当前特征:
- Trust Score: {trust_score:.2f}
- Turn Number: {turn_number}
- 预测特征: {current_features}

请按以下格式输出三段论推理：

<major_premise>
[描述普遍规律，例如：当用户表现出X行为模式时，通常意味着Y关系状态]
置信度: [0.0-1.0]
证据: [支持这个规律的理论依据或统计规律]
</major_premise>

<minor_premise>
[描述具体观察，例如：用户在过去N轮中表现出了X行为]
置信度: [0.0-1.0]
证据: [具体的对话轮次和行为实例]
</minor_premise>

<conclusion>
[推理结论，例如：因此关系状态应该是Z]
置信度: [0.0-1.0]
证据: [综合推理依据]
</conclusion>
"""

        response = await self.llm_router.chat(
            messages=[{"role": "user", "content": prompt}],
            role="logic_tree_builder",
            temperature=0.3  # 低温度保证逻辑一致性
        )

        # 解析响应
        return self._parse_syllogism_response(response)

    async def build_feature_logic_tree(
        self,
        conversation_history: List[Dict[str, str]],
        feature_name: str,
        current_value: Any,
        current_confidence: float
    ) -> LogicTreeNode:
        """
        构建特征预测的逻辑树

        示例推理：
        大前提: 频繁使用情感词汇（如"感觉"、"喜欢"）的用户通常具有高 openness 特征
        小前提: 用户在过去8轮中使用了12次情感词汇
        结论: 用户的 openness 特征值应为 0.75（置信度 0.68）
        """
        recent_turns = conversation_history[-15:] if len(conversation_history) > 15 else conversation_history
        conversation_text = "\n".join([
            f"{msg['speaker']}: {msg['content']}"
            for msg in recent_turns
        ])

        prompt = f"""你是一个心理特征分析专家。请基于对话历史构建三段论推理树，预测用户的 {feature_name} 特征。

对话历史:
{conversation_text}

当前预测:
- 特征: {feature_name}
- 当前值: {current_value}
- 置信度: {current_confidence:.2f}

请按以下格式输出三段论推理：

<major_premise>
[描述特征与行为的关联规律]
置信度: [0.0-1.0]
证据: [心理学理论或统计规律]
</major_premise>

<minor_premise>
[描述用户的具体行为表现]
置信度: [0.0-1.0]
证据: [具体对话实例]
</minor_premise>

<conclusion>
[推理出的特征值]
置信度: [0.0-1.0]
证据: [综合推理依据]
</conclusion>
"""

        response = await self.llm_router.chat(
            messages=[{"role": "user", "content": prompt}],
            role="logic_tree_builder",
            temperature=0.3
        )

        return self._parse_syllogism_response(response)

    def _parse_syllogism_response(self, response: str) -> LogicTreeNode:
        """解析 LLM 返回的三段论结构"""
        import re

        # 提取大前提
        major_match = re.search(
            r'<major_premise>(.*?)</major_premise>',
            response,
            re.DOTALL
        )
        major_content = ""
        major_confidence = 0.7
        major_evidence = []

        if major_match:
            major_text = major_match.group(1).strip()
            # 提取内容
            content_match = re.search(r'^(.*?)(?=置信度:|$)', major_text, re.DOTALL)
            if content_match:
                major_content = content_match.group(1).strip()
            # 提取置信度
            conf_match = re.search(r'置信度:\s*([\d.]+)', major_text)
            if conf_match:
                major_confidence = float(conf_match.group(1))
            # 提取证据
            evidence_match = re.search(r'证据:\s*(.+?)(?=\n|$)', major_text, re.DOTALL)
            if evidence_match:
                major_evidence = [evidence_match.group(1).strip()]

        # 提取小前提
        minor_match = re.search(
            r'<minor_premise>(.*?)</minor_premise>',
            response,
            re.DOTALL
        )
        minor_content = ""
        minor_confidence = 0.7
        minor_evidence = []

        if minor_match:
            minor_text = minor_match.group(1).strip()
            content_match = re.search(r'^(.*?)(?=置信度:|$)', minor_text, re.DOTALL)
            if content_match:
                minor_content = content_match.group(1).strip()
            conf_match = re.search(r'置信度:\s*([\d.]+)', minor_text)
            if conf_match:
                minor_confidence = float(conf_match.group(1))
            evidence_match = re.search(r'证据:\s*(.+?)(?=\n|$)', minor_text, re.DOTALL)
            if evidence_match:
                minor_evidence = [evidence_match.group(1).strip()]

        # 提取结论
        conclusion_match = re.search(
            r'<conclusion>(.*?)</conclusion>',
            response,
            re.DOTALL
        )
        conclusion_content = ""
        conclusion_confidence = 0.7
        conclusion_evidence = []

        if conclusion_match:
            conclusion_text = conclusion_match.group(1).strip()
            content_match = re.search(r'^(.*?)(?=置信度:|$)', conclusion_text, re.DOTALL)
            if content_match:
                conclusion_content = content_match.group(1).strip()
            conf_match = re.search(r'置信度:\s*([\d.]+)', conclusion_text)
            if conf_match:
                conclusion_confidence = float(conf_match.group(1))
            evidence_match = re.search(r'证据:\s*(.+?)(?=\n|$)', conclusion_text, re.DOTALL)
            if evidence_match:
                conclusion_evidence = [evidence_match.group(1).strip()]

        # 构建树结构
        major_node = LogicTreeNode(
            node_type=NodeType.MAJOR_PREMISE,
            content=major_content,
            confidence=major_confidence,
            evidence=major_evidence
        )

        minor_node = LogicTreeNode(
            node_type=NodeType.MINOR_PREMISE,
            content=minor_content,
            confidence=minor_confidence,
            evidence=minor_evidence
        )

        conclusion_node = LogicTreeNode(
            node_type=NodeType.CONCLUSION,
            content=conclusion_content,
            confidence=conclusion_confidence,
            evidence=conclusion_evidence,
            children=[major_node, minor_node]
        )

        return conclusion_node

    def detect_logic_conflicts(
        self,
        tree1: LogicTreeNode,
        tree2: LogicTreeNode
    ) -> Optional[str]:
        """
        检测两个逻辑树之间的冲突

        返回冲突描述，如果无冲突则返回 None
        """
        # 简单冲突检测：比较结论的置信度和内容
        if tree1.node_type == NodeType.CONCLUSION and tree2.node_type == NodeType.CONCLUSION:
            # 如果两个结论的置信度都很高但内容矛盾
            if tree1.confidence > 0.7 and tree2.confidence > 0.7:
                # 这里可以添加更复杂的语义相似度检测
                if tree1.content != tree2.content:
                    return f"逻辑冲突: 两个高置信度结论不一致\n结论1: {tree1.content} (conf={tree1.confidence:.2f})\n结论2: {tree2.content} (conf={tree2.confidence:.2f})"

        return None

    async def resolve_conflict(
        self,
        tree1: LogicTreeNode,
        tree2: LogicTreeNode,
        conflict_description: str
    ) -> LogicTreeNode:
        """
        解决逻辑冲突 - 通过多智能体讨论达成共识

        返回解决后的逻辑树
        """
        prompt = f"""你是一个逻辑冲突调解专家。以下两个推理树存在冲突，请分析并给出最合理的结论。

冲突描述:
{conflict_description}

推理树1:
{tree1.to_syllogism()}

推理树2:
{tree2.to_syllogism()}

请分析：
1. 哪个推理更可靠？为什么？
2. 是否可以综合两个推理？
3. 给出最终的三段论推理（使用相同的 XML 格式）
"""

        response = await self.llm_router.chat(
            messages=[{"role": "user", "content": prompt}],
            role="logic_conflict_resolver",
            temperature=0.2
        )

        return self._parse_syllogism_response(response)
