"""
Agent Discussion Room - Multi-agent debate and consensus mechanism

Implements the multi-agent collaboration pattern from the research paper:
- Each agent proposes their view
- Agents critique each other's proposals
- Weighted voting to reach consensus
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from loguru import logger

from src.agents.llm_router import LLMRouter, AgentRole


@dataclass
class AgentProposal:
    """A single agent's proposal"""
    agent_name: str
    proposal: str
    confidence: float
    reasoning: str


@dataclass
class AgentCritique:
    """A critique of other agents' proposals"""
    critic_name: str
    target_proposal: str
    critique: str
    agreement_score: float  # 0.0-1.0


@dataclass
class ConsensusResult:
    """Final consensus reached by agents"""
    decision: Any
    confidence: float
    reasoning: str
    proposals: List[AgentProposal]
    critiques: List[AgentCritique]
    voting_weights: Dict[str, float]


class AgentDiscussionRoom:
    """
    Multi-agent discussion room for collaborative decision making.

    Workflow:
    1. Propose: Each agent independently proposes their view
    2. Critique: Agents critique each other's proposals
    3. Vote: Weighted voting based on confidence and agreement
    4. Consensus: Return final decision with reasoning
    """

    def __init__(self, router: LLMRouter):
        self.router = router

    async def discuss(
        self,
        topic: str,
        context: Dict[str, Any],
        agents: List[Dict[str, Any]],
        voting_weights: Optional[Dict[str, float]] = None
    ) -> ConsensusResult:
        """
        Conduct multi-agent discussion and reach consensus.

        Args:
            topic: The question/topic to discuss
            context: Shared context for all agents
            agents: List of agent configs, each with:
                - name: Agent name
                - role: AgentRole
                - expertise: What this agent specializes in
                - system_prompt: Agent's system prompt
            voting_weights: Optional custom weights (default: equal)

        Returns:
            ConsensusResult with final decision and reasoning
        """
        logger.info(f"[AgentDiscussionRoom] Starting discussion on: {topic}")

        # Default equal weights
        if voting_weights is None:
            voting_weights = {agent["name"]: 1.0 / len(agents) for agent in agents}

        # Phase 1: Propose (parallel)
        proposals = await self._collect_proposals(topic, context, agents)

        # Phase 2: Critique (parallel)
        critiques = await self._collect_critiques(proposals, agents)

        # Phase 3: Vote and reach consensus
        consensus = self._reach_consensus(proposals, critiques, voting_weights)

        logger.info(f"[AgentDiscussionRoom] Consensus reached: {consensus.decision} (confidence: {consensus.confidence:.2f})")

        return consensus

    async def _collect_proposals(
        self,
        topic: str,
        context: Dict[str, Any],
        agents: List[Dict[str, Any]]
    ) -> List[AgentProposal]:
        """Phase 1: Each agent proposes their view"""

        async def get_proposal(agent: Dict[str, Any]) -> AgentProposal:
            system_prompt = f"""
{agent['system_prompt']}

Your expertise: {agent['expertise']}

Task: Analyze the following question and provide your expert opinion.
Format your response as JSON:
{{
    "proposal": "your answer to the question",
    "confidence": 0.0-1.0,
    "reasoning": "why you believe this"
}}
"""

            context_str = "\n".join([f"{k}: {v}" for k, v in context.items()])

            response = await asyncio.to_thread(
                self.router.chat,
                role=agent["role"],
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Context:\n{context_str}\n\nQuestion: {topic}"
                    }
                ],
                max_tokens=500
            )

            # Parse JSON response (handle markdown fences)
            import json
            import re
            try:
                # Remove markdown code fences if present
                clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
                data = json.loads(clean_response)
                return AgentProposal(
                    agent_name=agent["name"],
                    proposal=data["proposal"],
                    confidence=float(data["confidence"]),
                    reasoning=data["reasoning"]
                )
            except Exception as e:
                logger.warning(f"[AgentDiscussionRoom] Failed to parse proposal from {agent['name']}: {e}")
                # Fallback if JSON parsing fails
                return AgentProposal(
                    agent_name=agent["name"],
                    proposal=response[:200],
                    confidence=0.5,
                    reasoning="(parsing failed)"
                )

        proposals = await asyncio.gather(*[get_proposal(agent) for agent in agents])

        logger.info(f"[AgentDiscussionRoom] Collected {len(proposals)} proposals")
        return proposals

    async def _collect_critiques(
        self,
        proposals: List[AgentProposal],
        agents: List[Dict[str, Any]]
    ) -> List[AgentCritique]:
        """Phase 2: Agents critique each other's proposals"""

        async def get_critique(agent: Dict[str, Any]) -> List[AgentCritique]:
            # Don't critique your own proposal
            other_proposals = [p for p in proposals if p.agent_name != agent["name"]]

            if not other_proposals:
                return []

            proposals_str = "\n\n".join([
                f"Proposal by {p.agent_name}:\n{p.proposal}\nReasoning: {p.reasoning}\nConfidence: {p.confidence:.2f}"
                for p in other_proposals
            ])

            system_prompt = f"""
{agent['system_prompt']}

Your expertise: {agent['expertise']}

Task: Critique the following proposals from other agents.
For each proposal, provide:
1. Your critique (what's good/bad about it)
2. Agreement score (0.0-1.0, how much you agree)

Format as JSON array:
[
    {{
        "target_agent": "agent name",
        "critique": "your critique",
        "agreement_score": 0.0-1.0
    }},
    ...
]
"""

            response = await asyncio.to_thread(
                self.router.chat,
                role=agent["role"],
                system=system_prompt,
                messages=[
                    {
                        "role": "user",
                        "content": f"Proposals to critique:\n\n{proposals_str}"
                    }
                ],
                max_tokens=800
            )

            # Parse JSON response (handle markdown fences)
            import json
            import re
            try:
                # Remove markdown code fences if present
                clean_response = re.sub(r'^```json\s*|\s*```$', '', response.strip(), flags=re.MULTILINE)
                data = json.loads(clean_response)
                return [
                    AgentCritique(
                        critic_name=agent["name"],
                        target_proposal=item["target_agent"],
                        critique=item["critique"],
                        agreement_score=float(item["agreement_score"])
                    )
                    for item in data
                ]
            except Exception as e:
                logger.warning(f"[AgentDiscussionRoom] Failed to parse critiques from {agent['name']}: {e}")
                return []

        all_critiques = await asyncio.gather(*[get_critique(agent) for agent in agents])
        critiques = [c for sublist in all_critiques for c in sublist]  # Flatten

        logger.info(f"[AgentDiscussionRoom] Collected {len(critiques)} critiques")
        return critiques

    def _reach_consensus(
        self,
        proposals: List[AgentProposal],
        critiques: List[AgentCritique],
        voting_weights: Dict[str, float]
    ) -> ConsensusResult:
        """Phase 3: Weighted voting to reach consensus"""

        # Calculate final scores for each proposal
        scores = {}
        for proposal in proposals:
            # Base score: agent's own confidence * voting weight
            base_score = proposal.confidence * voting_weights.get(proposal.agent_name, 0.0)

            # Bonus: agreement from other agents
            agreement_bonus = 0.0
            for critique in critiques:
                if critique.target_proposal == proposal.agent_name:
                    critic_weight = voting_weights.get(critique.critic_name, 0.0)
                    agreement_bonus += critique.agreement_score * critic_weight

            final_score = base_score + agreement_bonus
            scores[proposal.agent_name] = final_score

        # Winner: highest score
        winner_name = max(scores, key=scores.get)
        winner_proposal = next(p for p in proposals if p.agent_name == winner_name)

        # Confidence: normalized score
        total_score = sum(scores.values())
        confidence = scores[winner_name] / total_score if total_score > 0 else 0.5

        # Reasoning: combine winner's reasoning + supporting critiques
        supporting_critiques = [
            c for c in critiques
            if c.target_proposal == winner_name and c.agreement_score > 0.6
        ]

        reasoning = f"{winner_proposal.reasoning}\n\nSupporting views:\n"
        for c in supporting_critiques[:2]:  # Top 2 supporting critiques
            reasoning += f"- {c.critic_name}: {c.critique}\n"

        return ConsensusResult(
            decision=winner_proposal.proposal,
            confidence=confidence,
            reasoning=reasoning,
            proposals=proposals,
            critiques=critiques,
            voting_weights=voting_weights
        )
