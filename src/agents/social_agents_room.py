"""
Social Agents Room - Demographic Diverse Agents for Relationship Assessment

Inspired by Social Agents (ICLR 2026): Synthesizes diverse social personas
with different demographic backgrounds to evaluate relationship compatibility.

Core Innovation:
- NOT just "expert roles" with different system prompts
- REAL demographic diversity: age, gender, occupation, life experience
- Wisdom of crowds through heterogeneous perspectives
- Each agent judges based on their own values and experiences

Example Agents:
- 25-year-old female fitness trainer (optimistic, values health)
- 45-year-old male divorced lawyer (cautious, values stability)
- 30-year-old non-binary artist (creative, values authenticity)
- 35-year-old female single mother (practical, values family)
- 28-year-old male tech entrepreneur (ambitious, values growth)
"""

import asyncio
import json
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from loguru import logger

from src.agents.llm_router import LLMRouter, AgentRole


@dataclass
class SocialPersona:
    """A social agent with demographic background and life experience"""
    name: str
    age: int
    gender: str  # "male", "female", "non-binary"
    occupation: str
    relationship_status: str  # "single", "married", "divorced", "widowed"
    life_experience: str  # Brief description of key life experiences
    values: List[str]  # Core values (e.g., "honesty", "adventure", "stability")
    personality_traits: List[str]  # e.g., "optimistic", "cautious", "empathetic"

    def to_system_prompt(self) -> str:
        """Generate system prompt from demographic profile"""
        return f"""You are {self.name}, a {self.age}-year-old {self.gender} {self.occupation}.

Your Background:
- Relationship Status: {self.relationship_status}
- Life Experience: {self.life_experience}
- Core Values: {', '.join(self.values)}
- Personality: {', '.join(self.personality_traits)}

You are evaluating a romantic relationship between two people based on YOUR OWN life experience and values.
Be authentic to your background - your age, gender, occupation, and past experiences shape how you see relationships.
Judge whether this relationship has potential based on what YOU believe makes relationships work."""


@dataclass
class SocialAgentVote:
    """A single social agent's vote on relationship compatibility"""
    agent_name: str
    agent_demographics: Dict[str, Any]  # age, gender, relationship_status for similarity calculation
    vote: str  # "compatible", "incompatible", "uncertain"
    rel_status: str  # "stranger", "acquaintance", "crush", "dating", "committed"
    rel_type: str  # "love", "friendship", "family", "other"
    confidence: float  # 0.0-1.0
    reasoning: str  # Why they voted this way based on their experience
    key_factors: List[str]  # What factors influenced their decision


@dataclass
class SocialConsensus:
    """Consensus reached by social agents through demographic-weighted voting"""
    decision: str  # "compatible", "incompatible", "uncertain"
    rel_status: str  # Consensus relationship status
    rel_type: str  # Consensus relationship type
    rel_status_probs: Dict[str, float]  # Probability distribution over statuses
    rel_type_probs: Dict[str, float]  # Probability distribution over types
    confidence: float  # Weighted average confidence
    votes: List[SocialAgentVote]
    vote_distribution: Dict[str, int]  # {"compatible": 3, "incompatible": 1, "uncertain": 1}
    reasoning: str  # Synthesized reasoning from all votes
    demographic_insights: Dict[str, Any]  # Insights by demographic group


class SocialAgentsRoom:
    """
    Social Agents Room for relationship assessment with demographic diversity.

    Implements wisdom of crowds through heterogeneous social personas.
    """

    def __init__(self, router: LLMRouter):
        self.router = router
        self.social_agents = self._create_default_agents()

    def _create_default_agents(self) -> List[SocialPersona]:
        """Create 5 demographically diverse social agents"""
        return [
            SocialPersona(
                name="Emma",
                age=25,
                gender="female",
                occupation="fitness trainer",
                relationship_status="single",
                life_experience="Recently ended a 3-year relationship. Values personal growth and healthy communication. Believes in taking time to know someone before committing.",
                values=["health", "honesty", "personal_growth", "independence"],
                personality_traits=["optimistic", "energetic", "direct", "empathetic"]
            ),
            SocialPersona(
                name="Marcus",
                age=45,
                gender="male",
                occupation="divorce lawyer",
                relationship_status="divorced",
                life_experience="Divorced after 15 years of marriage. Sees relationships through a realistic lens. Values stability and clear communication. Cautious about red flags.",
                values=["stability", "honesty", "respect", "financial_security"],
                personality_traits=["cautious", "analytical", "pragmatic", "protective"]
            ),
            SocialPersona(
                name="River",
                age=30,
                gender="non-binary",
                occupation="visual artist",
                relationship_status="single",
                life_experience="Has had several meaningful relationships. Values authenticity and emotional depth. Believes love should be creative and evolving.",
                values=["authenticity", "creativity", "emotional_depth", "freedom"],
                personality_traits=["intuitive", "open-minded", "sensitive", "unconventional"]
            ),
            SocialPersona(
                name="Sarah",
                age=35,
                gender="female",
                occupation="single mother and teacher",
                relationship_status="single",
                life_experience="Raising two kids alone after partner left. Values reliability and family commitment. Looks for partners who understand parental responsibilities.",
                values=["family", "reliability", "patience", "commitment"],
                personality_traits=["practical", "nurturing", "resilient", "careful"]
            ),
            SocialPersona(
                name="Alex",
                age=28,
                gender="male",
                occupation="tech startup founder",
                relationship_status="single",
                life_experience="Building a company, values ambition and growth mindset. Believes relationships should support mutual goals. Seeks intellectual connection.",
                values=["ambition", "growth", "intelligence", "innovation"],
                personality_traits=["driven", "curious", "confident", "forward-thinking"]
            ),
        ]

    async def assess_relationship(
        self,
        conversation_summary: str,
        relationship_context: Dict[str, Any],
        user_demographics: Optional[Dict[str, Any]] = None,
        custom_agents: Optional[List[SocialPersona]] = None
    ) -> SocialConsensus:
        """
        Assess relationship compatibility through social agents voting.

        Args:
            conversation_summary: Summary of the conversation between two people
            relationship_context: Context including:
                - rel_status: Current relationship status
                - sentiment: Overall sentiment
                - trust_score: Trust level
                - turn_count: Number of conversation turns
            user_demographics: User's demographic info for similarity weighting:
                - age: int
                - gender: str
                - relationship_status: str
            custom_agents: Optional custom social agents (default: use built-in 5)

        Returns:
            SocialConsensus with votes and reasoning
        """
        agents = custom_agents or self.social_agents

        logger.info(f"[SocialAgentsRoom] Assessing relationship with {len(agents)} social agents")

        # Collect votes from all agents in parallel
        votes = await self._collect_votes(conversation_summary, relationship_context, agents)

        # Reach consensus through demographic-weighted voting
        consensus = self._reach_consensus(votes, user_demographics)

        logger.info(f"[SocialAgentsRoom] Consensus: {consensus.decision} (rel_status={consensus.rel_status}, confidence: {consensus.confidence:.2f})")

        return consensus

    async def _collect_votes(
        self,
        conversation_summary: str,
        context: Dict[str, Any],
        agents: List[SocialPersona]
    ) -> List[SocialAgentVote]:
        """Collect votes from all social agents in parallel"""

        tasks = [
            self._get_agent_vote(agent, conversation_summary, context)
            for agent in agents
        ]

        votes = await asyncio.gather(*tasks, return_exceptions=True)

        # Filter out exceptions
        valid_votes = [v for v in votes if isinstance(v, SocialAgentVote)]

        if len(valid_votes) < len(agents):
            logger.warning(f"[SocialAgentsRoom] Only {len(valid_votes)}/{len(agents)} votes succeeded")

        return valid_votes

    async def _get_agent_vote(
        self,
        agent: SocialPersona,
        conversation_summary: str,
        context: Dict[str, Any]
    ) -> SocialAgentVote:
        """Get a single agent's vote based on their demographic background"""

        prompt = f"""You are evaluating a romantic relationship between two people.

Conversation Summary:
{conversation_summary}

Current Status:
- Relationship Stage: {context.get('rel_status', 'unknown')}
- Overall Sentiment: {context.get('sentiment', 'unknown')}
- Trust Level: {context.get('trust_score', 0.5):.2f}
- Conversation Turns: {context.get('turn_count', 0)}

Based on YOUR life experience as a {agent.age}-year-old {agent.gender} {agent.occupation} who is {agent.relationship_status},
and YOUR values ({', '.join(agent.values)}), evaluate this relationship:

1. Are they compatible? (compatible/incompatible/uncertain)
2. What relationship stage are they at? (stranger/acquaintance/crush/dating/committed)
3. What type of relationship is this? (love/friendship/family/other)

Consider:
- What would YOU look for in a relationship at your age and life stage?
- How do YOUR past experiences shape your view of this relationship?
- What red flags or green flags do YOU see based on YOUR values?

Respond with JSON:
{{
    "vote": "compatible" | "incompatible" | "uncertain",
    "rel_status": "stranger" | "acquaintance" | "crush" | "dating" | "committed",
    "rel_type": "love" | "friendship" | "family" | "other",
    "confidence": 0.0-1.0,
    "reasoning": "Explain based on YOUR background and values",
    "key_factors": ["factor1", "factor2", "factor3"]
}}
"""

        try:
            response = self.router.chat(
                role=AgentRole.GENERAL,
                system=agent.to_system_prompt(),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,  # Allow personality to show through
                max_tokens=400,
                json_mode=True,
            )

            # Parse JSON
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]

            result = json.loads(response.strip())

            return SocialAgentVote(
                agent_name=agent.name,
                agent_demographics={
                    "age": agent.age,
                    "gender": agent.gender,
                    "relationship_status": agent.relationship_status
                },
                vote=result["vote"],
                rel_status=result.get("rel_status", context.get("rel_status", "stranger")),
                rel_type=result.get("rel_type", "other"),
                confidence=result["confidence"],
                reasoning=result["reasoning"],
                key_factors=result.get("key_factors", [])
            )

        except Exception as e:
            logger.error(f"[SocialAgentsRoom] Vote from {agent.name} failed: {e}")
            # Return uncertain vote as fallback
            return SocialAgentVote(
                agent_name=agent.name,
                agent_demographics={
                    "age": agent.age,
                    "gender": agent.gender,
                    "relationship_status": agent.relationship_status
                },
                vote="uncertain",
                rel_status=context.get("rel_status", "stranger"),
                rel_type="other",
                confidence=0.5,
                reasoning=f"Vote failed: {str(e)}",
                key_factors=[]
            )

    def _reach_consensus(
        self,
        votes: List[SocialAgentVote],
        user_demographics: Optional[Dict[str, Any]] = None
    ) -> SocialConsensus:
        """
        Reach consensus through demographic-weighted voting.
        
        Based on ICLR 2026 Social Agents paper: agents with similar demographics
        to the user have higher voting weight.
        """

        if not votes:
            return SocialConsensus(
                decision="uncertain",
                rel_status="stranger",
                rel_type="other",
                rel_status_probs={"stranger": 1.0},
                rel_type_probs={"other": 1.0},
                confidence=0.0,
                votes=[],
                vote_distribution={},
                reasoning="No votes collected",
                demographic_insights={}
            )

        # Calculate demographic similarity weights for each agent
        weights = []
        for vote in votes:
            if user_demographics:
                # Create temporary agent for similarity calculation
                temp_agent = type('obj', (object,), vote.agent_demographics)()
                similarity = self._calculate_demographic_similarity(temp_agent, user_demographics)
                weights.append(similarity)
            else:
                # Equal weights if no user demographics available
                weights.append(1.0)

        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / len(votes)] * len(votes)

        # Weighted voting for compatibility
        vote_scores = {"compatible": 0.0, "incompatible": 0.0, "uncertain": 0.0}
        vote_counts = {"compatible": 0, "incompatible": 0, "uncertain": 0}
        
        for vote, weight in zip(votes, weights):
            vote_counts[vote.vote] += 1
            vote_scores[vote.vote] += weight * vote.confidence

        # Majority decision (with demographic weighting)
        decision = max(vote_scores, key=vote_scores.get)

        # Weighted voting for rel_status
        status_scores = {}
        for vote, weight in zip(votes, weights):
            if vote.rel_status not in status_scores:
                status_scores[vote.rel_status] = 0.0
            status_scores[vote.rel_status] += weight * vote.confidence

        # Normalize to probabilities
        total_status_score = sum(status_scores.values())
        rel_status_probs = {k: v / total_status_score for k, v in status_scores.items()} if total_status_score > 0 else {"stranger": 1.0}
        rel_status = max(rel_status_probs, key=rel_status_probs.get)

        # Weighted voting for rel_type
        type_scores = {}
        for vote, weight in zip(votes, weights):
            if vote.rel_type not in type_scores:
                type_scores[vote.rel_type] = 0.0
            type_scores[vote.rel_type] += weight * vote.confidence

        # Normalize to probabilities
        total_type_score = sum(type_scores.values())
        rel_type_probs = {k: v / total_type_score for k, v in type_scores.items()} if total_type_score > 0 else {"other": 1.0}
        rel_type = max(rel_type_probs, key=rel_type_probs.get)

        # Average confidence of winning decision (weighted)
        winning_votes = [(v, w) for v, w in zip(votes, weights) if v.vote == decision]
        avg_confidence = sum(v.confidence * w for v, w in winning_votes) / sum(w for _, w in winning_votes) if winning_votes else 0.5

        # Synthesize reasoning from all votes (prioritize high-weight agents)
        reasoning_parts = []
        for vote, weight in sorted(zip(votes, weights), key=lambda x: x[1], reverse=True):
            reasoning_parts.append(f"{vote.agent_name} (weight={weight:.2f}, {vote.vote}, {vote.confidence:.2f}): {vote.reasoning[:100]}...")

        synthesized_reasoning = "\n".join(reasoning_parts)

        # Demographic insights
        demographic_insights = {
            "total_agents": len(votes),
            "vote_distribution": vote_counts,
            "weighted_scores": vote_scores,
            "demographic_weights": {vote.agent_name: weight for vote, weight in zip(votes, weights)},
            "consensus_strength": vote_scores[decision] / sum(vote_scores.values()) if sum(vote_scores.values()) > 0 else 0.0,
            "status_distribution": status_scores,
            "type_distribution": type_scores
        }

        return SocialConsensus(
            decision=decision,
            rel_status=rel_status,
            rel_type=rel_type,
            rel_status_probs=rel_status_probs,
            rel_type_probs=rel_type_probs,
            confidence=avg_confidence,
            votes=votes,
            vote_distribution=vote_counts,
            reasoning=synthesized_reasoning,
            demographic_insights=demographic_insights
        )

    def _calculate_demographic_similarity(
        self,
        agent: SocialPersona,
        user_demographics: Dict[str, Any]
    ) -> float:
        """
        Calculate demographic similarity between agent and user.
        
        Based on ICLR 2026 Social Agents paper: agents with similar demographics
        to the user should have higher voting weight.
        
        Args:
            agent: Social agent persona
            user_demographics: User's demographic info (age, gender, relationship_status)
        
        Returns:
            Similarity score 0.0-1.0
        """
        similarity = 0.0
        weight_sum = 0.0
        
        # Age similarity (weight: 0.3)
        if 'age' in user_demographics and user_demographics['age']:
            age_diff = abs(agent.age - user_demographics['age'])
            age_similarity = max(0, 1 - age_diff / 50)  # Normalize by 50 years
            similarity += age_similarity * 0.3
            weight_sum += 0.3
        
        # Gender similarity (weight: 0.3)
        if 'gender' in user_demographics and user_demographics['gender']:
            gender_match = 1.0 if agent.gender == user_demographics['gender'] else 0.0
            similarity += gender_match * 0.3
            weight_sum += 0.3
        
        # Relationship status similarity (weight: 0.4)
        if 'relationship_status' in user_demographics and user_demographics['relationship_status']:
            status_match = 1.0 if agent.relationship_status == user_demographics['relationship_status'] else 0.0
            similarity += status_match * 0.4
            weight_sum += 0.4
        
        # Normalize by actual weights used
        if weight_sum > 0:
            similarity = similarity / weight_sum
        else:
            similarity = 0.5  # Default if no demographics available
        
        return similarity
