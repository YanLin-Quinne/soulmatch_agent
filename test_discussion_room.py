"""
Test Agent Discussion Room - Multi-agent debate mechanism
"""

import asyncio
from src.agents.agent_discussion_room import AgentDiscussionRoom
from src.agents.llm_router import LLMRouter, AgentRole


async def test_relationship_advancement_discussion():
    """Test: Should the relationship advance from acquaintance to crush?"""

    router = LLMRouter()
    room = AgentDiscussionRoom(router)

    # Context from a hypothetical conversation
    context = {
        "turn_count": 15,
        "current_rel_status": "acquaintance",
        "trust_score": 0.68,
        "sentiment_trend": "improving",
        "recent_emotions": ["joy", "interest", "excitement"],
        "shared_interests": ["hiking", "photography", "travel"],
        "conversation_depth": "moderate",
    }

    # Define 3 expert agents
    agents = [
        {
            "name": "EmotionExpert",
            "role": AgentRole.EMOTION,
            "expertise": "Emotional dynamics and sentiment analysis",
            "system_prompt": "You are an expert in emotional intelligence and relationship dynamics."
        },
        {
            "name": "ValuesExpert",
            "role": AgentRole.FEATURE,
            "expertise": "Value alignment and compatibility assessment",
            "system_prompt": "You are an expert in personality psychology and value systems."
        },
        {
            "name": "BehaviorExpert",
            "role": AgentRole.GENERAL,
            "expertise": "Behavioral patterns and interaction quality",
            "system_prompt": "You are an expert in behavioral psychology and social interactions."
        }
    ]

    # Weighted voting (emotion matters most for relationship advancement)
    voting_weights = {
        "EmotionExpert": 0.4,
        "ValuesExpert": 0.3,
        "BehaviorExpert": 0.3
    }

    print("=" * 70)
    print("Agent Discussion Room Test: Relationship Advancement")
    print("=" * 70)
    print()

    # Conduct discussion
    consensus = await room.discuss(
        topic="Based on the context, should this relationship advance from 'acquaintance' to 'crush'?",
        context=context,
        agents=agents,
        voting_weights=voting_weights
    )

    print(f"Decision: {consensus.decision}")
    print(f"Confidence: {consensus.confidence:.2f}")
    print()
    print("Reasoning:")
    print(consensus.reasoning)
    print()
    print("All Proposals:")
    for p in consensus.proposals:
        print(f"  - {p.agent_name}: {p.proposal} (confidence: {p.confidence:.2f})")
    print()
    print("Critiques:")
    for c in consensus.critiques[:3]:  # Show top 3
        print(f"  - {c.critic_name} → {c.target_proposal}: {c.critique[:100]}...")
    print()
    print("=" * 70)

    return consensus


async def test_mbti_inference_discussion():
    """Test: What is the user's MBTI type?"""

    router = LLMRouter()
    room = AgentDiscussionRoom(router)

    context = {
        "big_five_openness": 0.82,
        "big_five_conscientiousness": 0.45,
        "big_five_extraversion": 0.38,
        "big_five_agreeableness": 0.71,
        "big_five_neuroticism": 0.55,
        "conversation_style": "thoughtful, introspective, creative",
        "interests": ["philosophy", "art", "writing"],
    }

    agents = [
        {
            "name": "PersonalityExpert",
            "role": AgentRole.FEATURE,
            "expertise": "MBTI and personality type assessment",
            "system_prompt": "You are an expert in MBTI personality types and psychometric assessment."
        },
        {
            "name": "BehaviorExpert",
            "role": AgentRole.GENERAL,
            "expertise": "Behavioral patterns and cognitive functions",
            "system_prompt": "You are an expert in cognitive functions and behavioral psychology."
        }
    ]

    print("=" * 70)
    print("Agent Discussion Room Test: MBTI Inference")
    print("=" * 70)
    print()

    consensus = await room.discuss(
        topic="Based on the Big Five scores and behavioral patterns, what is the most likely MBTI type?",
        context=context,
        agents=agents
    )

    print(f"Decision: {consensus.decision}")
    print(f"Confidence: {consensus.confidence:.2f}")
    print()
    print("Reasoning:")
    print(consensus.reasoning)
    print()
    print("=" * 70)

    return consensus


if __name__ == "__main__":
    print("Testing Agent Discussion Room...")
    print()

    # Test 1: Relationship advancement
    result1 = asyncio.run(test_relationship_advancement_discussion())

    print("\n\n")

    # Test 2: MBTI inference
    result2 = asyncio.run(test_mbti_inference_discussion())

    print("\n✓ All discussion room tests completed!")
