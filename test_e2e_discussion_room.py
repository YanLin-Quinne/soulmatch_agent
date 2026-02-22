"""
End-to-end test for RelationshipPredictionAgent with Agent Discussion Room
"""

import asyncio
from src.agents.relationship_prediction_agent import RelationshipPredictionAgent
from src.agents.agent_context import AgentContext
from src.agents.llm_router import LLMRouter


async def test_relationship_prediction_with_discussion_room():
    """Test RelationshipPredictionAgent using Agent Discussion Room"""

    print("=" * 70)
    print("End-to-End Test: RelationshipPredictionAgent with Discussion Room")
    print("=" * 70)
    print()

    # Initialize
    router = LLMRouter()
    agent = RelationshipPredictionAgent(llm_router=router, use_discussion_room=True)

    # Create mock context
    ctx = AgentContext(
        user_id="test_user",
        bot_id="mina",
        turn_count=10,  # Trigger milestone evaluation
        rel_status="acquaintance",
        rel_type="other",
    )

    # Mock conversation history (correct format with speaker/message)
    ctx.conversation_history = [
        {"speaker": "user", "message": "Hi, I love hiking!"},
        {"speaker": "bot", "message": "That's great! I love hiking too!"},
        {"speaker": "user", "message": "What's your favorite trail?"},
        {"speaker": "bot", "message": "I really enjoy mountain trails with scenic views."},
        {"speaker": "user", "message": "We should go hiking together sometime!"},
        {"speaker": "bot", "message": "I'd love that! When are you free?"},
        {"speaker": "user", "message": "How about next weekend?"},
        {"speaker": "bot", "message": "Sounds perfect! I'm looking forward to it."},
        {"speaker": "user", "message": "Me too! You seem really nice."},
        {"speaker": "bot", "message": "Thank you! I think we have a lot in common."},
    ]

    # Mock emotion history
    ctx.emotion_history = ["joy", "interest", "excitement", "joy", "trust"]

    # Mock predicted features
    ctx.predicted_features = {
        "big_five": {
            "openness": 0.75,
            "conscientiousness": 0.65,
            "extraversion": 0.70,
            "agreeableness": 0.80,
            "neuroticism": 0.40,
        },
        "interests": ["hiking", "photography", "travel"],
    }

    ctx.extended_features = {
        "trust_score": 0.68,
        "trust_velocity": 0.05,
        "mbti_type": "ENFP",
        "attachment_style": "secure",
    }

    print("Context:")
    print(f"  Turn: {ctx.turn_count}")
    print(f"  Current status: {ctx.rel_status}")
    print(f"  Trust score: {ctx.extended_features['trust_score']}")
    print(f"  Recent emotions: {ctx.emotion_history[-5:]}")
    print()

    # Execute relationship prediction
    print("Executing RelationshipPredictionAgent with Discussion Room...")
    print()

    result = await agent.execute(ctx)

    print("=" * 70)
    print("Results:")
    print("=" * 70)
    print(f"Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.2f})")
    print(f"Relationship Type: {result['rel_type']}")
    print(f"  Probabilities: {result['rel_type_probs']}")
    print(f"Relationship Status: {result['rel_status']}")
    print(f"  Probabilities: {result['rel_status_probs']}")
    print(f"Can Advance: {result['can_advance']}")
    print(f"  Prediction Set: {result['advance_prediction_set']}")
    print(f"  Coverage Guarantee: {result['advance_coverage_guarantee']}")
    print(f"Next Status Prediction: {result['next_status_prediction']}")
    print(f"  Probabilities: {result['next_status_probs']}")
    print()
    print("Reasoning:")
    print(result['reasoning_trace'])
    print()
    print("=" * 70)

    # Verify context was updated
    assert ctx.rel_status == result['rel_status']
    assert ctx.rel_type == result['rel_type']
    assert ctx.sentiment_label == result['sentiment']
    assert ctx.can_advance == result['can_advance']
    assert len(ctx.relationship_snapshots) > 0

    print("✓ All assertions passed!")
    print()

    return result


async def test_comparison_single_vs_multi():
    """Compare single LLM vs multi-agent discussion room"""

    print("=" * 70)
    print("Comparison Test: Single LLM vs Multi-Agent Discussion Room")
    print("=" * 70)
    print()

    router = LLMRouter()

    # Create identical contexts
    ctx1 = AgentContext(
        user_id="test_user",
        bot_id="mina",
        turn_count=10,
        rel_status="acquaintance",
    )
    ctx2 = AgentContext(
        user_id="test_user",
        bot_id="mina",
        turn_count=10,
        rel_status="acquaintance",
    )

    # Mock data (same for both)
    for ctx in [ctx1, ctx2]:
        ctx.conversation_history = [
            {"speaker": "user", "message": "Hi!"},
            {"speaker": "bot", "message": "Hello!"},
        ] * 5
        ctx.emotion_history = ["joy", "interest", "excitement", "joy", "trust"]
        ctx.predicted_features = {"big_five": {}, "interests": ["hiking"]}
        ctx.extended_features = {"trust_score": 0.68}

    # Test 1: Single LLM
    print("Test 1: Single LLM (use_discussion_room=False)")
    agent1 = RelationshipPredictionAgent(llm_router=router, use_discussion_room=False)
    result1 = await agent1.execute(ctx1)
    print(f"  Result: {result1['rel_status']} ({result1['rel_type']})")
    print()

    # Test 2: Multi-agent discussion room
    print("Test 2: Multi-Agent Discussion Room (use_discussion_room=True)")
    agent2 = RelationshipPredictionAgent(llm_router=router, use_discussion_room=True)
    result2 = await agent2.execute(ctx2)
    print(f"  Result: {result2['rel_status']} ({result2['rel_type']})")
    print()

    print("=" * 70)
    print("Comparison:")
    print("=" * 70)
    print(f"Single LLM:        {result1['rel_status']} ({result1['rel_type']})")
    print(f"Discussion Room:   {result2['rel_status']} ({result2['rel_type']})")
    print()
    print("Both methods completed successfully!")
    print("=" * 70)


if __name__ == "__main__":
    print("Running end-to-end tests...\n")

    # Test 1: Full relationship prediction with discussion room
    result1 = asyncio.run(test_relationship_prediction_with_discussion_room())

    print("\n\n")

    # Test 2: Comparison
    asyncio.run(test_comparison_single_vs_multi())

    print("\n✓ All end-to-end tests completed!")
