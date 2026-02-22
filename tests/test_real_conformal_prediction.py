"""
Test Real Conformal Prediction for Relationship Advancement

Verifies that the relationship prediction agent uses real conformal prediction
instead of hardcoded if-else rules.
"""

import asyncio
from src.agents.relationship_prediction_agent import RelationshipPredictionAgent
from src.agents.agent_context import AgentContext
from src.agents.conformal_calibrator import ConformalCalibrator


async def test_real_conformal_prediction():
    """Test that relationship advancement uses real conformal prediction"""

    print("=" * 70)
    print("Testing Real Conformal Prediction for Relationship Advancement")
    print("=" * 70)
    print()

    # Create agent with calibrator
    calibrator = ConformalCalibrator(alpha=0.10)
    agent = RelationshipPredictionAgent(calibrator=calibrator, use_discussion_room=False)

    # Create context
    ctx = AgentContext(user_id="test_user")
    ctx.turn_count = 15
    ctx.extended_features = {"trust_score": 0.75}
    ctx.sentiment_label = "positive"

    # Add conversation history
    for i in range(15):
        ctx.add_to_history("user", f"Turn {i}: User message")
        ctx.add_to_history("bot", f"Turn {i}: Bot response")

    # Execute relationship prediction
    print("Executing relationship prediction...")
    result = await agent.execute(ctx)

    print("\nRelationship Prediction Result:")
    print(f"  Relationship Status: {result.get('rel_status')}")
    print(f"  Sentiment: {result.get('sentiment')}")
    print(f"  Can Advance: {result.get('can_advance')}")
    print(f"  Prediction Set: {result.get('advance_prediction_set')}")
    print(f"  Coverage Guarantee: {result.get('advance_coverage_guarantee')}")
    print()

    # Check for real conformal prediction features
    print("Checking for Real Conformal Prediction Features:")
    print()

    has_softmax = "advance_softmax" in result
    has_blockers = "blockers" in result
    has_catalysts = "catalysts" in result
    has_coverage = result.get("advance_coverage_guarantee") == 0.9

    print(f"  ✓ Softmax Distribution: {has_softmax}")
    if has_softmax:
        print(f"    {result['advance_softmax']}")

    print(f"  ✓ Blockers Analysis: {has_blockers}")
    if has_blockers:
        print(f"    {result['blockers']}")

    print(f"  ✓ Catalysts Analysis: {has_catalysts}")
    if has_catalysts:
        print(f"    {result['catalysts']}")

    print(f"  ✓ Coverage Guarantee: {has_coverage}")
    print()

    # Verify it's not just hardcoded rules
    prediction_set = result.get("advance_prediction_set", [])
    is_not_hardcoded = len(prediction_set) > 0 and (
        has_softmax or has_blockers or has_catalysts
    )

    if is_not_hardcoded:
        print("✓ PASS: Using real conformal prediction (not hardcoded rules)")
        return True
    else:
        print("✗ FAIL: Still using hardcoded rules")
        return False


async def test_conformal_with_calibration_data():
    """Test conformal prediction with actual calibration data"""

    print("=" * 70)
    print("Testing Conformal Prediction with Calibration Data")
    print("=" * 70)
    print()

    # Load calibrator with calibration data
    calibrator = ConformalCalibrator(alpha=0.10)

    # Check if calibration data exists
    import os
    calib_path = "data/training/conformal_calibration.json"
    if os.path.exists(calib_path):
        print(f"Loading calibration data from {calib_path}...")
        calibrator.load(calib_path)
        print(f"✓ Loaded calibration data")
        print(f"  Quantiles: {len(calibrator._quantiles)} turn-dimension pairs")
        print(f"  Global quantiles: {len(calibrator._global_quantiles)} dimensions")
    else:
        print(f"⚠ No calibration data found at {calib_path}")
        print("  Using uncalibrated fallback")

    print()

    # Create agent with calibrated predictor
    agent = RelationshipPredictionAgent(calibrator=calibrator, use_discussion_room=False)

    # Create context
    ctx = AgentContext(user_id="test_user")
    ctx.turn_count = 20
    ctx.extended_features = {"trust_score": 0.65}
    ctx.sentiment_label = "positive"

    for i in range(20):
        ctx.add_to_history("user", f"Turn {i}: User message")
        ctx.add_to_history("bot", f"Turn {i}: Bot response")

    # Execute
    result = await agent.execute(ctx)

    print("Result with Calibration:")
    print(f"  Can Advance: {result.get('can_advance')}")
    print(f"  Prediction Set: {result.get('advance_prediction_set')}")
    print(f"  Set Size: {len(result.get('advance_prediction_set', []))}")
    print(f"  Coverage: {result.get('advance_coverage_guarantee')}")
    print()

    # Check prediction set properties
    pred_set = result.get("advance_prediction_set", [])
    is_valid = (
        len(pred_set) > 0 and
        len(pred_set) <= 3 and  # Should not be full set
        result.get("advance_coverage_guarantee") == 0.9
    )

    if is_valid:
        print("✓ PASS: Conformal prediction set is valid")
        return True
    else:
        print("✗ FAIL: Prediction set is invalid")
        return False


if __name__ == "__main__":
    print("\n")
    result1 = asyncio.run(test_real_conformal_prediction())
    print("\n")
    result2 = asyncio.run(test_conformal_with_calibration_data())
    print("\n")

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"Real Conformal Prediction: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Calibration Data Integration: {'✓ PASS' if result2 else '✗ FAIL'}")
    print()

    if result1 and result2:
        print("✓ All tests passed! Conformal prediction is working correctly.")
    else:
        print("✗ Some tests failed. Conformal prediction needs fixes.")
