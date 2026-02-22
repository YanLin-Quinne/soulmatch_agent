"""
简单集成测试: 验证v2.0核心组件
"""

import sys
sys.path.insert(0, '/Users/quinne/Desktop/soulmatch_agent_test')

from src.agents.agent_context import AgentContext
from src.agents.feature_transition_predictor import FeatureTransitionPredictor
from src.agents.milestone_evaluator import MilestoneEvaluator

def test_agent_context_extended_fields():
    """测试AgentContext扩展字段"""
    ctx = AgentContext(user_id="test_user")

    # 测试新字段
    assert ctx.participant_type == "bot"
    assert ctx.is_human == False
    assert ctx.rel_status == "stranger"
    assert ctx.rel_type == "other"
    assert ctx.sentiment_label == "neutral"
    assert isinstance(ctx.extended_features, dict)
    assert isinstance(ctx.relationship_snapshots, list)
    assert isinstance(ctx.feature_history, list)
    assert isinstance(ctx.milestone_reports, dict)

    print("✓ AgentContext扩展字段测试通过")

def test_feature_transition_predictor():
    """测试特征时序预测器"""
    predictor = FeatureTransitionPredictor()

    current_features = {
        "big_five_openness": 0.7,
        "trust_score": 0.6,
    }

    result = predictor.predict_next(
        current_features=current_features,
        emotion_trend="improving",
        relationship_status="acquaintance",
        memory_trigger=False,
    )

    assert "likely_to_change" in result
    assert "predicted_direction" in result
    assert "change_probability" in result
    assert "stable_features" in result

    # trust_score应该在likely_to_change中(高频变化)
    assert "trust_score" in result["likely_to_change"]

    print("✓ FeatureTransitionPredictor测试通过")
    print(f"  预测变化特征: {result['likely_to_change']}")

def test_milestone_evaluator():
    """测试里程碑评估器"""
    evaluator = MilestoneEvaluator()

    # 模拟10轮对话的快照
    snapshots = [
        {"turn": i, "rel_status": "stranger" if i < 5 else "acquaintance",
         "sentiment": "neutral" if i < 3 else "positive",
         "trust_score": 0.3 + i * 0.05}
        for i in range(1, 11)
    ]

    # 测试第10轮评估
    report = evaluator.evaluate(
        turn=10,
        feature_history=[],
        relationship_snapshots=snapshots,
        current_features={},
    )

    assert report["turn"] == 10
    assert report["type"] == "initial_assessment"
    assert "current_status" in report
    assert "predicted_status_at_turn_30" in report

    print("✓ MilestoneEvaluator测试通过")
    print(f"  第10轮评估: {report['message']}")

def test_relationship_context_block():
    """测试关系上下文块格式化"""
    ctx = AgentContext(user_id="test_user")
    ctx.relationship_result = {
        "rel_status": "crush",
        "rel_type": "love",
        "sentiment": "positive",
    }
    ctx.rel_status = "crush"
    ctx.rel_type = "love"
    ctx.sentiment_label = "positive"
    ctx.can_advance = True
    ctx.extended_features["trust_score"] = 0.75

    block = ctx.relationship_context_block()

    assert "crush" in block
    assert "love" in block
    assert "positive" in block
    assert "0.75" in block

    print("✓ relationship_context_block测试通过")
    print(f"  输出:\n{block}")

if __name__ == "__main__":
    print("=" * 60)
    print("SoulMatch v2.0 集成测试")
    print("=" * 60)

    test_agent_context_extended_fields()
    test_feature_transition_predictor()
    test_milestone_evaluator()
    test_relationship_context_block()

    print("\n" + "=" * 60)
    print("所有测试通过! ✓")
    print("=" * 60)
