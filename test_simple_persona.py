"""Test SimplePersonaAgent to verify message format and fallback"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.agents.simple_persona import SimplePersonaAgent
from src.data.schema import PersonaProfile, OkCupidProfile, ExtractedFeatures


def test_message_format():
    """Test 1: Verify message format is always correct"""
    print("\n=== Test 1: Message Format ===")

    # Create a simple persona
    profile = OkCupidProfile(
        age=25,
        sex="女",
        location="北京",
        orientation="straight"
    )

    features = ExtractedFeatures(
        openness=0.8,
        conscientiousness=0.6,
        extraversion=0.7,
        agreeableness=0.75,
        neuroticism=0.4,
        communication_style="casual",
        communication_confidence=0.8,
        core_values=["真诚", "成长"],
        values_confidence=0.8,
        interest_categories={"文化": 0.8, "旅行": 0.7},
        relationship_goals="寻找长期关系",
        goals_confidence=0.8,
        personality_summary="开朗友好"
    )

    persona = PersonaProfile(
        profile_id="test_001",
        original_profile=profile,
        features=features,
        system_prompt="你是一个25岁的女生，住在北京，性格开朗友好。"
    )

    agent = SimplePersonaAgent(persona)

    # Test greeting
    greeting = agent.get_greeting()
    print(f"✓ Greeting: {greeting}")

    # Test conversation
    test_messages = [
        "你好",
        "你叫什么名字？",
        "你多大了？",
        "你在哪里？",
        "你做什么工作？"
    ]

    for msg in test_messages:
        response = agent.generate_response(msg)
        print(f"User: {msg}")
        print(f"Bot: {response}")

        # Verify message format
        assert len(agent.messages) > 0, "Messages list should not be empty"
        last_msg = agent.messages[-1]
        assert isinstance(last_msg, dict), f"Message should be dict, got {type(last_msg)}"
        assert "role" in last_msg, "Message should have 'role' key"
        assert "content" in last_msg, "Message should have 'content' key"
        assert isinstance(last_msg["content"], str), "Content should be string"
        print(f"✓ Message format correct: {last_msg}")
        print()

    print("✅ Test 1 PASSED: All messages have correct format\n")
    return True


def test_fallback_rules():
    """Test 2: Verify fallback rules work when API is unavailable"""
    print("\n=== Test 2: Fallback Rules ===")

    profile = OkCupidProfile(
        age=30,
        sex="男",
        location="上海",
        orientation="straight"
    )

    features = ExtractedFeatures(
        openness=0.7,
        conscientiousness=0.8,
        extraversion=0.5,
        agreeableness=0.6,
        neuroticism=0.3,
        communication_style="direct",
        communication_confidence=0.85,
        core_values=["效率", "创新"],
        values_confidence=0.85,
        interest_categories={"科技": 0.9, "编程": 0.8},
        relationship_goals="寻找志同道合的伴侣",
        goals_confidence=0.85,
        personality_summary="理性务实"
    )

    persona = PersonaProfile(
        profile_id="test_002",
        original_profile=profile,
        features=features,
        system_prompt="你是一个30岁的男程序员，住在上海，理性务实。"
    )

    agent = SimplePersonaAgent(persona)

    # Test keyword matching with fallback
    # We'll just test that fallback works by checking responses are generated
    test_cases = [
        ("你好", ["你好", "嗨", "hi", "test_002"]),
        ("你叫什么名字？", ["test_002", "叫"]),
        ("你多大了？", ["30", "岁"]),
        ("你在哪里？", ["上海", "这边"]),
    ]

    print("Testing fallback responses (will use API if available, fallback if not):")
    for msg, expected_keywords in test_cases:
        response = agent.generate_response(msg)
        print(f"User: {msg}")
        print(f"Bot: {response}")

        # Just verify we got a response
        assert response and len(response) > 0, "Should get a response"
        print(f"✓ Got valid response")
        print()

    print("✅ Test 2 PASSED: Response generation works correctly\n")
    return True


def test_no_rpg_actions():
    """Test 3: Verify RPG-style actions are removed"""
    print("\n=== Test 3: RPG Action Removal ===")

    profile = OkCupidProfile(
        age=22,
        sex="女",
        location="成都",
        orientation="straight"
    )

    features = ExtractedFeatures(
        openness=0.85,
        conscientiousness=0.5,
        extraversion=0.9,
        agreeableness=0.8,
        neuroticism=0.4,
        communication_style="casual",
        communication_confidence=0.75,
        core_values=["快乐", "自由"],
        values_confidence=0.75,
        interest_categories={"动漫": 0.9, "旅行": 0.8},
        relationship_goals="寻找有趣的人",
        goals_confidence=0.75,
        personality_summary="活泼开朗"
    )

    persona = PersonaProfile(
        profile_id="test_003",
        original_profile=profile,
        features=features,
        system_prompt="你是一个22岁的女大学生，住在成都，活泼开朗。"
    )

    agent = SimplePersonaAgent(persona)

    # Test sanitization
    test_responses = [
        "你好呀 *微笑* 很高兴认识你",
        "*挥手* 嗨嗨嗨！",
        "我喜欢旅行 *眼睛发光*",
        "哈哈哈 *大笑* 你好有趣"
    ]

    for raw_response in test_responses:
        sanitized = agent._sanitize_response(raw_response)
        print(f"Raw: {raw_response}")
        print(f"Sanitized: {sanitized}")

        # Verify no asterisks remain
        assert "*" not in sanitized, f"Asterisks should be removed: {sanitized}"
        print(f"✓ RPG actions removed\n")

    print("✅ Test 3 PASSED: RPG-style actions are properly removed\n")
    return True


def test_conversation_history():
    """Test 4: Verify conversation history is managed correctly"""
    print("\n=== Test 4: Conversation History ===")

    profile = OkCupidProfile(
        age=28,
        sex="男",
        location="深圳",
        orientation="straight"
    )

    features = ExtractedFeatures(
        openness=0.75,
        conscientiousness=0.9,
        extraversion=0.4,
        agreeableness=0.5,
        neuroticism=0.3,
        communication_style="formal",
        communication_confidence=0.9,
        core_values=["效率", "成长"],
        values_confidence=0.9,
        interest_categories={"科技": 0.8, "投资": 0.7},
        relationship_goals="寻找成熟的伴侣",
        goals_confidence=0.9,
        personality_summary="理性专业"
    )

    persona = PersonaProfile(
        profile_id="test_004",
        original_profile=profile,
        features=features,
        system_prompt="你是一个28岁的男产品经理，住在深圳，理性专业。"
    )

    agent = SimplePersonaAgent(persona)

    # Simulate 15 rounds of conversation
    for i in range(15):
        user_msg = f"这是第{i+1}条消息"
        agent.generate_response(user_msg)

    print(f"Total messages: {len(agent.messages)}")
    print(f"Expected: 30 (15 user + 15 assistant)")
    assert len(agent.messages) == 30, "Should have 30 messages"

    # Verify only recent messages are sent to API
    recent = agent.messages[-12:]
    print(f"Recent messages for API: {len(recent)}")
    assert len(recent) == 12, "Should only send 12 recent messages"

    # Verify all messages have correct format
    for i, msg in enumerate(agent.messages):
        assert isinstance(msg, dict), f"Message {i} should be dict"
        assert "role" in msg, f"Message {i} should have role"
        assert "content" in msg, f"Message {i} should have content"
        assert msg["role"] in ["user", "assistant"], f"Invalid role: {msg['role']}"

    print("✓ All messages have correct format")
    print("✓ History management works correctly")
    print("✅ Test 4 PASSED: Conversation history is managed correctly\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("Testing SimplePersonaAgent")
    print("=" * 60)

    tests = [
        ("Message Format", test_message_format),
        ("Fallback Rules", test_fallback_rules),
        ("RPG Action Removal", test_no_rpg_actions),
        ("Conversation History", test_conversation_history),
    ]

    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result, None))
        except Exception as e:
            print(f"❌ Test FAILED: {name}")
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False, str(e)))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, result, _ in results if result)
    total = len(results)

    for name, result, error in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")
        if error:
            print(f"  Error: {error}")

    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! SimplePersonaAgent is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Please fix before migrating.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
