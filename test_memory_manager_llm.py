"""
Test MemoryManager LLM Decision Making

Verifies that _llm_decide actually works and doesn't silently fail with NameError.
"""

import asyncio
from src.memory.memory_manager import MemoryManager


def test_llm_decide():
    """Test that LLM memory decision actually works"""

    print("=" * 70)
    print("Testing MemoryManager LLM Decision")
    print("=" * 70)
    print()

    # Create manager with LLM enabled
    manager = MemoryManager(user_id="test_user", use_llm=True, use_three_layer=False)

    # Add some conversation turns
    manager.add_conversation_turn("user", "I love hiking in the mountains")
    manager.add_conversation_turn("bot", "That's great! I enjoy hiking too")
    manager.add_conversation_turn("user", "Do you like photography?")
    manager.add_conversation_turn("bot", "Yes, I take photos while hiking")

    # Test LLM decision
    recent = manager.conversation_history[-4:]
    print("Recent conversation:")
    for msg in recent:
        print(f"  {msg['speaker']}: {msg['message']}")
    print()

    print("Calling _llm_decide...")
    try:
        action = manager._llm_decide(recent, current_features=None)
        print(f"✓ LLM decision succeeded!")
        print(f"  Operation: {action.operation}")
        print(f"  Reasoning: {action.reasoning}")
        print()

        if action.operation == "NOOP" and action.reasoning and "failed" in action.reasoning.lower():
            print("⚠ WARNING: LLM decision returned NOOP due to error!")
            print("  This means the LLM call failed silently.")
            return False
        else:
            print("✓ LLM decision working correctly!")
            print(f"  Note: NOOP is valid if LLM decides nothing needs to be remembered")
            return True

    except Exception as e:
        print(f"✗ LLM decision failed with exception: {e}")
        return False


def test_memory_manager_with_three_layer():
    """Test MemoryManager with three-layer memory"""

    print("=" * 70)
    print("Testing MemoryManager with Three-Layer Memory")
    print("=" * 70)
    print()

    manager = MemoryManager(user_id="test_user", use_llm=True, use_three_layer=True)

    # Add 15 turns to trigger episodic compression
    for i in range(15):
        manager.add_conversation_turn("user", f"Turn {i}: User message")
        manager.add_conversation_turn("bot", f"Turn {i}: Bot response")

    stats = manager.get_memory_stats()
    print("Memory statistics:")
    print(f"  Total turns: {stats['current_turn']}")
    print(f"  Episodic memories: {stats.get('episodic_memory_count', 0)}")
    print()

    if stats.get('episodic_memory_count', 0) > 0:
        print("✓ Three-layer memory working correctly!")
        return True
    else:
        print("✗ Three-layer memory not creating episodic memories!")
        return False


if __name__ == "__main__":
    print("\n")
    result1 = test_llm_decide()
    print("\n")
    result2 = test_memory_manager_with_three_layer()
    print("\n")

    print("=" * 70)
    print("Test Summary")
    print("=" * 70)
    print(f"LLM Decision: {'✓ PASS' if result1 else '✗ FAIL'}")
    print(f"Three-Layer Memory: {'✓ PASS' if result2 else '✗ FAIL'}")
    print()

    if result1 and result2:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed!")
