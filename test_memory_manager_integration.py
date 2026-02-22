"""
Test MemoryManager with Three-Layer Memory Integration
"""

import asyncio
from src.memory.memory_manager import MemoryManager


async def test_memory_manager_with_three_layer():
    """Test MemoryManager with three-layer memory system enabled"""

    print("=" * 70)
    print("MemoryManager + Three-Layer Memory Integration Test")
    print("=" * 70)
    print()

    # Initialize with three-layer memory enabled
    manager = MemoryManager(user_id="test_user", use_three_layer=True)

    print("Simulating 55 turns of conversation...")
    print()

    conversations = [
        ("user", "Hi, I'm interested in hiking."),
        ("bot", "That's great! I love hiking too!"),
        ("user", "What's your favorite trail?"),
        ("bot", "I enjoy mountain trails with scenic views."),
        ("user", "We should go hiking together sometime!"),
        ("bot", "I'd love that! When are you free?"),
        ("user", "How about next weekend?"),
        ("bot", "Sounds perfect! I'm looking forward to it."),
        ("user", "Me too! You seem really nice."),
        ("bot", "Thank you! I think we have a lot in common."),
        ("user", "Do you like photography?"),
        ("bot", "Yes! I often take photos while hiking."),
        ("user", "That's amazing! We could combine both hobbies."),
        ("bot", "Absolutely! I'd love to share tips with you."),
        ("user", "This is so exciting!"),
        ("bot", "I feel the same way!"),
        ("user", "Tell me more about your photography style."),
        ("bot", "I focus on landscape and nature shots."),
        ("user", "I prefer portraits and candid moments."),
        ("bot", "That's a great complement to my style!"),
        ("user", "Have you traveled much?"),
        ("bot", "Yes, I've been to several national parks."),
        ("user", "Which one was your favorite?"),
        ("bot", "Yosemite - the views are breathtaking."),
        ("user", "I've always wanted to go there!"),
        ("bot", "We should plan a trip together!"),
        ("user", "That would be incredible!"),
        ("bot", "Let's make it happen soon."),
        ("user", "I'm so glad we met."),
        ("bot", "Me too! This feels special."),
        ("user", "What do you do for work?"),
        ("bot", "I'm a software engineer. You?"),
        ("user", "I'm a designer."),
        ("bot", "That's a great combination!"),
        ("user", "I think so too!"),
        ("bot", "We could collaborate on projects."),
        ("user", "I'd love that!"),
        ("bot", "Let's exchange contact info."),
        ("user", "Sure! Here's my number."),
        ("bot", "Thanks! I'll text you soon."),
        ("user", "I had a great time chatting with you."),
        ("bot", "Me too! Can't wait for our hike."),
        ("user", "Same here!"),
        ("bot", "Talk to you soon!"),
        ("user", "Bye!"),
        ("bot", "Bye!"),
        ("user", "Hey, just wanted to say hi again."),
        ("bot", "Hi! How are you?"),
        ("user", "Great! Excited for the weekend."),
        ("bot", "Me too! It's going to be fun."),
        ("user", "I've been thinking about our trip."),
        ("bot", "Me too! I'm planning the route."),
        ("user", "That's so thoughtful!"),
        ("bot", "I want it to be perfect for us."),
        ("user", "You're amazing."),
    ]

    for i, (speaker, message) in enumerate(conversations):
        manager.add_conversation_turn(speaker, message)

        if (i + 1) in [10, 20, 30, 40, 50]:
            print(f"[Turn {i + 1}] Memory stats:")
            stats = manager.get_memory_stats()
            print(f"  Current turn: {stats['current_turn']}")
            print(f"  Working memory: {stats['working_memory_size']} items")
            print(f"  Episodic memory: {stats['episodic_memory_count']} episodes")
            print(f"  Semantic memory: {stats['semantic_memory_count']} reflections")
            print()

    print("=" * 70)
    print("Final Memory Statistics")
    print("=" * 70)
    stats = manager.get_memory_stats()
    print(f"Total turns: {stats['current_turn']}")
    print(f"Working memory: {stats['working_memory_size']}")
    print(f"Episodic memory: {stats['episodic_memory_count']}")
    print(f"Semantic memory: {stats['semantic_memory_count']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}")
    print()

    print("=" * 70)
    print("Memory Retrieval Test")
    print("=" * 70)
    print()

    memories = manager.retrieve_relevant_memories("hiking trip", n=3)
    print(f"Retrieved {len(memories)} memory contexts for 'hiking trip':")
    for i, mem in enumerate(memories):
        print(f"\nMemory {i + 1}:")
        print(mem[:300] + "..." if len(mem) > 300 else mem)

    print()
    print("=" * 70)
    print("✓ Integration test completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_memory_manager_with_three_layer())
