"""
Test Three-Layer Memory System
"""

import asyncio
from src.memory.three_layer_memory import ThreeLayerMemory
from src.agents.llm_router import LLMRouter


async def test_three_layer_memory():
    """Test complete three-layer memory workflow"""

    print("=" * 70)
    print("Three-Layer Memory System Test")
    print("=" * 70)
    print()

    router = LLMRouter()
    memory = ThreeLayerMemory(llm_router=router, working_memory_size=20)

    # Simulate 55 turns of conversation
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
        # Turn 10 - should trigger episodic compression
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
        # Turn 20 - should trigger episodic compression + consistency check
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
        # Turn 30 - should trigger episodic compression
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
        # Turn 40 - should trigger episodic compression + consistency check
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
        # Turn 50 - should trigger episodic compression + semantic reflection
        ("user", "I've been thinking about our trip."),
        ("bot", "Me too! I'm planning the route."),
        ("user", "That's so thoughtful!"),
        ("bot", "I want it to be perfect for us."),
        ("user", "You're amazing."),
    ]

    for i, (speaker, message) in enumerate(conversations):
        memory.add_to_working_memory(speaker, message)

        # Print status at key turns
        if (i + 1) in [10, 20, 30, 40, 50]:
            print(f"[Turn {i + 1}] Memory stats:")
            stats = memory.get_memory_stats()
            print(f"  Working memory: {stats['working_memory_size']} items")
            print(f"  Episodic memory: {stats['episodic_memory_count']} episodes")
            print(f"  Semantic memory: {stats['semantic_memory_count']} reflections")
            print()

    # Final statistics
    print("=" * 70)
    print("Final Memory Statistics")
    print("=" * 70)
    stats = memory.get_memory_stats()
    print(f"Total turns: {stats['current_turn']}")
    print(f"Working memory size: {stats['working_memory_size']}")
    print(f"Episodic memory count: {stats['episodic_memory_count']}")
    print(f"Semantic memory count: {stats['semantic_memory_count']}")
    print(f"Compression ratio: {stats['compression_ratio']:.2f}")
    print()

    # Test context retrieval
    print("=" * 70)
    print("Context Retrieval Test")
    print("=" * 70)
    print()

    full_context = memory.get_full_context(query="hiking trip")
    print("Full context (with query 'hiking trip'):")
    print(full_context[:500] + "..." if len(full_context) > 500 else full_context)
    print()

    # Test episodic retrieval
    print("=" * 70)
    print("Episodic Memory Retrieval")
    print("=" * 70)
    print()

    relevant_episodes = memory.retrieve_relevant_episodes("photography", top_k=2)
    print(f"Found {len(relevant_episodes)} relevant episodes for 'photography':")
    for ep in relevant_episodes:
        print(f"  Episode {ep.episode_id} (Turns {ep.turn_range[0]}-{ep.turn_range[1]})")
        print(f"    Summary: {ep.summary[:100]}...")
        print(f"    Key events: {ep.key_events}")
        print()

    # Test semantic memory
    print("=" * 70)
    print("Semantic Memory")
    print("=" * 70)
    print()

    semantic_context = memory.get_semantic_memory_context()
    print("Semantic memory context:")
    print(semantic_context)
    print()

    print("=" * 70)
    print("✓ Three-layer memory test completed!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(test_three_layer_memory())
