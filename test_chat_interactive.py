"""Interactive chat test for SimplePersonaAgent

Run this to have a real conversation with a bot and verify:
- Message format is correct
- Responses are natural
- No "role" errors occur
- Fallback works if API fails
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.agents.simple_persona import SimplePersonaAgent
from src.data.schema import PersonaProfile, OkCupidProfile, ExtractedFeatures


def create_test_persona():
    """Create a test persona for chatting"""
    profile = OkCupidProfile(
        age=25,
        sex="女",
        location="北京",
        orientation="straight"
    )

    features = ExtractedFeatures(
        openness=0.85,
        conscientiousness=0.6,
        extraversion=0.8,
        agreeableness=0.75,
        neuroticism=0.4,
        communication_style="casual",
        communication_confidence=0.8,
        core_values=["真诚", "成长", "快乐"],
        values_confidence=0.8,
        interest_categories={"旅行": 0.9, "美食": 0.8, "阅读": 0.7},
        relationship_goals="寻找有趣的灵魂",
        goals_confidence=0.8,
        personality_summary="开朗活泼，喜欢旅行和美食，热爱生活"
    )

    persona = PersonaProfile(
        profile_id="小雨",
        original_profile=profile,
        features=features,
        system_prompt="""你是小雨，25岁女生，住在北京。

性格特点：
- 开朗活泼，喜欢用颜文字(😊🤔😂等)
- 热爱旅行和美食，经常分享见闻
- 说话自然随意，像朋友聊天
- 偶尔会用"哈哈"、"嘿嘿"等语气词

兴趣爱好：
- 旅行：去过20多个城市，最喜欢成都和厦门
- 美食：是个吃货，喜欢探店
- 阅读：喜欢看小说和心理学书籍

价值观：
- 真诚待人，不喜欢虚伪
- 相信成长，愿意尝试新事物
- 追求快乐，享受当下

回复风格：
- 1-3句话，简短自然
- 多用颜文字表达情绪
- 像发微信一样说话
- 绝不使用*动作*这种RPG描述"""
    )

    return persona


def main():
    """Run interactive chat"""
    print("=" * 60)
    print("SimplePersonaAgent 交互式测试")
    print("=" * 60)
    print()
    print("你将与'小雨'聊天（25岁女生，北京，喜欢旅行美食）")
    print()
    print("命令：")
    print("  - 输入消息直接聊天")
    print("  - 输入 'quit' 或 'exit' 退出")
    print("  - 输入 'reset' 重置对话")
    print("  - 输入 'history' 查看对话历史")
    print("  - 输入 'debug' 查看最后一条消息的格式")
    print()
    print("=" * 60)
    print()

    # Create agent
    persona = create_test_persona()
    agent = SimplePersonaAgent(persona, temperature=0.8)

    # Send greeting
    greeting = agent.get_greeting()
    print(f"小雨: {greeting}")
    print()

    # Chat loop
    while True:
        try:
            # Get user input
            user_input = input("你: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("\n再见！👋")
                break

            elif user_input.lower() == 'reset':
                agent.reset()
                print("\n✓ 对话已重置\n")
                greeting = agent.get_greeting()
                print(f"小雨: {greeting}\n")
                continue

            elif user_input.lower() == 'history':
                print("\n--- 对话历史 ---")
                for i, msg in enumerate(agent.messages, 1):
                    role = "你" if msg["role"] == "user" else "小雨"
                    print(f"{i}. {role}: {msg['content']}")
                print(f"\n总计: {len(agent.messages)} 条消息")
                print("---------------\n")
                continue

            elif user_input.lower() == 'debug':
                if agent.messages:
                    last_msg = agent.messages[-1]
                    print("\n--- 最后一条消息格式 ---")
                    print(f"类型: {type(last_msg)}")
                    print(f"内容: {last_msg}")
                    print(f"有 'role' 键: {'role' in last_msg}")
                    print(f"有 'content' 键: {'content' in last_msg}")
                    if 'role' in last_msg:
                        print(f"role 值: {last_msg['role']}")
                    if 'content' in last_msg:
                        print(f"content 类型: {type(last_msg['content'])}")
                    print("------------------------\n")
                else:
                    print("\n还没有消息\n")
                continue

            # Generate response
            try:
                response = agent.generate_response(user_input)
                print(f"小雨: {response}")
                print()

            except Exception as e:
                print(f"\n❌ 错误: {e}")
                print(f"错误类型: {type(e).__name__}")
                import traceback
                print("\n详细错误信息:")
                traceback.print_exc()
                print()

        except KeyboardInterrupt:
            print("\n\n再见！👋")
            break

        except EOFError:
            print("\n\n再见！👋")
            break


if __name__ == "__main__":
    main()
