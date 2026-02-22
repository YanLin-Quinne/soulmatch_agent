"""Simple Persona Agent - Reliable conversation with fallback rules"""

import re
from typing import Optional
from loguru import logger

from src.data.schema import PersonaProfile
from src.agents.llm_router import router, AgentRole


class SimplePersonaAgent:
    """
    Simplified persona agent with guaranteed message format and fallback rules.

    Design principles from social-forecast demo:
    1. Messages always have correct format: {role: str, content: str}
    2. Conversation history is a simple list
    3. Fallback to rule-based responses if API fails
    4. No complex state management
    """

    def __init__(self, persona: PersonaProfile, temperature: float = 0.8):
        self.persona = persona
        self.temperature = temperature
        self.messages = []  # Simple list: [{role, content}, ...]
        self.fallback_rules = self._build_fallback_rules()
        logger.info(f"SimplePersonaAgent created: {persona.profile_id}")

    def generate_response(self, user_message: str) -> str:
        """Generate response with guaranteed format and fallback."""
        # Add user message (guaranteed format)
        self.messages.append({"role": "user", "content": user_message})

        # Try API call
        try:
            response = self._call_api()
            if response and response.strip():
                # Clean response (remove RPG-style actions)
                response = self._sanitize_response(response)
                self.messages.append({"role": "assistant", "content": response})
                return response
        except Exception as e:
            logger.warning(f"API failed for {self.persona.profile_id}: {e}")

        # Fallback to rules
        response = self._fallback_response(user_message)
        self.messages.append({"role": "assistant", "content": response})
        return response

    def _call_api(self) -> str:
        """Call LLM API with guaranteed message format."""
        # Only send recent messages (last 12)
        recent = self.messages[-12:] if len(self.messages) > 12 else self.messages

        # Verify format before sending
        for i, msg in enumerate(recent):
            if not isinstance(msg, dict):
                raise ValueError(f"Message {i} is not dict: {type(msg)}")
            if "role" not in msg or "content" not in msg:
                raise ValueError(f"Message {i} missing role/content: {msg}")
            if not isinstance(msg["content"], str):
                msg["content"] = str(msg["content"])

        # Build system prompt
        system = f"""{self.persona.system_prompt}

规则:
- 保持角色，绝不承认AI
- 回复1-3句，自然展现性格
- 绝对禁止使用星号动作描写(如*微笑*、*叹气*等)
- 想表达情绪用颜文字(😊🤔😂等)或自然文字
- 像真人发微信一样说话"""

        # Call router
        return router.chat(
            role=AgentRole.PERSONA,
            system=system,
            messages=recent,
            temperature=self.temperature,
            max_tokens=300
        )

    def _sanitize_response(self, text: str) -> str:
        """Remove RPG-style actions and clean up."""
        # Remove *action* patterns
        text = re.sub(r'\*[^*]{1,50}\*', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip() or text

    def _build_fallback_rules(self) -> dict:
        """Build keyword-based fallback rules."""
        profile = self.persona.original_profile
        age = profile.age if profile.age else 25
        name = getattr(profile, 'name', None) or self.persona.profile_id
        location = getattr(profile, 'location', None) or '这边'
        occupation = getattr(profile, 'occupation', None) or '工作'

        # Age-based greeting
        if age < 20:
            greetings = ['嗯 hi', '你好', '来了']
        elif age >= 60:
            greetings = [f'你好啊，我是{name}。', '你好你好，来聊天的？好好好。']
        else:
            greetings = [f'嗨！我是{name}，很高兴认识你~', '你好呀，来聊聊天吧', 'Hi！来认识新朋友~']

        # Build keyword responses based on profile
        keywords = {
            '你好|嗨|hi': greetings,
            '名字|你是': [f'我叫{name}~'],
            '年龄|多大': [f'我{age}岁~', f'{age}岁啦'],
            '哪里|城市|地方': [f'我在{location}', f'{location}这边'],
            '工作|职业|做什么': [f'我是{occupation}', f'做{occupation}的'],
        }

        # Generic fallbacks
        generic = [
            '嗯嗯是的~',
            '哈哈对',
            '你说得有道理',
            '确实诶',
            '有意思',
            '你呢？',
            '还行吧',
            '嗯这个我也想过',
        ]

        return {"keywords": keywords, "generic": generic}

    def _fallback_response(self, user_message: str) -> str:
        """Generate rule-based response."""
        msg_lower = user_message.lower()

        # Try keyword matching
        for pattern, responses in self.fallback_rules["keywords"].items():
            if re.search(pattern, msg_lower):
                import random
                return random.choice(responses)

        # Use generic fallback
        import random
        turn = len(self.messages) // 2
        return self.fallback_rules["generic"][turn % len(self.fallback_rules["generic"])]

    def get_greeting(self) -> str:
        """Generate initial greeting."""
        profile = self.persona.original_profile
        age = profile.age if profile.age else 25

        # Get name from profile_id if name not available
        name = getattr(profile, 'name', None) or self.persona.profile_id

        if age < 20:
            greetings = ['嗯 hi', '你好', '来了']
        elif age >= 60:
            greetings = [f'你好啊，我是{name}。', '你好你好，来聊天的？好好好。']
        else:
            greetings = [f'嗨！我是{name}，很高兴认识你~', '你好呀，来聊聊天吧', 'Hi！来认识新朋友~']

        import random
        return random.choice(greetings)

    def reset(self):
        """Clear conversation history."""
        self.messages = []
