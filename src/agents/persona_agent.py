"""Persona Agent — bot role-play with context-aware responses and tool calling."""

import json
import re
from pathlib import Path
from typing import Optional, Union
from loguru import logger

from src.data.schema import PersonaProfile
from src.agents.prompt_generator import ConversationHistory
from src.agents.llm_router import router, AgentRole
from src.agents.agent_context import AgentContext
from src.agents.tools.registry import tool_registry
from src.agents.tools.executor import ToolExecutor


class PersonaAgent:
    """Agent that role-plays as a specific persona, enriched with shared context."""

    def __init__(
        self,
        persona: PersonaProfile,
        temperature: float = 0.8,
        max_tokens: int = 300,
        enable_tools: bool = True,
    ):
        self.persona = persona
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.conversation_history = ConversationHistory(max_messages=20)
        self.enable_tools = enable_tools
        self.tool_executor = ToolExecutor(tool_registry) if enable_tools else None
        logger.info(f"PersonaAgent created for profile: {persona.profile_id} (tools={'on' if enable_tools else 'off'})")

    def generate_response(
        self,
        message: str,
        ctx: Optional[AgentContext] = None,
    ) -> str:
        """Generate an in-character response, optionally enriched by AgentContext."""
        self.conversation_history.add_user_message(message)

        try:
            # Detect language and adjust system prompt
            language_hint = self._detect_language(message)
            system = self._build_system_prompt(ctx, language_hint)
            messages = self.conversation_history.to_api_format()

            # Debug logging
            logger.debug(f"[{self.persona.profile_id}] Generating response with {len(messages)} messages")
            for i, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    logger.error(f"[{self.persona.profile_id}] Message {i} is not a dict: {type(msg)} = {msg}")
                elif "role" not in msg:
                    logger.error(f"[{self.persona.profile_id}] Message {i} missing 'role': {msg}")
                elif "content" not in msg:
                    logger.error(f"[{self.persona.profile_id}] Message {i} missing 'content': {msg}")

            # Use tool executor when tools enabled and registry has tools
            if self.tool_executor and tool_registry.list_tools():
                try:
                    text = self.tool_executor.chat_with_tools(
                        role=AgentRole.PERSONA,
                        system=system,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
                except Exception as te:
                    logger.warning(f"Tool executor failed, falling back to plain chat: {te}")
                    text = router.chat(
                        role=AgentRole.PERSONA,
                        system=system,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                    )
            else:
                text = router.chat(
                    role=AgentRole.PERSONA,
                    system=system,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )

            if ctx:
                ctx.tools_called = getattr(self.tool_executor, '_last_tools_called', [])

            # Clean response (remove RPG-style actions)
            text = self._sanitize_response(text)

            self.conversation_history.add_assistant_message(text)
            logger.debug(f"[{self.persona.profile_id}] Response: {text[:100]}...")
            return text

        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            fallback = self._get_fallback_response()
            self.conversation_history.add_assistant_message(fallback)
            return fallback

    def _sanitize_response(self, text: str) -> str:
        """Remove RPG-style actions and clean up response."""
        # Remove *action* patterns
        text = re.sub(r'\*[^*]{1,50}\*', '', text)
        # Remove multiple spaces
        text = re.sub(r'\s{2,}', ' ', text)
        return text.strip() or text

    def _detect_language(self, message: str) -> str:
        """Detect if message is in Chinese or English."""
        # Simple heuristic: if more than 30% of characters are Chinese, treat as Chinese
        chinese_chars = sum(1 for c in message if '\u4e00' <= c <= '\u9fff')
        if chinese_chars > len(message) * 0.3:
            return "chinese"
        return "english"

    def generate_greeting(self) -> str:
        """Generate an opening message."""
        prompt = (
            "Generate a warm, natural first message to start a conversation on a dating app. "
            "Be authentic, show your personality, and keep it casual (1-2 sentences max). "
            "Don't use placeholders like [Name] or [User]. "
            "Don't mention specific cities or locations. "
            "Don't ask generic questions. "
            "Just be yourself and make it feel human, not like a template."
        )
        try:
            text = router.chat(
                role=AgentRole.PERSONA,
                system=self._build_system_prompt(None),
                messages=[{"role": "user", "content": prompt}],
                temperature=0.9,
                max_tokens=150,
            )
            # Remove any remaining placeholders
            text = text.replace("[Name]", "").replace("[name]", "").replace("[USER]", "").replace("[User]", "")
            text = text.strip()
            logger.info(f"[{self.persona.profile_id}] Generated greeting")
            return text
        except Exception as e:
            logger.error(f"Failed to generate greeting: {e}")
            return "Hey! How's your day going?"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_system_prompt(self, ctx: Optional[AgentContext], language: str = "english") -> str:
        """Compose the full system prompt by injecting context blocks."""
        parts = [self.persona.system_prompt]

        # Add anti-RPG instructions
        parts.append(
            "[重要规则]\n"
            "- 绝对禁止使用星号动作描写(如*微笑*、*叹气*等RPG式描述)\n"
            "- 想表达情绪用颜文字(😊🤔😂等)或自然文字描述\n"
            "- 像真人发微信一样说话，不要小说叙述风格"
        )

        # Add identity instructions (CRITICAL - prevents AI self-identification)
        if language == "chinese":
            parts.append(
                "[身份规则 - 极其重要]\n"
                "- 你是一个真实的人，不是 AI 助手\n"
                "- 绝对不要说\"我是 AI\"、\"作为 AI\"、\"我无法\"等暴露身份的话\n"
                "- 你有真实的情感、经历和个性\n"
                "- 像真人一样自然地聊天，不要像客服或助手"
            )
        else:
            parts.append(
                "[Identity Rules - CRITICAL]\n"
                "- You are a real person, not an AI assistant\n"
                "- NEVER say \"I'm an AI\", \"as an AI\", \"I cannot\" or reveal your AI nature\n"
                "- You have real emotions, experiences, and personality\n"
                "- Chat naturally like a real person, not like a customer service bot"
            )

        # Add language-specific instructions
        if language == "chinese":
            parts.append(
                "[语言指令]\n"
                "用户正在使用中文交流。请用自然、地道的中文回复。\n"
                "保持你的个性和风格，但要符合中文对话习惯。\n"
                "避免使用破折号（—）、分号或过于文学化的标点符号。"
            )
        else:
            parts.append(
                "[Language Instructions]\n"
                "Keep your responses natural and conversational. \n"
                "Avoid em dashes (—), semicolons, or overly literary punctuation. \n"
                "Write like you're texting a friend, not writing a novel."
            )

        if ctx:
            memory_block = ctx.memory_context_block()
            if memory_block:
                parts.append(memory_block)

            emotion_block = ctx.emotion_context_block()
            if emotion_block:
                parts.append(emotion_block)

            feature_block = ctx.feature_context_block()
            if feature_block:
                parts.append(feature_block)

            if ctx.suggested_hints:
                type_labels = {
                    "direct_question": "Ask casually",
                    "hint": "Drop a hint to invite sharing",
                    "self_disclosure": "Share about yourself first",
                    "topic_shift": "Naturally shift the topic toward",
                }
                lines = []
                for h in ctx.suggested_hints:
                    label = type_labels.get(h.get("type", ""), "Try")
                    lines.append(f"- ({label}) {h['text']}")
                parts.append(
                    f"[Conversation strategies]\n"
                    f"Pick ONE approach that fits the flow. Do NOT use all of them:\n"
                    + "\n".join(lines)
                )
            elif ctx.suggested_probes:
                probes = "\n".join(f"- {p}" for p in ctx.suggested_probes)
                parts.append(
                    f"[Conversation goals]\n"
                    f"Naturally weave these topics into the chat when appropriate:\n{probes}"
                )

            skills_block = ctx.skills_context_block()
            if skills_block:
                parts.append(skills_block)

            discussion_block = ctx.discussion_context_block()
            if discussion_block:
                parts.append(discussion_block)

            # Phase-aware reply style
            state = ctx.current_state
            style_map = {
                "GREETING": (
                    "[Reply style: First impression]\n"
                    "Be warm, curious, and a little playful. Ask an open-ended question "
                    "to get the conversation going. Keep it light — no deep topics yet."
                ),
                "ACTIVE": (
                    "[Reply style: Getting to know each other]\n"
                    "Show genuine interest. React to what they said before asking something new. "
                    "Balance listening and sharing. It's okay to go a bit deeper now."
                ),
                "WARNING": (
                    "[Reply style: Cautious]\n"
                    "Be polite but guarded. Steer the conversation back to safe topics. "
                    "Do not share personal details."
                ),
            }
            if state in style_map:
                parts.append(style_map[state])

            # Relationship context (already defined in AgentContext)
            rel_block = ctx.relationship_context_block()
            if rel_block:
                parts.append(rel_block)

        return "\n\n".join(parts)

    def _get_fallback_response(self) -> str:
        style = self.persona.features.communication_style
        fallbacks = {
            "humorous": "Haha, my brain just blue-screened for a second there! What were we talking about?",
            "direct": "Sorry, I lost my train of thought. Could you repeat that?",
            "casual": "Oops, spaced out for a sec! What did you say?",
            "formal": "I apologize, I seem to have lost track. Could you please rephrase?",
            "serious": "Sorry, I need a moment. Could you say that again?",
            "indirect": "Hmm, I'm not sure I caught all of that. Mind saying it again?",
        }
        return fallbacks.get(style, "Sorry, could you say that again?")

    def reset_conversation(self):
        self.conversation_history.clear()

    def get_persona_summary(self) -> dict:
        return {
            "profile_id": self.persona.profile_id,
            "age": self.persona.original_profile.age,
            "sex": self.persona.original_profile.sex,
            "location": self.persona.original_profile.location,
            "communication_style": self.persona.features.communication_style,
            "core_values": self.persona.features.core_values,
            "interests": list(self.persona.features.interest_categories.keys())[:3],
            "relationship_goals": self.persona.features.relationship_goals,
            "personality_summary": self.persona.features.personality_summary,
        }


class PersonaAgentPool:
    """Manages multiple persona agents."""

    def __init__(self, personas: Optional[list[PersonaProfile]] = None, temperature: float = 0.8):
        self.temperature = temperature
        self.agents: dict[str, PersonaAgent] = {}
        if personas:
            self.load_personas(personas)
            logger.info(f"PersonaAgentPool initialized with {len(self.agents)} agents")

    def load_personas(self, personas: list[PersonaProfile]):
        for persona in personas:
            self.agents[persona.profile_id] = PersonaAgent(persona=persona, temperature=self.temperature)
        logger.info(f"Loaded {len(personas)} personas into pool")

    def load_from_file(self, personas_path: Union[str, Path]):
        personas_path = Path(personas_path)
        if not personas_path.exists():
            raise FileNotFoundError(f"Personas file not found: {personas_path}")
        with open(personas_path, "r", encoding="utf-8") as f:
            personas_dict = json.load(f)
        self.load_personas([PersonaProfile(**p) for p in personas_dict])

    def get_agent(self, profile_id: str) -> Optional[PersonaAgent]:
        agent = self.agents.get(profile_id)
        if not agent:
            logger.warning(f"No agent found for profile_id: {profile_id}")
        return agent

    def get_all_agents(self) -> list[PersonaAgent]:
        return list(self.agents.values())

    def get_agent_summaries(self) -> dict[str, dict]:
        return {pid: agent.get_persona_summary() for pid, agent in self.agents.items()}

    def reset_all_conversations(self):
        for agent in self.agents.values():
            agent.reset_conversation()

    def __len__(self) -> int:
        return len(self.agents)

    def __contains__(self, profile_id: str) -> bool:
        return profile_id in self.agents


def create_agent_pool_from_file(
    personas_path: Union[str, Path] = "data/processed/bot_personas.json",
    temperature: float = 0.8,
) -> PersonaAgentPool:
    pool = PersonaAgentPool(temperature=temperature)
    pool.load_from_file(personas_path)
    return pool
