"""
AgentContext — shared mutable context passed through all agents each turn.

Every agent reads from and writes to this context so that downstream agents
benefit from upstream analysis within the same turn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class AgentContext:
    """Shared per-turn context that flows through the orchestrator pipeline."""

    # --- Conversation state ---
    user_id: str = ""
    bot_id: str = ""
    turn_count: int = 0
    conversation_history: list[dict] = field(default_factory=list)

    # --- Current turn input ---
    user_message: str = ""

    # --- Emotion (written by EmotionAgent) ---
    current_emotion: Optional[str] = None
    emotion_confidence: float = 0.0
    emotion_intensity: float = 0.0
    emotion_history: list[str] = field(default_factory=list)
    reply_strategy: Optional[str] = None

    # --- Feature predictions (written by FeaturePredictionAgent) ---
    predicted_features: dict[str, Any] = field(default_factory=dict)
    feature_confidences: dict[str, float] = field(default_factory=dict)
    low_confidence_features: list[str] = field(default_factory=list)

    # --- Memory (written by MemoryManager) ---
    retrieved_memories: list[str] = field(default_factory=list)
    memory_summary: str = ""

    # --- Scam detection (written by ScamDetectionAgent) ---
    scam_risk_score: float = 0.0
    scam_warning_level: str = "none"
    scam_warning_message: str = ""

    # --- Question strategy (written by QuestionStrategyAgent) ---
    suggested_probes: list[str] = field(default_factory=list)

    # --- Tool use (written by ToolExecutor) ---
    tool_results: dict[str, Any] = field(default_factory=dict)
    tools_called: list[str] = field(default_factory=list)

    # --- Discussion (written by DiscussionEngine) ---
    discussion_synthesis: str = ""

    # --- Skills (written by SkillRegistry) ---
    active_skills: list[str] = field(default_factory=list)
    skill_prompt_additions: list[str] = field(default_factory=list)

    # --- Bot response (written by PersonaAgent) ---
    bot_response: str = ""

    # --- Extended features (v2.0) ---
    participant_type: str = "bot"  # "bot" | "human"
    is_human: bool = False
    extended_features: dict[str, Any] = field(default_factory=dict)

    # --- Relationship prediction (written by RelationshipPredictionAgent) ---
    relationship_result: Optional[dict] = None
    rel_status: str = "stranger"
    rel_type: str = "other"
    sentiment_label: str = "neutral"
    can_advance: Optional[bool] = None
    relationship_snapshots: list[dict] = field(default_factory=list)

    # --- Feature time-series ---
    feature_history: list[dict] = field(default_factory=list)
    predicted_feature_changes: dict[str, Any] = field(default_factory=dict)

    # --- Milestone evaluation ---
    milestone_reports: dict[int, dict] = field(default_factory=dict)

    # --- Virtual space (simplified) ---
    virtual_room: str = "lobby"
    virtual_position: tuple[float, float] = (0.0, 0.0)

    # --- Anti-AI mechanism ---
    reply_delay_seconds: float = 0.0

    # --- Multi-participant session ---
    session_participants: list[str] = field(default_factory=list)
    active_pair: tuple[str, str] = ("", "")

    # -----------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------

    def add_to_history(self, speaker: str, message: str):
        from loguru import logger
        # Ensure message is string
        if not isinstance(message, str):
            logger.warning(f"[AgentContext] Converting non-string message to string: {type(message)}")
            message = str(message)

        self.conversation_history.append({
            "speaker": speaker,
            "message": message,
            "turn": self.turn_count,
        })
        # Keep last 50 messages
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]

    def recent_history(self, n: int = 10) -> list[dict]:
        return self.conversation_history[-n:]

    def history_as_messages(self) -> list[dict[str, str]]:
        """Convert to OpenAI/Anthropic message format."""
        from loguru import logger
        out = []
        for i, h in enumerate(self.conversation_history):
            try:
                role = "user" if h["speaker"] == "user" else "assistant"
                message = h["message"]

                # Ensure message is string
                if not isinstance(message, str):
                    logger.warning(f"[AgentContext] Converting non-string message to string at index {i}: {type(message)}")
                    message = str(message)

                out.append({"role": role, "content": message})
            except Exception as e:
                logger.error(f"[AgentContext] Error processing history item {i}: {e}. Item: {h}")
                continue
        return out

    def memory_context_block(self) -> str:
        """Format retrieved memories as a text block for prompt injection."""
        if not self.retrieved_memories:
            return ""
        lines = "\n".join(f"- {m}" for m in self.retrieved_memories)
        return f"[Relevant memories about this user]\n{lines}"

    def feature_context_block(self) -> str:
        """Format predicted features as a text block for prompt injection."""
        if not self.predicted_features:
            return ""
        lines = []
        for k, v in self.predicted_features.items():
            conf = self.feature_confidences.get(k, 0)
            lines.append(f"- {k}: {v} (confidence: {conf:.0%})")
        return f"[Predicted user traits]\n" + "\n".join(lines)

    def emotion_context_block(self) -> str:
        """Format current emotion as a text block for prompt injection."""
        if not self.current_emotion:
            return ""
        parts = [f"[User emotional state] {self.current_emotion}"]
        if self.reply_strategy:
            parts.append(f"[Suggested approach] {self.reply_strategy}")
        return "\n".join(parts)

    def discussion_context_block(self) -> str:
        """Format discussion synthesis as a text block for prompt injection."""
        if not self.discussion_synthesis:
            return ""
        return f"[Expert advisory brief]\n{self.discussion_synthesis}"

    def skills_context_block(self) -> str:
        """Format active skill prompt additions as a text block."""
        if not self.skill_prompt_additions:
            return ""
        return "\n\n".join(self.skill_prompt_additions)

    def relationship_context_block(self) -> str:
        """Format relationship state for PersonaAgent injection."""
        if not self.relationship_result:
            return ""
        parts = [
            f"[Relationship state] {self.rel_status} ({self.rel_type})",
            f"[Sentiment] {self.sentiment_label}",
        ]
        if self.can_advance is not None:
            parts.append(f"[Can advance] {'Yes' if self.can_advance else 'No/Uncertain'}")
        if self.extended_features.get("trust_score"):
            trust = self.extended_features["trust_score"]
            parts.append(f"[Trust level] {trust:.2f}")
        return "\n".join(parts)
