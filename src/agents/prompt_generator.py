"""Prompt generation utilities for persona agents"""

from typing import Optional


class Message:
    """Single message in conversation history"""
    
    def __init__(self, role: str, content: str):
        self.role = role  # "user" or "assistant"
        self.content = content
    
    def to_dict(self) -> dict:
        """Convert to dictionary format for API"""
        return {"role": self.role, "content": self.content}


class ConversationHistory:
    """Manages conversation history with sliding window"""
    
    def __init__(self, max_messages: int = 20):
        """
        Args:
            max_messages: Maximum number of messages to retain (10 rounds = 20 messages)
        """
        self.messages: list[Message] = []
        self.max_messages = max_messages
    
    def add_message(self, role: str, content: str):
        """Add a message to history"""
        self.messages.append(Message(role, content))
        
        # Keep only the most recent messages
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages:]
    
    def add_user_message(self, content: str):
        """Add a user message"""
        self.add_message("user", content)
    
    def add_assistant_message(self, content: str):
        """Add an assistant message"""
        self.add_message("assistant", content)
    
    def to_api_format(self) -> list[dict]:
        """Convert to API format (list of message dicts)"""
        return [msg.to_dict() for msg in self.messages]
    
    def clear(self):
        """Clear all messages"""
        self.messages = []
    
    def get_last_n_rounds(self, n: int = 5) -> list[Message]:
        """Get last N conversation rounds (2 messages per round)"""
        num_messages = n * 2
        return self.messages[-num_messages:] if len(self.messages) > num_messages else self.messages
    
    def __len__(self) -> int:
        """Get number of messages"""
        return len(self.messages)


def format_conversation_context(
    history: ConversationHistory, 
    include_last_n: Optional[int] = None
) -> str:
    """
    Format conversation history as readable text context
    
    Args:
        history: Conversation history
        include_last_n: Only include last N rounds (default: all)
    
    Returns:
        Formatted conversation string
    """
    if include_last_n:
        messages = history.get_last_n_rounds(include_last_n)
    else:
        messages = history.messages
    
    if not messages:
        return "No previous conversation."
    
    formatted = []
    for msg in messages:
        role_label = "User" if msg.role == "user" else "You"
        formatted.append(f"{role_label}: {msg.content}")
    
    return "\n".join(formatted)


def create_greeting_prompt(persona_name: Optional[str] = None) -> str:
    """
    Create a natural first message prompt
    
    Args:
        persona_name: Optional name to personalize greeting
    
    Returns:
        Prompt for generating a greeting
    """
    if persona_name:
        return f"Generate a warm, natural first message to start a conversation. Keep it casual and friendly, in line with your personality. Don't mention your name unless it feels natural."
    else:
        return "Generate a warm, natural first message to start a conversation on a dating app. Be authentic and show your personality."


def enhance_prompt_with_context(
    base_message: str,
    conversation_history: ConversationHistory,
    additional_context: Optional[str] = None
) -> str:
    """
    Enhance a message with conversation context
    
    Args:
        base_message: The current user message
        conversation_history: Previous conversation
        additional_context: Optional additional context to include
    
    Returns:
        Enhanced message with context
    """
    parts = [base_message]
    
    if additional_context:
        parts.insert(0, f"Context: {additional_context}")
    
    if len(conversation_history) > 0:
        context = format_conversation_context(conversation_history, include_last_n=3)
        parts.insert(0, f"Recent conversation:\n{context}\n---")
    
    return "\n\n".join(parts)


def extract_system_prompt_summary(system_prompt: str, max_length: int = 200) -> str:
    """
    Extract a brief summary from system prompt for logging/debugging
    
    Args:
        system_prompt: Full system prompt
        max_length: Maximum length of summary
    
    Returns:
        Brief summary of the persona
    """
    # Try to extract personality summary section
    if "PERSONALITY:" in system_prompt:
        lines = system_prompt.split("\n")
        for i, line in enumerate(lines):
            if "PERSONALITY:" in line and i + 1 < len(lines):
                summary = lines[i + 1].strip()
                if len(summary) > max_length:
                    return summary[:max_length] + "..."
                return summary
    
    # Fallback: just truncate
    if len(system_prompt) > max_length:
        return system_prompt[:max_length] + "..."
    return system_prompt
