"""
Structured Reasoning — CoT and ReAct chains for complex agent decisions.

Provides two reasoning strategies:
1. ChainOfThought: step-by-step decomposition before answering
2. ReActReasoner: interleaved Thought → Action → Observation loops with tool use
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from loguru import logger

from src.agents.llm_router import router, AgentRole


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single step in a reasoning chain."""
    step_type: str  # "thought", "action", "observation", "conclusion"
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Complete trace of a reasoning process."""
    steps: list[ReasoningStep] = field(default_factory=list)
    final_answer: str = ""
    confidence: float = 0.0
    strategy: str = ""  # "cot" or "react"

    def add_step(self, step_type: str, content: str, **meta):
        self.steps.append(ReasoningStep(step_type=step_type, content=content, metadata=meta))

    def summary(self) -> str:
        lines = []
        for i, s in enumerate(self.steps):
            prefix = {"thought": "Think", "action": "Act", "observation": "Obs", "conclusion": "=>"}
            lines.append(f"[{prefix.get(s.step_type, s.step_type)}] {s.content[:120]}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Chain of Thought
# ---------------------------------------------------------------------------

class ChainOfThought:
    """
    Step-by-step reasoning before producing a final answer.

    The LLM is asked to break down its reasoning into explicit steps,
    then produce a final answer. This improves accuracy for complex
    multi-factor decisions like feature prediction and compatibility scoring.
    """

    COT_SYSTEM_SUFFIX = """

When answering, follow this structured reasoning process:
1. Break the problem into sub-questions
2. Analyze each sub-question step by step
3. Synthesize your findings
4. State your conclusion with a confidence level (0.0-1.0)

Format your response as:
<reasoning>
Step 1: [first sub-question and analysis]
Step 2: [second sub-question and analysis]
...
</reasoning>
<conclusion confidence="0.X">
[your final answer]
</conclusion>
"""

    @staticmethod
    def enhance_prompt(system: str) -> str:
        """Add CoT instructions to a system prompt."""
        return system + ChainOfThought.COT_SYSTEM_SUFFIX

    @staticmethod
    def reason(
        role: AgentRole,
        system: str,
        messages: list[dict],
        *,
        temperature: float = 0.3,
        max_tokens: int = 800,
    ) -> ReasoningTrace:
        """
        Execute a chain-of-thought reasoning call.

        Returns a ReasoningTrace with parsed steps and conclusion.
        """
        enhanced_system = ChainOfThought.enhance_prompt(system)

        text = router.chat(
            role=role,
            system=enhanced_system,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return ChainOfThought.parse_response(text)

    @staticmethod
    def parse_response(text: str) -> ReasoningTrace:
        """Parse a CoT response into structured steps."""
        trace = ReasoningTrace(strategy="cot")

        # Extract reasoning block
        reasoning_match = re.search(r"<reasoning>(.*?)</reasoning>", text, re.DOTALL)
        if reasoning_match:
            reasoning_text = reasoning_match.group(1).strip()
            # Parse numbered steps
            step_pattern = re.compile(r"Step\s*\d+\s*:\s*(.*?)(?=Step\s*\d+\s*:|$)", re.DOTALL)
            steps = step_pattern.findall(reasoning_text)
            if steps:
                for step_text in steps:
                    trace.add_step("thought", step_text.strip())
            else:
                # No numbered steps — treat whole block as one thought
                trace.add_step("thought", reasoning_text)

        # Extract conclusion
        conclusion_match = re.search(
            r'<conclusion\s+confidence="([^"]*)">(.*?)</conclusion>', text, re.DOTALL
        )
        if conclusion_match:
            trace.confidence = float(conclusion_match.group(1))
            trace.final_answer = conclusion_match.group(2).strip()
            trace.add_step("conclusion", trace.final_answer, confidence=trace.confidence)
        else:
            # Fallback: use everything after reasoning as conclusion
            if reasoning_match:
                remainder = text[reasoning_match.end():].strip()
                trace.final_answer = remainder if remainder else text
            else:
                trace.final_answer = text
            trace.add_step("conclusion", trace.final_answer)

        return trace


# ---------------------------------------------------------------------------
# ReAct Reasoner
# ---------------------------------------------------------------------------

class ReActReasoner:
    """
    ReAct (Reason + Act) — interleaved thinking and tool use.

    The LLM alternates between:
      - Thought: analyze what to do next
      - Action: call a tool or function
      - Observation: observe the result
    Until it reaches a final answer.

    This is used when the agent needs to gather external information
    (e.g., weather, user preferences, compatibility data) before responding.
    """

    REACT_SYSTEM_SUFFIX = """

You can use tools to gather information. Follow this pattern:
Thought: [analyze what you need to know]
Action: [tool_name(param=value)]
Observation: [you'll see the tool result here]
... (repeat if needed)
Thought: [final analysis]
Answer: [your response]

Available actions: {tool_list}

If you don't need any tools, just provide your Answer directly.
"""

    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps

    def reason(
        self,
        role: AgentRole,
        system: str,
        messages: list[dict],
        available_tools: Optional[dict[str, Any]] = None,
        *,
        temperature: float = 0.4,
        max_tokens: int = 600,
    ) -> ReasoningTrace:
        """
        Execute a ReAct reasoning loop.

        Args:
            available_tools: dict mapping tool names to callables.
                             If None, uses a no-tool thought-only mode.
        """
        trace = ReasoningTrace(strategy="react")

        tool_list = ", ".join(available_tools.keys()) if available_tools else "none"
        enhanced_system = system + self.REACT_SYSTEM_SUFFIX.format(tool_list=tool_list)

        working_messages = list(messages)

        for step in range(self.max_steps):
            text = router.chat(
                role=role,
                system=enhanced_system,
                messages=working_messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )

            # Parse the response for Thought/Action/Answer patterns
            thought_match = re.search(r"Thought:\s*(.*?)(?=Action:|Answer:|$)", text, re.DOTALL)
            action_match = re.search(r"Action:\s*(.*?)(?=Observation:|Thought:|Answer:|$)", text, re.DOTALL)
            answer_match = re.search(r"Answer:\s*(.*)", text, re.DOTALL)

            if thought_match:
                trace.add_step("thought", thought_match.group(1).strip())

            if action_match and available_tools:
                action_text = action_match.group(1).strip()
                trace.add_step("action", action_text)

                # Parse and execute action
                observation = self._execute_action(action_text, available_tools)
                trace.add_step("observation", observation)

                # Feed observation back
                working_messages = working_messages + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": f"Observation: {observation}"},
                ]
                continue

            if answer_match:
                trace.final_answer = answer_match.group(1).strip()
                trace.add_step("conclusion", trace.final_answer)
                break
            else:
                # No clear answer pattern — use the whole response
                trace.final_answer = text.strip()
                trace.add_step("conclusion", trace.final_answer)
                break

        return trace

    @staticmethod
    def _execute_action(action_text: str, tools: dict[str, Any]) -> str:
        """Parse and execute a tool action."""
        # Parse "tool_name(param=value, ...)" format
        match = re.match(r"(\w+)\((.*?)\)", action_text, re.DOTALL)
        if not match:
            return f"Could not parse action: {action_text}"

        tool_name = match.group(1)
        args_text = match.group(2).strip()

        if tool_name not in tools:
            return f"Unknown tool: {tool_name}"

        # Parse kwargs
        kwargs = {}
        if args_text:
            for pair in args_text.split(","):
                pair = pair.strip()
                if "=" in pair:
                    k, v = pair.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    kwargs[k] = v

        try:
            result = tools[tool_name](**kwargs)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------

def cot_reason(
    role: AgentRole,
    system: str,
    messages: list[dict],
    **kwargs,
) -> ReasoningTrace:
    """Shortcut for ChainOfThought reasoning."""
    return ChainOfThought.reason(role, system, messages, **kwargs)


def react_reason(
    role: AgentRole,
    system: str,
    messages: list[dict],
    tools: Optional[dict] = None,
    **kwargs,
) -> ReasoningTrace:
    """Shortcut for ReAct reasoning."""
    return ReActReasoner().reason(role, system, messages, tools, **kwargs)
