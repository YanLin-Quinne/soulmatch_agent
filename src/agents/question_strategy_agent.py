"""Question Strategy Agent — generates structured conversation strategies with hint types."""

import json
from typing import Dict, List
from loguru import logger

from src.agents.llm_router import router, AgentRole


class QuestionStrategyAgent:
    """
    Given low-confidence feature keys and conversation history, suggests 1-3 natural
    conversation strategies with different approach types:
    - direct_question: A casual but direct question
    - hint: An indirect hint that invites the user to share
    - self_disclosure: Bot shares about itself first to encourage reciprocity
    - topic_shift: A natural topic transition that leads toward the target trait
    """

    def suggest_probes(
        self,
        low_confidence_features: List[str],
        conversation_history: List[dict],
        max_probes: int = 3,
    ) -> List[str]:
        """Legacy interface: returns flat list of probe strings."""
        hints = self.suggest_hints(low_confidence_features, conversation_history, max_probes)
        return [h["text"] for h in hints]

    def suggest_hints(
        self,
        low_confidence_features: List[str],
        conversation_history: List[dict],
        max_hints: int = 3,
    ) -> List[Dict[str, str]]:
        """Return structured hints: [{"type": "hint|self_disclosure|...", "text": "..."}]"""
        if not low_confidence_features:
            return []

        recent = "\n".join(
            f"{m['speaker']}: {m['message']}" for m in conversation_history[-6:]
        )

        prompt = (
            f"We're chatting on a dating app and trying to naturally learn about the user.\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"We still need to learn about these traits: {', '.join(low_confidence_features[:8])}\n\n"
            f"Suggest {max_hints} conversation strategies using DIFFERENT approach types:\n"
            f"- direct_question: A casual question (not interview-like)\n"
            f"- hint: An indirect comment that invites sharing (e.g. 'I've been thinking about...')\n"
            f"- self_disclosure: Share something about yourself first to encourage reciprocity\n"
            f"- topic_shift: A natural segue to a new topic related to the trait\n\n"
            f"Use at least 2 different types. Keep everything casual — dating chat, NOT interview.\n\n"
            f'Respond with JSON: {{"hints":[{{"type":"hint","text":"..."}},{{"type":"self_disclosure","text":"..."}}]}}'
        )

        try:
            text = router.chat(
                role=AgentRole.QUESTION,
                system="You are a conversational strategist for a dating app.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=300,
                json_mode=True,
            )
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            data = json.loads(text)
            hints = data.get("hints", [])
            # Validate structure
            valid = []
            for h in hints[:max_hints]:
                if isinstance(h, dict) and "text" in h:
                    h.setdefault("type", "direct_question")
                    valid.append(h)
                elif isinstance(h, str):
                    valid.append({"type": "direct_question", "text": h})
            return valid
        except Exception as e:
            logger.error(f"Question strategy failed: {e}")
            return []

    def suggest_accuracy_boosting_probes(
        self,
        feature_confidences: Dict[str, float],
        conversation_history: List[dict],
        threshold: float = 0.6,
        max_probes: int = 3,
    ) -> List[Dict[str, str]]:
        """Suggest probes that target the features with lowest confidence to boost prediction accuracy.

        Unlike suggest_hints which targets explicitly low-confidence features,
        this method analyses the full confidence distribution and focuses on
        features just below the threshold that would benefit most from one
        more data point.
        """
        # Rank features by confidence ascending, pick those below threshold
        candidates = sorted(
            [(k, v) for k, v in feature_confidences.items() if v < threshold],
            key=lambda x: x[1],
        )
        if not candidates:
            return []

        target_features = [k for k, _ in candidates[:max_probes * 2]]

        recent = "\n".join(
            f"{m['speaker']}: {m['message']}" for m in conversation_history[-6:]
        )

        conf_summary = ", ".join(
            f"{k}: {v:.0%}" for k, v in candidates[:max_probes * 2]
        )

        prompt = (
            f"We need to boost prediction accuracy for a dating-app user.\n\n"
            f"Recent conversation:\n{recent}\n\n"
            f"Features with low prediction confidence:\n{conf_summary}\n\n"
            f"Target features to probe: {', '.join(target_features)}\n\n"
            f"Suggest {max_probes} conversation strategies that would most efficiently "
            f"reveal information about multiple low-confidence features at once.\n"
            f"Each probe should naturally cover 2-3 features if possible.\n"
            f"Use different approach types: direct_question, hint, self_disclosure, topic_shift.\n\n"
            f'Respond with JSON: {{"probes":[{{"type":"hint","text":"...","targets":["feature1","feature2"]}}]}}'
        )

        try:
            text = router.chat(
                role=AgentRole.QUESTION,
                system="You are a conversational strategist optimising for information gain in a dating app.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=400,
                json_mode=True,
            )
            text = text.strip()
            # Strip markdown fences
            if "json" in text and text.count(chr(96) * 3) >= 2:
                parts = text.split(chr(96) * 3)
                text = parts[1].lstrip("json").strip() if len(parts) >= 3 else text
            data = json.loads(text)
            probes = data.get("probes", [])
            valid: List[Dict[str, str]] = []
            for p in probes[:max_probes]:
                if isinstance(p, dict) and "text" in p:
                    p.setdefault("type", "direct_question")
                    p.setdefault("targets", target_features[:2])
                    valid.append(p)
                elif isinstance(p, str):
                    valid.append({"type": "direct_question", "text": p, "targets": target_features[:2]})
            return valid
        except Exception as e:
            logger.error(f"Accuracy boosting probes failed: {e}")
            return []
