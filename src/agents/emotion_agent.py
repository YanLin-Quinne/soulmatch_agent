"""Emotion Agent — detect, predict, and strategize around user emotions."""

import json
from typing import Dict, List, Optional
from loguru import logger

from src.agents.llm_router import router, AgentRole
from src.agents.emotion_predictor import EmotionPredictor, EMOTIONS


EMOTION_DETECTION_EXAMPLES = [
    {"message": "I'm so excited about our date tomorrow! Can't wait to see you", "emotion": "joy", "intensity": 0.9},
    {"message": "I really miss talking to you... haven't heard from you in days", "emotion": "sadness", "intensity": 0.7},
    {"message": "Why did you cancel on me again?! This is so frustrating", "emotion": "anger", "intensity": 0.8},
    {"message": "I'm nervous about meeting in person... what if it's awkward?", "emotion": "fear", "intensity": 0.6},
    {"message": "Oh wow, I didn't expect you to say that!", "emotion": "surprise", "intensity": 0.7},
    {"message": "That's really gross, I can't believe you like that", "emotion": "disgust", "intensity": 0.5},
    {"message": "Yeah, sounds good. What time works for you?", "emotion": "neutral", "intensity": 0.3},
    {"message": "You make me so happy! I think I'm falling for you", "emotion": "love", "intensity": 0.95},
]


class EmotionDetector:
    """Detects emotion in a message via LLMRouter."""

    def detect_emotion(self, message: str) -> Dict[str, any]:
        try:
            prompt = self._build_prompt(message)
            text = router.chat(
                role=AgentRole.EMOTION,
                system="You are an expert emotion classifier for dating app conversations.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=200,
                json_mode=True,
            )
            return self._parse(text)
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            return {"emotion": "neutral", "confidence": 0.5, "intensity": 0.3, "reasoning": "detection error"}

    def _build_prompt(self, message: str) -> str:
        examples = "\n".join(
            f'Message: "{ex["message"]}"\nEmotion: {ex["emotion"]}, Intensity: {ex["intensity"]}'
            for ex in EMOTION_DETECTION_EXAMPLES
        )
        return (
            f"Classify the following message into ONE of these emotions:\n"
            f"{', '.join(EMOTIONS)}\n\n"
            f"Rate intensity 0.0-1.0.\n\n"
            f"Examples:\n{examples}\n\n"
            f'Now classify:\nMessage: "{message}"\n\n'
            f'Respond with JSON: {{"emotion":"...","confidence":0.0-1.0,"intensity":0.0-1.0,"reasoning":"..."}}'
        )

    def _parse(self, text: str) -> Dict[str, any]:
        try:
            text = text.strip()
            if "```json" in text:
                text = text.split("```json")[1].split("```")[0].strip()
            elif "```" in text:
                text = text.split("```")[1].split("```")[0].strip()
            result = json.loads(text)
            if result.get("emotion") not in EMOTIONS:
                result["emotion"] = "neutral"
            result.setdefault("confidence", 0.7)
            result.setdefault("intensity", 0.5)
            result.setdefault("reasoning", "")
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            result["intensity"] = max(0.0, min(1.0, result["intensity"]))
            return result
        except (json.JSONDecodeError, KeyError):
            text_lower = text.lower()
            for emo in EMOTIONS:
                if emo in text_lower:
                    return {"emotion": emo, "confidence": 0.6, "intensity": 0.5, "reasoning": "parsed keyword"}
            return {"emotion": "neutral", "confidence": 0.5, "intensity": 0.3, "reasoning": "parse failed"}


class EmotionAgent:
    """Combines detection + Markov prediction + reply strategy."""

    def __init__(self, history_size: int = 20):
        self.detector = EmotionDetector()
        self.predictor = EmotionPredictor(history_window=5)
        self.history_size = history_size
        self.emotion_history: List[Dict[str, any]] = []

    def analyze_message(self, message: str) -> Dict[str, any]:
        detection = self.detector.detect_emotion(message)
        self.emotion_history.append({
            "message": message,
            "emotion": detection["emotion"],
            "confidence": detection["confidence"],
            "intensity": detection["intensity"],
        })
        if len(self.emotion_history) > self.history_size:
            self.emotion_history = self.emotion_history[-self.history_size:]

        seq = [h["emotion"] for h in self.emotion_history]
        prediction = self.predictor.predict_next_emotion(seq)
        strategy = self._reply_strategy(detection["emotion"])

        return {
            "current_emotion": detection,
            "predicted_next": prediction,
            "reply_strategy": strategy,
        }

    def _reply_strategy(self, emotion: str) -> Dict[str, any]:
        strategies = {
            "joy":      {"approach": "Resonate",  "tone": "enthusiastic and positive"},
            "love":     {"approach": "Reciprocate gently", "tone": "warm and affectionate"},
            "sadness":  {"approach": "Comfort",   "tone": "empathetic and supportive"},
            "anger":    {"approach": "Soothe",    "tone": "calm and understanding"},
            "fear":     {"approach": "Reassure",  "tone": "reassuring and patient"},
            "surprise": {"approach": "Explain",   "tone": "clear and friendly"},
            "disgust":  {"approach": "Understand", "tone": "respectful"},
            "neutral":  {"approach": "Engage",    "tone": "casual and friendly"},
        }
        return strategies.get(emotion, strategies["neutral"])

    def clear_history(self):
        self.emotion_history = []
