"""Emotion Agent for detecting and analyzing emotions in conversations"""

import json
from typing import Dict, List, Optional, Literal
from loguru import logger

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    logger.warning("anthropic package not available")

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    logger.warning("openai package not available")

from src.config import settings
from src.agents.emotion_predictor import EmotionPredictor, EMOTIONS


# Few-shot examples for emotion detection
EMOTION_DETECTION_EXAMPLES = [
    {"message": "I'm so excited about our date tomorrow! Can't wait to see you 😊", "emotion": "joy", "intensity": 0.9},
    {"message": "I really miss talking to you... haven't heard from you in days", "emotion": "sadness", "intensity": 0.7},
    {"message": "Why did you cancel on me again?! This is so frustrating", "emotion": "anger", "intensity": 0.8},
    {"message": "I'm nervous about meeting in person... what if it's awkward?", "emotion": "fear", "intensity": 0.6},
    {"message": "Oh wow, I didn't expect you to say that!", "emotion": "surprise", "intensity": 0.7},
    {"message": "That's really gross, I can't believe you like that", "emotion": "disgust", "intensity": 0.5},
    {"message": "Yeah, sounds good. What time works for you?", "emotion": "neutral", "intensity": 0.3},
    {"message": "You make me so happy! I think I'm falling for you ❤️", "emotion": "love", "intensity": 0.95},
]


class EmotionDetector:
    """Detects emotion in a message using LLM"""
    
    def __init__(
        self,
        use_claude: bool = True,
        model_name: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize emotion detector
        
        Args:
            use_claude: Use Claude API (Haiku for speed/cost) vs OpenAI
            model_name: Override default model (claude-sonnet-4-20250514 or gpt-4o-mini)
            temperature: Lower = more consistent (0.3 recommended for classification)
        """
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE
        self.temperature = temperature
        
        # Initialize API client
        if self.use_claude:
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            # Use Haiku for fast, cheap emotion detection
            self.model = model_name or "claude-sonnet-4-20250514"
            logger.info(f"EmotionDetector initialized with Claude model: {self.model}")
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            self.model = model_name or "gpt-4o-mini"
            logger.info(f"EmotionDetector initialized with GPT model: {self.model}")
    
    def detect_emotion(self, message: str) -> Dict[str, any]:
        """
        Detect emotion in a message
        
        Args:
            message: Text message to analyze
        
        Returns:
            Dictionary with:
            {
                "emotion": "joy",  # one of 8 emotions
                "confidence": 0.85,  # model confidence
                "intensity": 0.7,  # emotion intensity (0-1)
                "reasoning": "The message expresses excitement..."
            }
        """
        try:
            if self.use_claude:
                return self._detect_with_claude(message)
            else:
                return self._detect_with_gpt(message)
        except Exception as e:
            logger.error(f"Emotion detection failed: {e}")
            # Fallback to neutral
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "intensity": 0.3,
                "reasoning": "Error in emotion detection, defaulting to neutral"
            }
    
    def _build_detection_prompt(self, message: str) -> str:
        """Build prompt for emotion detection with few-shot examples"""
        
        # Build few-shot examples
        examples_text = "\n".join([
            f"Message: \"{ex['message']}\"\nEmotion: {ex['emotion']}, Intensity: {ex['intensity']}"
            for ex in EMOTION_DETECTION_EXAMPLES
        ])
        
        prompt = f"""You are an expert emotion classifier for dating app conversations.

Classify the following message into ONE of these 8 emotions:
{", ".join(EMOTIONS)}

Also rate the emotion intensity (0.0 to 1.0, where 0 is barely present and 1.0 is extremely strong).

Here are some examples:

{examples_text}

Now classify this message:
Message: "{message}"

Respond with ONLY a JSON object in this exact format (no other text):
{{
    "emotion": "one of the 8 emotions",
    "confidence": 0.0-1.0,
    "intensity": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""
        
        return prompt
    
    def _detect_with_claude(self, message: str) -> Dict[str, any]:
        """Detect emotion using Claude API"""
        
        prompt = self._build_detection_prompt(message)
        
        response = self.client.messages.create(
            model=self.model,
            max_tokens=200,
            temperature=self.temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        response_text = response.content[0].text.strip()
        
        # Parse JSON response
        result = self._parse_emotion_response(response_text)
        logger.debug(f"Detected emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        
        return result
    
    def _detect_with_gpt(self, message: str) -> Dict[str, any]:
        """Detect emotion using OpenAI GPT API"""
        
        prompt = self._build_detection_prompt(message)
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=200
        )
        
        response_text = response.choices[0].message.content.strip()
        
        # Parse JSON response
        result = self._parse_emotion_response(response_text)
        logger.debug(f"Detected emotion: {result['emotion']} (confidence: {result['confidence']:.2f})")
        
        return result
    
    def _parse_emotion_response(self, response_text: str) -> Dict[str, any]:
        """Parse LLM response into structured emotion result"""
        try:
            # Try to extract JSON from response
            # Sometimes LLM adds markdown code blocks
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Validate emotion
            if result.get("emotion") not in EMOTIONS:
                logger.warning(f"Invalid emotion '{result.get('emotion')}', defaulting to neutral")
                result["emotion"] = "neutral"
            
            # Ensure all fields exist
            result.setdefault("confidence", 0.7)
            result.setdefault("intensity", 0.5)
            result.setdefault("reasoning", "")
            
            # Clamp values to [0, 1]
            result["confidence"] = max(0.0, min(1.0, result["confidence"]))
            result["intensity"] = max(0.0, min(1.0, result["intensity"]))
            
            return result
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse emotion response: {e}")
            logger.debug(f"Response text: {response_text}")
            
            # Fallback: try to extract emotion keyword
            response_lower = response_text.lower()
            for emotion in EMOTIONS:
                if emotion in response_lower:
                    return {
                        "emotion": emotion,
                        "confidence": 0.6,
                        "intensity": 0.5,
                        "reasoning": "Parsed from unstructured response"
                    }
            
            # Ultimate fallback
            return {
                "emotion": "neutral",
                "confidence": 0.5,
                "intensity": 0.3,
                "reasoning": "Failed to parse response"
            }


class EmotionAgent:
    """
    Main Emotion Agent that combines detection and prediction
    Analyzes conversation emotions and suggests reply strategies
    """
    
    def __init__(
        self,
        use_claude: bool = True,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        history_size: int = 20
    ):
        """
        Initialize Emotion Agent
        
        Args:
            use_claude: Use Claude API vs OpenAI
            model_name: Override default model
            temperature: LLM temperature for detection
            history_size: Number of recent emotions to track
        """
        self.detector = EmotionDetector(
            use_claude=use_claude,
            model_name=model_name,
            temperature=temperature
        )
        self.predictor = EmotionPredictor(history_window=5)
        self.history_size = history_size
        
        # Emotion history tracking (per conversation)
        self.emotion_history: List[Dict[str, any]] = []
        
        logger.info(f"EmotionAgent initialized (history_size={history_size})")
    
    def analyze_message(
        self, 
        message: str,
        track_history: bool = True
    ) -> Dict[str, any]:
        """
        Analyze a single message for emotion
        
        Args:
            message: Message text
            track_history: Whether to add to emotion history
        
        Returns:
            Emotion detection result with prediction
        """
        # Detect emotion
        detection = self.detector.detect_emotion(message)
        
        # Add to history
        if track_history:
            self.emotion_history.append({
                "message": message,
                "emotion": detection["emotion"],
                "confidence": detection["confidence"],
                "intensity": detection["intensity"]
            })
            
            # Keep only recent history
            if len(self.emotion_history) > self.history_size:
                self.emotion_history = self.emotion_history[-self.history_size:]
        
        # Predict next emotion
        emotion_sequence = [item["emotion"] for item in self.emotion_history]
        prediction = self.predictor.predict_next_emotion(emotion_sequence)
        
        result = {
            "current_emotion": detection,
            "predicted_next": prediction,
            "reply_strategy": self.suggest_reply_strategy(detection["emotion"])
        }
        
        return result
    
    def analyze_conversation(
        self, 
        messages: List[str]
    ) -> Dict[str, any]:
        """
        Analyze a full conversation history
        
        Args:
            messages: List of messages in chronological order
        
        Returns:
            Comprehensive emotion analysis including trends
        """
        # Reset and analyze all messages
        self.emotion_history = []
        
        for message in messages:
            self.detector.detect_emotion(message)
            # Track in history (simplified - just emotion for trend analysis)
        
        # Analyze each message
        emotions_detected = []
        for message in messages:
            detection = self.detector.detect_emotion(message)
            emotions_detected.append({
                "message": message,
                "emotion": detection["emotion"],
                "confidence": detection["confidence"],
                "intensity": detection["intensity"]
            })
        
        # Update history
        self.emotion_history = emotions_detected[-self.history_size:]
        
        # Get emotion sequence
        emotion_sequence = [item["emotion"] for item in emotions_detected]
        
        # Analyze trend
        trend_analysis = self.predictor.analyze_emotion_trend(emotion_sequence)
        momentum = self.predictor.get_emotion_momentum(emotion_sequence)
        
        # Predict next
        prediction = self.predictor.predict_next_emotion(emotion_sequence)
        
        return {
            "messages_analyzed": len(messages),
            "emotions": emotions_detected,
            "trend": trend_analysis,
            "momentum": momentum,
            "predicted_next": prediction,
            "overall_strategy": self._suggest_overall_strategy(trend_analysis, momentum)
        }
    
    def suggest_reply_strategy(
        self, 
        current_emotion: str
    ) -> Dict[str, any]:
        """
        Suggest how to reply based on detected emotion
        
        Args:
            current_emotion: The detected emotion
        
        Returns:
            Strategy recommendation
        """
        strategies = {
            "joy": {
                "approach": "共鸣 (Resonate)",
                "tone": "enthusiastic and positive",
                "suggestions": [
                    "Match their energy and enthusiasm",
                    "Share in their excitement",
                    "Use emojis to show you're equally happy",
                    "Build on the positive momentum"
                ]
            },
            "love": {
                "approach": "温柔回应 (Gentle reciprocation)",
                "tone": "warm and affectionate",
                "suggestions": [
                    "Respond with warmth and care",
                    "Show appreciation for their feelings",
                    "Be genuine and heartfelt",
                    "Take things at a comfortable pace"
                ]
            },
            "sadness": {
                "approach": "安慰 (Comfort)",
                "tone": "empathetic and supportive",
                "suggestions": [
                    "Acknowledge their feelings",
                    "Offer emotional support",
                    "Be a good listener",
                    "Avoid being overly cheerful or dismissive"
                ]
            },
            "anger": {
                "approach": "安抚 (Soothe)",
                "tone": "calm and understanding",
                "suggestions": [
                    "Stay calm and non-defensive",
                    "Validate their feelings",
                    "Apologize if appropriate",
                    "Give them space if needed"
                ]
            },
            "fear": {
                "approach": "保证 (Reassure)",
                "tone": "reassuring and patient",
                "suggestions": [
                    "Provide reassurance",
                    "Be patient and understanding",
                    "Address their concerns directly",
                    "Build trust through consistency"
                ]
            },
            "surprise": {
                "approach": "解释 (Explain)",
                "tone": "clear and friendly",
                "suggestions": [
                    "Acknowledge the unexpected element",
                    "Clarify if needed",
                    "Keep the conversation engaging",
                    "Use it as a conversation opener"
                ]
            },
            "disgust": {
                "approach": "理解 (Understand)",
                "tone": "respectful and accommodating",
                "suggestions": [
                    "Respect their boundaries",
                    "Avoid the topic that caused disgust",
                    "Find common ground elsewhere",
                    "Don't take it personally"
                ]
            },
            "neutral": {
                "approach": "自然互动 (Natural interaction)",
                "tone": "casual and friendly",
                "suggestions": [
                    "Keep conversation flowing naturally",
                    "Ask engaging questions",
                    "Share interesting topics",
                    "Build connection gradually"
                ]
            }
        }
        
        return strategies.get(current_emotion, strategies["neutral"])
    
    def _suggest_overall_strategy(
        self, 
        trend_analysis: Dict[str, any],
        momentum: str
    ) -> Dict[str, any]:
        """Suggest overall conversation strategy based on trend"""
        
        trend = trend_analysis["trend"]
        volatility = trend_analysis["volatility"]
        
        if trend == "positive":
            strategy = "保持积极 (Maintain positivity)"
            advice = "The conversation is going well. Keep up the positive energy and deepen the connection."
        elif trend == "negative":
            strategy = "改善氛围 (Improve atmosphere)"
            advice = "The conversation mood is declining. Try to address concerns, show empathy, or introduce positive topics."
        elif trend == "volatile":
            strategy = "稳定情绪 (Stabilize emotions)"
            advice = "Emotions are fluctuating. Try to bring stability and consistency to the conversation."
        else:  # neutral
            strategy = "增加参与 (Increase engagement)"
            advice = "The conversation is neutral. Try to add more engaging topics or show more personality."
        
        # Momentum consideration
        if momentum == "improving":
            advice += " The emotional tone is improving - good sign!"
        elif momentum == "declining":
            advice += " Watch out: the emotional tone is declining."
        
        # Volatility consideration
        if volatility > 0.7:
            advice += " High emotional volatility detected - tread carefully."
        
        return {
            "strategy": strategy,
            "advice": advice,
            "trend": trend,
            "momentum": momentum,
            "volatility": volatility
        }
    
    def get_emotion_history(self) -> List[Dict[str, any]]:
        """Get current emotion history"""
        return self.emotion_history.copy()
    
    def clear_history(self):
        """Clear emotion history"""
        self.emotion_history = []
        logger.debug("Emotion history cleared")
    
    def export_analysis(self) -> Dict[str, any]:
        """Export full emotion analysis for debugging/logging"""
        emotion_sequence = [item["emotion"] for item in self.emotion_history]
        
        return {
            "history_size": len(self.emotion_history),
            "emotion_history": self.emotion_history,
            "trend_analysis": self.predictor.analyze_emotion_trend(emotion_sequence),
            "momentum": self.predictor.get_emotion_momentum(emotion_sequence),
            "predicted_next": self.predictor.predict_next_emotion(emotion_sequence)
        }
