"""Scam Detection Agent for romance fraud (杀猪盘) detection"""

import re
import json
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
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
from src.agents.scam_patterns import (
    ScamPattern,
    PATTERN_RULES,
    RISK_THRESHOLDS,
    COMPOSITE_PATTERNS,
    WARNING_MESSAGES,
    get_pattern_description,
)


class ScamDetector:
    """
    Core scam detection engine using rule-based matching + LLM semantic analysis
    """
    
    def __init__(self):
        """Initialize scam detector with pattern rules"""
        self.pattern_rules = PATTERN_RULES
        self.risk_thresholds = RISK_THRESHOLDS
        logger.info("ScamDetector initialized with pattern rules")
    
    def detect_scam_signals(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Detect scam signals in a message with conversation context
        
        Args:
            message: Current message to analyze
            conversation_history: Previous messages [{"role": "user/agent", "content": "..."}]
        
        Returns:
            {
                "risk_score": 0.0-1.0,
                "detected_patterns": [ScamPattern, ...],
                "pattern_details": {pattern: {"matches": [...], "confidence": 0.x}},
                "warning_level": "low/medium/high/critical",
                "message": "warning message"
            }
        """
        detected_patterns = {}
        total_risk = 0.0
        
        # 1. Rule-based keyword matching
        for pattern, rule in self.pattern_rules.items():
            matches = self._match_keywords(message, rule.keywords)
            
            if matches:
                # Calculate pattern-specific risk
                match_ratio = len(matches) / max(len(rule.keywords), 1)
                pattern_confidence = min(match_ratio * 3.0, 1.0)  # Scale up for visibility
                pattern_risk = rule.weight * pattern_confidence
                
                detected_patterns[pattern] = {
                    "matches": matches,
                    "confidence": pattern_confidence,
                    "risk_contribution": pattern_risk
                }
                
                total_risk += pattern_risk
        
        # 2. Regex-based URL/link detection for EXTERNAL_LINKS
        url_patterns = self._detect_urls(message)
        if url_patterns:
            if ScamPattern.EXTERNAL_LINKS in detected_patterns:
                detected_patterns[ScamPattern.EXTERNAL_LINKS]["matches"].extend(url_patterns)
            else:
                detected_patterns[ScamPattern.EXTERNAL_LINKS] = {
                    "matches": url_patterns,
                    "confidence": 0.8,
                    "risk_contribution": PATTERN_RULES[ScamPattern.EXTERNAL_LINKS].weight * 0.8
                }
                total_risk += PATTERN_RULES[ScamPattern.EXTERNAL_LINKS].weight * 0.8
        
        # 3. Check for composite patterns (multi-turn analysis)
        if conversation_history:
            composite_risk = self._check_composite_patterns(
                message,
                conversation_history,
                detected_patterns
            )
            total_risk += composite_risk
        
        # 4. Normalize risk score to [0, 1]
        risk_score = min(total_risk, 1.0)
        
        # 5. Determine warning level
        warning_level = self._get_warning_level(risk_score)
        
        return {
            "risk_score": round(risk_score, 3),
            "detected_patterns": list(detected_patterns.keys()),
            "pattern_details": detected_patterns,
            "warning_level": warning_level,
            "message": WARNING_MESSAGES.get(warning_level, WARNING_MESSAGES["low"])
        }
    
    def _match_keywords(self, message: str, keywords: set) -> List[str]:
        """Match keywords in message (case-insensitive)"""
        message_lower = message.lower()
        matches = []
        
        for keyword in keywords:
            if keyword.lower() in message_lower:
                matches.append(keyword)
        
        return matches
    
    def _detect_urls(self, message: str) -> List[str]:
        """Detect URLs and suspicious links using regex"""
        url_patterns = [
            r'https?://[^\s]+',  # Standard URLs
            r'www\.[^\s]+',      # www links
            r'[a-zA-Z0-9-]+\.(com|cn|net|org|io|co)[^\s]*',  # Domains
            r'(telegram|whatsapp|line)\.me/[^\s]+',  # Messaging app links
        ]
        
        detected = []
        for pattern in url_patterns:
            matches = re.findall(pattern, message, re.IGNORECASE)
            detected.extend(matches)
        
        return detected
    
    def _check_composite_patterns(
        self,
        message: str,
        history: List[Dict[str, str]],
        current_patterns: Dict[ScamPattern, dict]
    ) -> float:
        """
        Check for dangerous pattern combinations across conversation
        Returns additional risk score
        """
        additional_risk = 0.0
        
        # Build pattern timeline
        pattern_timeline = []
        
        # Analyze history
        for i, msg in enumerate(history[-20:]):  # Last 20 messages
            msg_text = msg.get("content", "")
            for pattern, rule in self.pattern_rules.items():
                if self._match_keywords(msg_text, rule.keywords):
                    pattern_timeline.append({"turn": i, "pattern": pattern})
        
        # Add current message patterns
        current_turn = len(history)
        for pattern in current_patterns.keys():
            pattern_timeline.append({"turn": current_turn, "pattern": pattern})
        
        # Check composite patterns
        for composite_name, composite_config in COMPOSITE_PATTERNS.items():
            required_patterns = composite_config["patterns"]
            turns_threshold = composite_config["turns_threshold"]
            multiplier = composite_config["risk_multiplier"]
            
            # Find occurrences of required patterns
            pattern_turns = defaultdict(list)
            for item in pattern_timeline:
                if item["pattern"] in required_patterns:
                    pattern_turns[item["pattern"]].append(item["turn"])
            
            # Check if all required patterns appear within threshold
            if len(pattern_turns) == len(required_patterns):
                # Get turn ranges
                all_turns = []
                for turns_list in pattern_turns.values():
                    all_turns.extend(turns_list)
                
                if all_turns:
                    turn_range = max(all_turns) - min(all_turns)
                    
                    if turn_range <= turns_threshold:
                        # Composite pattern detected!
                        logger.warning(f"Composite pattern detected: {composite_name}")
                        
                        # Add bonus risk
                        base_risk = sum(
                            PATTERN_RULES[p].weight 
                            for p in required_patterns
                        )
                        additional_risk += base_risk * (multiplier - 1.0)
        
        return additional_risk
    
    def _get_warning_level(self, risk_score: float) -> str:
        """Determine warning level from risk score"""
        if risk_score >= RISK_THRESHOLDS["critical"]:
            return "critical"
        elif risk_score >= RISK_THRESHOLDS["high"]:
            return "high"
        elif risk_score >= RISK_THRESHOLDS["medium"]:
            return "medium"
        elif risk_score >= RISK_THRESHOLDS["low"]:
            return "low"
        else:
            return "safe"


class SemanticScamAnalyzer:
    """
    LLM-based semantic analysis for detecting scam patterns beyond keywords
    """
    
    def __init__(
        self,
        use_claude: bool = True,
        model_name: Optional[str] = None,
        temperature: float = 0.3
    ):
        """
        Initialize semantic analyzer with LLM
        
        Args:
            use_claude: Use Claude API vs OpenAI
            model_name: Override default model
            temperature: LLM temperature (low for consistent detection)
        """
        self.use_claude = use_claude and ANTHROPIC_AVAILABLE
        self.temperature = temperature
        
        # Initialize API client
        if self.use_claude:
            if not settings.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not configured")
            self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
            self.model = model_name or "claude-3-5-haiku-20241022"
            logger.info(f"SemanticScamAnalyzer initialized with Claude: {self.model}")
        else:
            if not OPENAI_AVAILABLE:
                raise ImportError("openai package not installed")
            if not settings.openai_api_key:
                raise ValueError("OPENAI_API_KEY not configured")
            self.client = openai.OpenAI(api_key=settings.openai_api_key)
            self.model = model_name or "gpt-4o-mini"
            logger.info(f"SemanticScamAnalyzer initialized with GPT: {self.model}")
    
    def analyze_semantic_risk(
        self,
        message: str,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, any]:
        """
        Analyze message for semantic scam signals using LLM
        
        Returns:
            {
                "semantic_risk": 0.0-1.0,
                "detected_tactics": ["love_bombing", "urgency", ...],
                "reasoning": "explanation",
                "red_flags": ["具体的危险信号"]
            }
        """
        prompt = self._build_semantic_prompt(message, conversation_history)
        
        try:
            if self.use_claude:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=500,
                    temperature=self.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                response_text = response.content[0].text.strip()
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.temperature,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content.strip()
            
            # Parse response
            result = self._parse_semantic_response(response_text)
            logger.debug(f"Semantic risk: {result['semantic_risk']:.2f}")
            
            return result
            
        except Exception as e:
            logger.error(f"Semantic analysis failed: {e}")
            return {
                "semantic_risk": 0.0,
                "detected_tactics": [],
                "reasoning": "Analysis failed",
                "red_flags": []
            }
    
    def _build_semantic_prompt(
        self,
        message: str,
        history: Optional[List[Dict[str, str]]]
    ) -> str:
        """Build prompt for semantic scam detection"""
        
        # Build context from history
        context = ""
        if history and len(history) > 0:
            recent_messages = history[-5:]  # Last 5 messages
            context = "\n".join([
                f"{msg['role']}: {msg['content']}"
                for msg in recent_messages
            ])
            context = f"\n\nConversation History:\n{context}\n"
        
        prompt = f"""You are an expert at detecting romance scams (杀猪盘) in dating conversations.

Analyze this message for scam signals and manipulation tactics:

{context}
Current Message: "{message}"

Evaluate for these scam tactics:
1. **Love Bombing**: Excessive affection too early (e.g., "soulmate" after 1 day)
2. **Financial Manipulation**: Any mention of money, investment, gifts, or transfers
3. **Urgency Tactics**: Pressure to act quickly ("right now", "limited time")
4. **External Platform**: Trying to move conversation elsewhere (Telegram, WhatsApp, links)
5. **Too Good to Be True**: Unrealistic promises or claims
6. **Emotional Manipulation**: Creating dependency, guilt, or fear

Respond with ONLY a JSON object (no other text):
{{
    "semantic_risk": 0.0-1.0,
    "detected_tactics": ["love_bombing", "urgency", ...],
    "reasoning": "brief explanation of why this is/isn't suspicious",
    "red_flags": ["specific concerning phrases or behaviors"]
}}

Be especially alert for:
- Money requests disguised as emergencies
- Investment/crypto opportunities
- Moving too fast emotionally
- Inconsistent personal details
- Avoiding video calls or meetings
"""
        
        return prompt
    
    def _parse_semantic_response(self, response_text: str) -> Dict[str, any]:
        """Parse LLM response into structured result"""
        try:
            # Clean response
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Validate and set defaults
            result.setdefault("semantic_risk", 0.0)
            result.setdefault("detected_tactics", [])
            result.setdefault("reasoning", "")
            result.setdefault("red_flags", [])
            
            # Clamp risk to [0, 1]
            result["semantic_risk"] = max(0.0, min(1.0, result["semantic_risk"]))
            
            return result
            
        except (json.JSONDecodeError, KeyError) as e:
            logger.error(f"Failed to parse semantic response: {e}")
            logger.debug(f"Response: {response_text}")
            
            # Fallback
            return {
                "semantic_risk": 0.0,
                "detected_tactics": [],
                "reasoning": "Failed to parse LLM response",
                "red_flags": []
            }


class ScamDetectionAgent:
    """
    Main Scam Detection Agent combining rule-based and semantic analysis
    """
    
    def __init__(
        self,
        use_semantic: bool = True,
        use_claude: bool = True,
        model_name: Optional[str] = None,
        history_size: int = 50
    ):
        """
        Initialize Scam Detection Agent
        
        Args:
            use_semantic: Enable LLM semantic analysis (recommended)
            use_claude: Use Claude vs OpenAI for semantic analysis
            model_name: Override default LLM model
            history_size: Max conversation history to track
        """
        # Core detector (always enabled)
        self.detector = ScamDetector()
        
        # Semantic analyzer (optional, requires API key)
        self.use_semantic = use_semantic
        self.semantic_analyzer = None
        
        if use_semantic:
            try:
                self.semantic_analyzer = SemanticScamAnalyzer(
                    use_claude=use_claude,
                    model_name=model_name
                )
                logger.info("Semantic analysis enabled")
            except Exception as e:
                logger.warning(f"Semantic analysis disabled: {e}")
                self.use_semantic = False
        
        # Conversation tracking
        self.history_size = history_size
        self.conversation_history: List[Dict[str, str]] = []
        self.detected_signals_timeline: List[Dict[str, any]] = []
        
        logger.info(f"ScamDetectionAgent initialized (semantic={use_semantic})")
    
    def analyze_message(
        self,
        message: str,
        track_history: bool = True
    ) -> Dict[str, any]:
        """
        Analyze a single message for scam signals
        
        Args:
            message: Message to analyze
            track_history: Add to conversation history
        
        Returns:
            Complete scam analysis result
        """
        # 1. Rule-based detection
        rule_result = self.detector.detect_scam_signals(
            message,
            self.conversation_history
        )
        
        # 2. Semantic analysis (if enabled)
        semantic_result = None
        if self.use_semantic and self.semantic_analyzer:
            semantic_result = self.semantic_analyzer.analyze_semantic_risk(
                message,
                self.conversation_history
            )
        
        # 3. Combine results
        combined_result = self._combine_results(rule_result, semantic_result)
        
        # 4. Track in history
        if track_history:
            self.conversation_history.append({
                "role": "user",
                "content": message
            })
            
            if len(self.conversation_history) > self.history_size:
                self.conversation_history = self.conversation_history[-self.history_size:]
            
            # Track detection
            if combined_result["risk_score"] >= RISK_THRESHOLDS["low"]:
                self.detected_signals_timeline.append({
                    "turn": len(self.conversation_history),
                    "message": message,
                    "risk_score": combined_result["risk_score"],
                    "patterns": combined_result["detected_patterns"]
                })
        
        return combined_result
    
    def analyze_conversation(
        self,
        messages: List[str]
    ) -> Dict[str, any]:
        """
        Analyze full conversation for scam patterns
        
        Args:
            messages: List of messages in chronological order
        
        Returns:
            Comprehensive scam analysis with risk trend
        """
        # Reset and analyze
        self.conversation_history = []
        self.detected_signals_timeline = []
        
        results = []
        risk_scores = []
        all_patterns = Counter()
        
        for i, message in enumerate(messages):
            result = self.analyze_message(message, track_history=True)
            results.append(result)
            risk_scores.append(result["risk_score"])
            
            for pattern in result["detected_patterns"]:
                all_patterns[pattern] += 1
        
        # Analyze trend
        risk_trend = self._analyze_risk_trend(risk_scores)
        
        # Overall assessment
        max_risk = max(risk_scores) if risk_scores else 0.0
        avg_risk = sum(risk_scores) / len(risk_scores) if risk_scores else 0.0
        
        # Get overall warning level
        overall_level = self.detector._get_warning_level(max_risk)
        
        return {
            "messages_analyzed": len(messages),
            "max_risk_score": round(max_risk, 3),
            "average_risk_score": round(avg_risk, 3),
            "overall_warning_level": overall_level,
            "warning_message": WARNING_MESSAGES.get(overall_level),
            "risk_trend": risk_trend,
            "detected_patterns_summary": dict(all_patterns),
            "timeline": self.detected_signals_timeline,
            "detailed_results": results
        }
    
    def generate_warning(
        self,
        detected_patterns: List[ScamPattern],
        risk_score: float,
        lang: str = "zh"
    ) -> str:
        """
        Generate user-friendly warning message
        
        Args:
            detected_patterns: List of detected scam patterns
            risk_score: Overall risk score
            lang: Language ("zh" or "en")
        
        Returns:
            Formatted warning message
        """
        level = self.detector._get_warning_level(risk_score)
        
        # Base warning
        warning = WARNING_MESSAGES[level][lang]
        
        # Add pattern details
        if detected_patterns:
            warning += "\n\n检测到的可疑特征：\n" if lang == "zh" else "\n\nDetected patterns:\n"
            
            for pattern in detected_patterns:
                pattern_desc = get_pattern_description(pattern, lang)
                warning += f"• {pattern_desc}\n"
        
        # Add protective advice
        if level in ["high", "critical"]:
            advice = (
                "\n⚠️ 安全建议：\n"
                "1. 不要转账或提供财务信息\n"
                "2. 不要点击外部链接\n"
                "3. 不要转移到其他平台\n"
                "4. 考虑举报此用户"
            ) if lang == "zh" else (
                "\n⚠️ Safety Tips:\n"
                "1. Never send money or financial info\n"
                "2. Don't click external links\n"
                "3. Don't move to other platforms\n"
                "4. Consider reporting this user"
            )
            warning += advice
        
        return warning
    
    def _combine_results(
        self,
        rule_result: Dict[str, any],
        semantic_result: Optional[Dict[str, any]]
    ) -> Dict[str, any]:
        """Combine rule-based and semantic analysis results"""
        
        if not semantic_result:
            # Only rule-based
            return rule_result
        
        # Weighted combination: 60% rules, 40% semantic
        combined_risk = (
            rule_result["risk_score"] * 0.6 +
            semantic_result["semantic_risk"] * 0.4
        )
        
        # Get combined warning level
        warning_level = self.detector._get_warning_level(combined_risk)
        
        return {
            "risk_score": round(combined_risk, 3),
            "rule_risk": rule_result["risk_score"],
            "semantic_risk": semantic_result["semantic_risk"],
            "detected_patterns": rule_result["detected_patterns"],
            "pattern_details": rule_result["pattern_details"],
            "semantic_tactics": semantic_result["detected_tactics"],
            "semantic_reasoning": semantic_result["reasoning"],
            "red_flags": semantic_result["red_flags"],
            "warning_level": warning_level,
            "message": WARNING_MESSAGES.get(warning_level, WARNING_MESSAGES["low"])
        }
    
    def _analyze_risk_trend(self, risk_scores: List[float]) -> Dict[str, any]:
        """Analyze risk score trend over conversation"""
        if len(risk_scores) < 2:
            return {"trend": "insufficient_data", "direction": "unknown"}
        
        # Simple trend analysis
        first_half_avg = sum(risk_scores[:len(risk_scores)//2]) / (len(risk_scores)//2)
        second_half_avg = sum(risk_scores[len(risk_scores)//2:]) / (len(risk_scores) - len(risk_scores)//2)
        
        if second_half_avg > first_half_avg + 0.1:
            trend = "escalating"
        elif second_half_avg < first_half_avg - 0.1:
            trend = "de-escalating"
        else:
            trend = "stable"
        
        return {
            "trend": trend,
            "early_risk": round(first_half_avg, 3),
            "recent_risk": round(second_half_avg, 3),
            "peak_risk": round(max(risk_scores), 3),
            "is_escalating": trend == "escalating"
        }
    
    def get_conversation_summary(self) -> Dict[str, any]:
        """Get summary of tracked conversation"""
        if not self.conversation_history:
            return {"status": "no_conversation_tracked"}
        
        # Calculate stats
        total_signals = len(self.detected_signals_timeline)
        
        all_patterns = Counter()
        for signal in self.detected_signals_timeline:
            for pattern in signal["patterns"]:
                all_patterns[pattern] += 1
        
        return {
            "total_messages": len(self.conversation_history),
            "suspicious_messages": total_signals,
            "detected_patterns": dict(all_patterns),
            "timeline": self.detected_signals_timeline
        }
    
    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = []
        self.detected_signals_timeline = []
        logger.debug("Scam detection history cleared")
