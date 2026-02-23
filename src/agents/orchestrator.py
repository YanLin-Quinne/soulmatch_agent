"""Orchestrator Agent — coordinates all sub-agents with shared context and parallel execution."""

import asyncio
import random
from typing import Dict, List, Optional, Any
from loguru import logger

from src.agents.state_machine import ConversationStateMachine, ConversationState
from src.agents.persona_agent import PersonaAgent, PersonaAgentPool
from src.agents.feature_prediction_agent import FeaturePredictionAgent
from src.agents.emotion_agent import EmotionAgent
from src.agents.scam_detection_agent import ScamDetectionAgent
from src.agents.question_strategy_agent import QuestionStrategyAgent
from src.agents.discussion import DiscussionEngine
from src.agents.skills import skill_registry, register_builtin_skills
from src.agents.agent_context import AgentContext
from src.memory.memory_manager import MemoryManager
from src.matching.matching_engine import MatchingEngine
from src.data.schema import PersonaProfile
# v2.0: New agents
from src.agents.relationship_prediction_agent import RelationshipPredictionAgent
from src.agents.feature_transition_predictor import FeatureTransitionPredictor
from src.agents.milestone_evaluator import MilestoneEvaluator


class OrchestratorAgent:
    """
    Main orchestrator that coordinates all sub-agents via shared AgentContext.

    Pipeline per turn:
      1. [parallel] Emotion + Scam + Memory retrieval
      2. [sequential] Feature update (needs emotion)
      3. [sequential] Question strategy (needs features)
      4. [sequential] Bot response (needs everything)
      5. [sequential] Memory store
    """

    def __init__(self, user_id: str, bot_personas_pool: PersonaAgentPool, bot_id: Optional[str] = None):
        self.user_id = user_id
        self.preferred_bot_id = bot_id  # Store preferred bot_id if provided

        self.state_machine = ConversationStateMachine(user_id)

        self.bot_pool = bot_personas_pool
        self.current_bot: Optional[PersonaAgent] = None

        self.feature_agent = FeaturePredictionAgent(user_id)
        self.memory_manager = MemoryManager(user_id, use_three_layer=True)
        self.emotion_agent = EmotionAgent()
        self.scam_agent = ScamDetectionAgent(use_semantic=True)
        self.question_agent = QuestionStrategyAgent()
        self.discussion_engine = DiscussionEngine(min_turns_to_trigger=3)
        self.matching_engine = MatchingEngine()

        # v2.0: New agents
        from src.agents.llm_router import router
        self.relationship_agent = RelationshipPredictionAgent(router)
        self.feature_transition_predictor = FeatureTransitionPredictor()
        self.milestone_evaluator = MilestoneEvaluator()

        register_builtin_skills()

        self.ctx = AgentContext(user_id=user_id)
        self._pending_relationship_turn = False

        logger.info(f"Orchestrator initialized for user {user_id}")

    # ------------------------------------------------------------------
    # Start conversation
    # ------------------------------------------------------------------

    def start_new_conversation(self, bot_id: Optional[str] = None) -> Dict[str, Any]:
        # Reset context and state to prevent history leaking across bot switches
        self.ctx = AgentContext(user_id=self.user_id)
        self.state_machine.reset()
        self.current_bot = None
        self.memory_manager = MemoryManager(self.user_id, use_three_layer=True)

        bot_summaries = self.bot_pool.get_agent_summaries()
        if not bot_summaries:
            return {"success": False, "error": "No bots available"}

        # Use bot_id parameter first, then fall back to self.preferred_bot_id
        preferred_bot_id = bot_id or self.preferred_bot_id

        # If preferred_bot_id is provided, use it directly
        if preferred_bot_id:
            if preferred_bot_id in bot_summaries:
                selected_bot_id = preferred_bot_id
                compatibility_score = 0.85  # High score for user-selected bot
                match_explanation = "You selected this match!"
                logger.info(f"Using user-selected bot: {selected_bot_id}")
            else:
                logger.warning(f"Preferred bot_id {preferred_bot_id} not found, falling back to random")
                selected_bot_id = random.choice(list(bot_summaries.keys()))
                compatibility_score = 0.5
                match_explanation = "Let's see if you two click!"
        else:
            # Original logic: use matching engine or random selection
            user_features = self.feature_agent.get_feature_summary()

            if not user_features.get("personality"):
                selected_bot_id = random.choice(list(bot_summaries.keys()))
                compatibility_score = 0.5
                match_explanation = "Let's see if you two click!"
            else:
                user_vector = self._convert_features_to_vector(self.feature_agent.predicted_features)
                bot_personas = [self.bot_pool.get_agent(bid).persona for bid in bot_summaries]
                recs = self.matching_engine.recommend_top_n(user_features=user_vector, candidates=bot_personas, n=1)
                if recs:
                    selected_bot_id = recs[0]["profile_id"]
                    compatibility_score = recs[0]["score"]
                    match_explanation = recs[0]["explanation"]
                else:
                    selected_bot_id = list(bot_summaries.keys())[0]
                    compatibility_score = 0.5
                    match_explanation = "Let's give this a try!"

        self.current_bot = self.bot_pool.get_agent(selected_bot_id)
        self.ctx.bot_id = selected_bot_id
        self.state_machine.start_conversation(selected_bot_id, compatibility_score)

        greeting = self.current_bot.generate_greeting()
        self.ctx.add_to_history("bot", greeting)

        logger.info(f"Started conversation with {selected_bot_id}, compat={compatibility_score:.2f}")
        return {
            "success": True,
            "bot_id": selected_bot_id,
            "bot_profile": bot_summaries[selected_bot_id],
            "compatibility_score": compatibility_score,
            "match_explanation": match_explanation,
            "greeting": greeting,
        }

    # ------------------------------------------------------------------
    # Process user message
    # ------------------------------------------------------------------

    async def process_user_message(self, message: str) -> Dict[str, Any]:
        if not self.current_bot:
            return {"success": False, "error": "No active conversation. Call start_new_conversation() first."}

        # Update context
        self.ctx.user_message = message
        self.ctx.add_to_history("user", message)
        self.ctx.turn_count = self.state_machine.context.turn_count + 1
        self.ctx.current_state = self.state_machine.context.current_state
        self.memory_manager.add_conversation_turn("user", message)

        actions = self.state_machine.handle_user_message()
        logger.info(f"Turn {self.ctx.turn_count}, actions: {actions}")

        response: Dict[str, Any] = {"success": True, "turn": self.ctx.turn_count}

        # --- Phase 1: parallel independent analyses ---
        try:
            await self._run_parallel_analyses(message, actions)
        except Exception as e:
            logger.error(f"[Orchestrator] Parallel analyses failed: {e}")

        # --- Phase 2: feature update (benefits from emotion) ---
        if "feature_update" in actions:
            try:
                result = self._update_features()
                response["feature_update"] = result
                self.state_machine.handle_feature_updated()
            except Exception as e:
                logger.error(f"[Orchestrator] Feature update failed: {e}")

            # v2.0: Feature transition prediction (every 3 turns)
            if self.ctx.turn_count % 3 == 0:
                try:
                    emotion_trend = self._compute_emotion_trend()
                    transition_pred = self.feature_transition_predictor.predict_next(
                        current_features=self.ctx.predicted_features,
                        emotion_trend=emotion_trend,
                        relationship_status=self.ctx.rel_status,
                        memory_trigger=("memory_update" in actions),
                    )
                    self.ctx.predicted_feature_changes = transition_pred
                    response["feature_transition"] = {
                        "likely_to_change": transition_pred["likely_to_change"],
                        "change_probability": transition_pred["change_probability"],
                    }
                except Exception as e:
                    logger.error(f"[Orchestrator] Feature transition prediction failed: {e}")

        # --- Phase 2.5: Relationship prediction (deferred to background) ---
        self._pending_relationship_turn = (
            self.ctx.turn_count % 5 == 0 or self.ctx.turn_count in [10, 30]
        )

        # --- Phase 3: question strategy (needs features) ---
        try:
            self._update_question_strategy()
        except Exception as e:
            logger.error(f"[Orchestrator] Question strategy update failed: {e}")

        # --- Phase 3.3: skill matching ---
        try:
            skill_results = skill_registry.execute_matched(message, self.ctx)
            if skill_results:
                self.ctx.active_skills = [r.skill_name for r in skill_results]
                self.ctx.skill_prompt_additions = [r.prompt_addition for r in skill_results if r.prompt_addition]
                response["skills"] = self.ctx.active_skills
        except Exception as e:
            logger.error(f"[Orchestrator] Skill matching failed: {e}")

        # --- Phase 3.5: dynamic discussion (needs all context) ---
        try:
            if self.discussion_engine.should_trigger(self.ctx):
                disc = self.discussion_engine.run_discussion(self.ctx)
                self.ctx.discussion_synthesis = disc.synthesis
                response["discussion"] = {
                    "triggered": True,
                    "perspectives": len(disc.perspectives),
                    "synthesis_preview": disc.synthesis[:120] + "..." if len(disc.synthesis) > 120 else disc.synthesis,
                }
        except Exception as e:
            logger.error(f"[Orchestrator] Discussion engine failed: {e}")

        # --- Phase 4: bot response (needs everything) ---
        if "bot_response" in actions:
            try:
                bot_text = self._generate_bot_response()
                response["bot_message"] = bot_text
                self.ctx.add_to_history("bot", bot_text)
                self.memory_manager.add_conversation_turn("bot", bot_text)
                self.state_machine.handle_bot_message()

                # Conversational pacing: minimal delay for demo responsiveness
                char_count = len(bot_text)
                base_delay = min(char_count * 0.005, 0.5)  # ~5ms per char, cap 0.5s
                self.ctx.reply_delay_seconds = round(base_delay, 1)
                response["reply_delay"] = self.ctx.reply_delay_seconds
            except Exception as e:
                logger.error(f"[Orchestrator] Bot response generation failed: {e}")
                response["bot_message"] = "Sorry, I'm having a moment. Could you say that again?"
                response["error_hint"] = "bot_response_failed"

        # --- Phase 5: memory store ---
        if "memory_update" in actions:
            try:
                mem_result = self._update_memory()
                response["memory_update"] = mem_result
                self.state_machine.handle_memory_updated()
            except Exception as e:
                logger.error(f"[Orchestrator] Memory update failed: {e}")

        # Always include memory stats for frontend visibility
        try:
            response["memory_stats"] = self.memory_manager.get_memory_stats()
        except Exception:
            pass

        # Scam / emotion in response
        if self.ctx.scam_warning_level != "none":
            response["scam_detection"] = {
                "risk_score": self.ctx.scam_risk_score,
                "warning_level": self.ctx.scam_warning_level,
            }
        if self.ctx.current_emotion:
            response["emotion"] = {
                "current_emotion": {
                    "emotion": self.ctx.current_emotion,
                    "confidence": self.ctx.emotion_confidence,
                    "intensity": self.ctx.emotion_intensity,
                },
            }

        response["context"] = {
            "state": self.state_machine.context.current_state,
            "turn_count": self.ctx.turn_count,
            "risk_level": self.state_machine.context.current_risk_level,
            "user_emotion": self.ctx.current_emotion,
            "avg_feature_confidence": self.feature_agent._compute_overall_confidence(),
        }
        return response

    async def run_relationship_prediction(self) -> Dict[str, Any]:
        """Run relationship prediction + milestone as background task."""
        result = {}
        if not self._pending_relationship_turn:
            return result

        try:
            rel_result = await self.relationship_agent.execute(self.ctx)
            if rel_result:
                # Serialize social consensus votes for frontend
                social_votes = []
                consensus = rel_result.get("social_consensus")
                if consensus and hasattr(consensus, "votes"):
                    for v in consensus.votes:
                        social_votes.append({
                            "agent": v.agent_name,
                            "vote": v.vote,
                            "rel_status": v.rel_status,
                            "confidence": round(v.confidence, 2),
                            "reasoning": v.reasoning,
                            "key_factors": getattr(v, 'key_factors', []),
                            "demographics": getattr(v, 'agent_demographics', {}),
                        })

                result["relationship_prediction"] = {
                    "rel_status": rel_result.get("rel_status"),
                    "rel_type": rel_result.get("rel_type"),
                    "sentiment": rel_result.get("sentiment"),
                    "can_advance": rel_result.get("can_advance"),
                    "advance_prediction_set": rel_result.get("advance_prediction_set"),
                    "social_votes": social_votes,
                    "vote_distribution": getattr(consensus, "vote_distribution", {}) if consensus else {},
                }
                self.ctx.extended_features["trust_score"] = self.ctx.extended_features.get("trust_score", 0.5)
                self.ctx.extended_features["relationship_status"] = rel_result.get("rel_status")
                self.ctx.extended_features["sentiment_label"] = rel_result.get("sentiment")
        except Exception as e:
            logger.error(f"[Orchestrator] Relationship prediction failed at turn {self.ctx.turn_count}: {e}")

        if self.ctx.turn_count in [10, 30]:
            try:
                milestone_report = self.milestone_evaluator.evaluate(
                    turn=self.ctx.turn_count,
                    feature_history=self.ctx.feature_history,
                    relationship_snapshots=self.ctx.relationship_snapshots,
                    current_features=self.ctx.predicted_features,
                )
                if milestone_report:
                    self.ctx.milestone_reports[self.ctx.turn_count] = milestone_report
                    result["milestone_report"] = milestone_report
            except Exception as e:
                logger.error(f"[Orchestrator] Milestone evaluation failed at turn {self.ctx.turn_count}: {e}")

        self._pending_relationship_turn = False
        return result

    # ------------------------------------------------------------------
    # Pipeline helpers
    # ------------------------------------------------------------------

    async def _run_parallel_analyses(self, message: str, actions: list):
        """Run emotion, scam, and memory retrieval in parallel via asyncio."""

        async def _emotion():
            if "emotion_analysis" not in actions:
                return
            emo = await asyncio.to_thread(self.emotion_agent.analyze_message, message)
            if emo.get("current_emotion"):
                ce = emo["current_emotion"]
                self.ctx.current_emotion = ce["emotion"]
                self.ctx.emotion_confidence = ce["confidence"]
                self.ctx.emotion_intensity = ce["intensity"]
                self.ctx.emotion_history.append(ce["emotion"])
                self.state_machine.context.current_user_emotion = ce["emotion"]
                self.state_machine.context.add_emotion(ce["emotion"])
            if emo.get("reply_strategy"):
                strategy = emo["reply_strategy"]
                self.ctx.reply_strategy = strategy.get("approach", "")

        async def _scam():
            if "scam_check" not in actions:
                return
            scam = await asyncio.to_thread(self.scam_agent.analyze_message, message)
            self.ctx.scam_risk_score = scam.get("risk_score", 0.0)
            self.ctx.scam_warning_level = scam.get("warning_level", "none")
            self.ctx.scam_warning_message = str(scam.get("message", {}).get("en", ""))
            if scam["warning_level"] in ("high", "critical"):
                self.state_machine.handle_scam_detected(scam["warning_level"])

        async def _memory():
            memories = await asyncio.to_thread(self.memory_manager.retrieve_relevant_memories, message)
            self.ctx.retrieved_memories = memories

        await asyncio.gather(_emotion(), _scam(), _memory(), return_exceptions=True)

    def _update_features(self) -> Dict[str, Any]:
        result = self.feature_agent.predict_from_conversation(self.ctx.conversation_history)
        self.ctx.predicted_features = result.get("features", {})
        self.ctx.feature_confidences = result.get("confidences", {})
        self.ctx.low_confidence_features = result.get("low_confidence", [])
        return result

    def _update_question_strategy(self):
        low = self.ctx.low_confidence_features
        if low:
            hints = self.question_agent.suggest_hints(low, self.ctx.conversation_history)
            self.ctx.suggested_probes = [h["text"] for h in hints]
            self.ctx.suggested_hints = hints

    def _generate_bot_response(self) -> str:
        return self.current_bot.generate_response(message=self.ctx.user_message, ctx=self.ctx)

    def _update_memory(self) -> Dict[str, Any]:
        recent = self.ctx.recent_history(5)
        action = self.memory_manager.decide_memory_action(recent, self.feature_agent.predicted_features)
        memories = self.memory_manager.execute_action(action)
        return {
            "operation": action.operation,
            "reasoning": action.reasoning,
            "memories_affected": len(memories) if memories else 0,
        }

    def _compute_emotion_trend(self) -> str:
        """v2.0: 计算情绪趋势(improving/declining/stable)"""
        if len(self.ctx.emotion_history) < 3:
            return "stable"

        # 简化映射: positive情绪→+1, negative→-1
        valence_map = {
            "joy": 1.0, "excitement": 0.8, "interest": 0.6, "trust": 0.7,
            "sadness": -0.7, "anger": -0.9, "fear": -0.6, "disgust": -0.8,
            "neutral": 0.0, "surprise": 0.3,
        }

        recent_3 = self.ctx.emotion_history[-3:]
        valences = [valence_map.get(e, 0.0) for e in recent_3]

        # 简单线性趋势
        if valences[-1] > valences[0] + 0.2:
            return "improving"
        elif valences[-1] < valences[0] - 0.2:
            return "declining"
        else:
            return "stable"

    # ------------------------------------------------------------------

    def _convert_features_to_vector(self, features: Dict[str, Any]) -> Optional[list[float]]:
        import numpy as np
        vector = []
        for trait in ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"):
            vector.append(features.get(f"big_five_{trait}", 0.5))
        for style in ("direct", "indirect", "humorous", "serious", "casual", "formal"):
            vector.append(1.0 if features.get("communication_style") == style else 0.0)
        for interest in ("music", "sports", "travel", "food", "arts", "tech", "outdoors", "books"):
            vector.append(features.get(f"interest_{interest}", 0.0))
        for goal in ("casual", "serious", "friendship", "unsure"):
            vector.append(1.0 if features.get("relationship_goals") == goal else 0.0)
        arr = np.array(vector)
        norm = np.linalg.norm(arr)
        if norm > 0:
            arr = arr / norm
        return arr.tolist() if len(vector) == 23 else None

    def get_conversation_summary(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "bot_id": self.ctx.bot_id,
            "state": self.state_machine.context.current_state,
            "turn_count": self.ctx.turn_count,
            "message_count": len(self.ctx.conversation_history),
            "compatibility_score": self.state_machine.context.compatibility_score,
            "risk_level": self.state_machine.context.current_risk_level,
            "user_features": self.feature_agent.get_feature_summary(),
            "emotion_history": self.ctx.emotion_history,
        }

    def reset_conversation(self):
        self.state_machine.reset()
        self.ctx = AgentContext(user_id=self.user_id)
        self.current_bot = None
        self.memory_manager = MemoryManager(self.user_id)
        logger.info(f"Conversation reset for user {self.user_id}")
