"""Extended Ablation Study for SoulMatch.

Tests the contribution of each system component by selectively disabling them:
1. Three-Layer Memory vs Flat Storage
2. Multi-Agent Discussion vs Single Agent
3. Bayesian Update vs No Update
4. Conformal Prediction vs Raw LLM Confidence
5. Chain-of-Thought vs Direct Extraction

Usage:
    python experiments/ablation_study.py
    python experiments/ablation_study.py --configs "Full System" "w/o Bayesian Update"
    python experiments/ablation_study.py --output experiments/ablation_custom.json
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional

from loguru import logger

from src.memory.memory_manager import MemoryManager
from src.agents.llm_router import LLMRouter, AgentRole, router
from src.agents.feature_prediction_agent import FeaturePredictionAgent
from src.agents.agent_discussion_room import AgentDiscussionRoom


# =====================================================================
# Ablation configurations
# =====================================================================

ABLATION_CONFIGS: Dict[str, Dict[str, bool]] = {
    "Full System": {
        "use_three_layer_memory": True,
        "use_discussion_room": True,
        "use_bayesian": True,
        "use_conformal": True,
        "use_cot": True,
    },
    "w/o Three-Layer Memory": {
        "use_three_layer_memory": False,
        "use_discussion_room": True,
        "use_bayesian": True,
        "use_conformal": True,
        "use_cot": True,
    },
    "w/o Multi-Agent Discussion": {
        "use_three_layer_memory": True,
        "use_discussion_room": False,
        "use_bayesian": True,
        "use_conformal": True,
        "use_cot": True,
    },
    "w/o Bayesian Update": {
        "use_three_layer_memory": True,
        "use_discussion_room": True,
        "use_bayesian": False,
        "use_conformal": True,
        "use_cot": True,
    },
    "w/o Conformal Prediction": {
        "use_three_layer_memory": True,
        "use_discussion_room": True,
        "use_bayesian": True,
        "use_conformal": False,
        "use_cot": True,
    },
    "w/o CoT Reasoning": {
        "use_three_layer_memory": True,
        "use_discussion_room": True,
        "use_bayesian": True,
        "use_conformal": True,
        "use_cot": False,
    },
}


# Test conversation templates for reproducible ablation
_TEST_CONVERSATION_TURNS = [
    {"speaker": "user", "message": "Hey! I just got back from a week-long hiking trip in Patagonia. The views were unreal."},
    {"speaker": "bot", "message": "Patagonia sounds amazing! What was the highlight of the trip for you?"},
    {"speaker": "user", "message": "Definitely the sunrise at Torres del Paine. I'm a photographer so I was up before dawn every day."},
    {"speaker": "bot", "message": "A photographer who loves hiking - that's a great combination! Do you shoot professionally?"},
    {"speaker": "user", "message": "Yeah, I do landscape and travel photography freelance. Studied fine arts at NYU."},
    {"speaker": "bot", "message": "NYU is a great school for the arts. How long have you been freelancing?"},
    {"speaker": "user", "message": "About 5 years now. I'm 29. Quit my corporate marketing job to pursue it full-time."},
    {"speaker": "bot", "message": "That takes courage! Do you find freelancing gives you enough work-life balance?"},
    {"speaker": "user", "message": "It does, mostly. I'm pretty introverted so I enjoy the solo work. But sometimes it gets lonely."},
    {"speaker": "bot", "message": "I can understand that. Is that part of why you're on this app?"},
    {"speaker": "user", "message": "Honestly yes. Looking for something serious. I'm tired of casual dating."},
    {"speaker": "bot", "message": "That's a mature perspective. What qualities matter most to you in a partner?"},
    {"speaker": "user", "message": "Kindness first. Someone intellectually curious. I'm a big reader - currently into Haruki Murakami."},
    {"speaker": "bot", "message": "Murakami is wonderful! Do you have a favorite novel of his?"},
    {"speaker": "user", "message": "Norwegian Wood, easily. I love how he captures melancholy. I tend to be a bit of a deep thinker myself."},
    {"speaker": "bot", "message": "Sounds like you appreciate emotional depth. What else do you enjoy besides reading and photography?"},
    {"speaker": "user", "message": "I cook a lot - mostly vegetarian stuff. I do yoga in the mornings. And I play guitar, badly."},
    {"speaker": "bot", "message": "A creative with many interests! Do you drink or go out much?"},
    {"speaker": "user", "message": "I'll have wine socially but I rarely go to bars. Not really my scene. I prefer dinner parties."},
    {"speaker": "bot", "message": "That's lovely. It sounds like you value quality connections over quantity."},
]


class AblationStudy:
    """Extended ablation study testing contribution of each SoulMatch component."""

    def __init__(self):
        self.router = LLMRouter()
        self.configs = ABLATION_CONFIGS

    # ------------------------------------------------------------------
    # Memory ablation (original test preserved)
    # ------------------------------------------------------------------

    def run_memory_ablation(self, use_three_layer: bool) -> Dict[str, Any]:
        """Test three-layer memory vs flat storage.

        Creates a MemoryManager, feeds 50 simulated turns, and collects stats.
        This is the original ablation logic from the first version of the file.
        """
        logger.info(f"Memory ablation: use_three_layer={use_three_layer}")

        results: Dict[str, Any] = {
            "use_three_layer": use_three_layer,
            "total_turns": 0,
            "episodic_memories": 0,
            "semantic_memories": 0,
            "inconsistencies_detected": 0,
            "compression_ratio": 0.0,
        }

        manager = MemoryManager(
            user_id=f"ablation_memory_{use_three_layer}",
            use_three_layer=use_three_layer,
        )

        # Simulate 50 turns
        for i in range(50):
            manager.add_conversation_turn("user", f"Turn {i}: Test message about hobbies and personality")
            manager.add_conversation_turn("bot", f"Turn {i}: Bot response with follow-up questions")

        stats = manager.get_memory_stats()
        results["total_turns"] = stats.get("current_turn", 0)

        if use_three_layer:
            results["episodic_memories"] = stats.get("episodic_memory_count", 0)
            results["semantic_memories"] = stats.get("semantic_memory_count", 0)
            results["compression_ratio"] = stats.get("compression_ratio", 0.0)

        # Retrieval quality test
        test_queries = [
            "What are the user's hobbies?",
            "What is the user's personality like?",
            "Does the user prefer indoor or outdoor activities?",
        ]
        retrieval_results = []
        for query in test_queries:
            memories = manager.retrieve_relevant_memories(query, n=3)
            retrieval_results.append({
                "query": query,
                "n_results": len(memories),
                "total_chars": sum(len(m) for m in memories),
            })
        results["retrieval_tests"] = retrieval_results

        logger.info(f"Memory ablation done: {results}")
        return results

    # ------------------------------------------------------------------
    # Feature prediction ablation
    # ------------------------------------------------------------------

    def run_feature_prediction_ablation(
        self, config: Dict[str, bool], test_conversation: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Test feature prediction with different component configs.

        Creates a FeaturePredictionAgent with CoT and conformal toggled,
        optionally disables Bayesian updates, and measures prediction quality.
        """
        agent = FeaturePredictionAgent(
            user_id="ablation_feature_test",
            use_cot=config["use_cot"],
        )

        # Disable conformal calibrator if config says so
        if not config["use_conformal"]:
            agent.conformal = None

        results: Dict[str, Any] = {
            "config": {k: v for k, v in config.items()},
            "turns_processed": 0,
            "feature_snapshots": [],
        }

        start_time = time.time()

        # Feed conversation incrementally in 5-turn windows
        for end_idx in range(2, len(test_conversation) + 1, 2):
            window = test_conversation[:end_idx]

            # If Bayesian update disabled, reset accumulated features before each call
            # so the agent cannot build posterior incrementally
            if not config["use_bayesian"]:
                agent.predicted_features = {}
                agent.feature_confidences = {}
                agent.conversation_count = 0

            try:
                result = agent.predict_from_conversation(window)
                snapshot = {
                    "turn": end_idx,
                    "n_features": len(result.get("features", {})),
                    "avg_confidence": result.get("average_confidence", 0.0),
                    "low_confidence_count": len(result.get("low_confidence", [])),
                }
                # Record conformal stats if available
                if result.get("conformal"):
                    snapshot["conformal_avg_set_size"] = result["conformal"].get("avg_set_size", 0.0)
                    snapshot["conformal_singletons"] = result["conformal"].get("singletons", 0)
                    snapshot["conformal_total_dims"] = result["conformal"].get("total_dims", 0)

                results["feature_snapshots"].append(snapshot)
            except Exception as e:
                logger.warning(f"Feature prediction failed at turn {end_idx}: {e}")
                results["feature_snapshots"].append({
                    "turn": end_idx,
                    "error": str(e),
                })

        results["turns_processed"] = len(test_conversation)
        results["elapsed_seconds"] = round(time.time() - start_time, 2)

        # Final feature summary
        try:
            summary = agent.get_feature_summary()
            results["final_summary"] = summary
        except Exception as e:
            results["final_summary_error"] = str(e)

        # Perception accuracy (proxy, no ground truth)
        try:
            acc = agent.compute_perception_accuracy()
            results["perception_accuracy"] = acc
        except Exception as e:
            results["perception_accuracy_error"] = str(e)

        return results

    # ------------------------------------------------------------------
    # Discussion room ablation
    # ------------------------------------------------------------------

    async def run_discussion_ablation(
        self, use_discussion_room: bool, test_conversation: List[Dict[str, str]]
    ) -> Dict[str, Any]:
        """Test multi-agent discussion vs single-agent decision.

        When enabled, runs AgentDiscussionRoom with three perspective agents.
        When disabled, uses a single LLM call for the same task.
        """
        topic = "Based on the conversation, what is the most important personality trait to explore next?"
        context_text = "\n".join(
            f"{t['speaker']}: {t['message']}" for t in test_conversation[-10:]
        )
        context = {"conversation_excerpt": context_text}

        results: Dict[str, Any] = {
            "use_discussion_room": use_discussion_room,
        }

        start_time = time.time()

        if use_discussion_room:
            room = AgentDiscussionRoom(router=self.router)
            agents = [
                {
                    "name": "Personality Expert",
                    "role": AgentRole.FEATURE,
                    "expertise": "Big Five personality inference from conversation",
                    "system_prompt": "You are a personality psychology expert.",
                },
                {
                    "name": "Relationship Advisor",
                    "role": AgentRole.EMOTION,
                    "expertise": "Relationship dynamics and compatibility assessment",
                    "system_prompt": "You are a relationship counselor.",
                },
                {
                    "name": "Communication Analyst",
                    "role": AgentRole.GENERAL,
                    "expertise": "Communication style and intent analysis",
                    "system_prompt": "You are a communication style analyst.",
                },
            ]
            try:
                consensus = await room.discuss(
                    topic=topic,
                    context=context,
                    agents=agents,
                )
                results["decision"] = consensus.decision
                results["confidence"] = consensus.confidence
                results["reasoning"] = consensus.reasoning
                results["n_proposals"] = len(consensus.proposals)
                results["n_critiques"] = len(consensus.critiques)
            except Exception as e:
                logger.warning(f"Discussion room failed: {e}")
                results["error"] = str(e)
        else:
            # Single-agent fallback
            try:
                response = router.chat(
                    role=AgentRole.FEATURE,
                    system="You are a personality analysis expert. Provide a JSON response with 'decision', 'confidence' (0-1), and 'reasoning'.",
                    messages=[{
                        "role": "user",
                        "content": f"Context:\n{context_text}\n\nQuestion: {topic}",
                    }],
                    max_tokens=400,
                )
                results["decision"] = response[:200]
                results["confidence"] = 0.5  # no multi-agent calibration
                results["n_proposals"] = 1
                results["n_critiques"] = 0
            except Exception as e:
                logger.warning(f"Single-agent call failed: {e}")
                results["error"] = str(e)

        results["elapsed_seconds"] = round(time.time() - start_time, 2)
        return results

    # ------------------------------------------------------------------
    # Full ablation runner
    # ------------------------------------------------------------------

    async def run_full_ablation(self, configs: Optional[List[str]] = None) -> Dict[str, Any]:
        """Run all ablation experiments across specified configurations.

        Args:
            configs: List of config names to run. None means run all.

        Returns:
            Dict keyed by config name, each containing memory, feature, and
            discussion ablation results.
        """
        results: Dict[str, Any] = {}
        test_conversation = self._generate_test_conversation()

        logger.info("=" * 70)
        logger.info("Starting Extended Ablation Study")
        logger.info(f"Configs: {configs or list(self.configs.keys())}")
        logger.info("=" * 70)

        for name, config in self.configs.items():
            if configs and name not in configs:
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Running ablation: {name}")
            logger.info(f"Config: {config}")
            logger.info(f"{'='*60}")

            config_start = time.time()

            # 1. Memory ablation
            memory_result = self.run_memory_ablation(config["use_three_layer_memory"])

            # 2. Feature prediction ablation
            feature_result = self.run_feature_prediction_ablation(config, test_conversation)

            # 3. Discussion room ablation
            discussion_result = await self.run_discussion_ablation(
                config["use_discussion_room"], test_conversation
            )

            results[name] = {
                "config": config,
                "memory": memory_result,
                "feature_prediction": feature_result,
                "discussion": discussion_result,
                "total_elapsed_seconds": round(time.time() - config_start, 2),
            }

            logger.info(f"Completed: {name} in {results[name]['total_elapsed_seconds']}s")

        # Generate comparative summary
        results["_summary"] = self._generate_summary(results)

        return results

    # ------------------------------------------------------------------
    # Test conversation generation
    # ------------------------------------------------------------------

    def _generate_test_conversation(self) -> List[Dict[str, str]]:
        """Return the static 20-turn test conversation for reproducible ablation.

        Uses a hardcoded realistic conversation to eliminate randomness across
        ablation runs. Every config sees the exact same input.
        """
        return list(_TEST_CONVERSATION_TURNS)

    # ------------------------------------------------------------------
    # Summary generation
    # ------------------------------------------------------------------

    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comparative summary across all ablation configs."""
        summary: Dict[str, Any] = {"configs_run": [], "comparison": {}}

        for name, data in results.items():
            if name.startswith("_"):
                continue
            summary["configs_run"].append(name)

            feature = data.get("feature_prediction", {})
            snapshots = feature.get("feature_snapshots", [])

            # Extract final-turn metrics
            final_snap = snapshots[-1] if snapshots else {}
            summary["comparison"][name] = {
                "memory_compression": data.get("memory", {}).get("compression_ratio", 0.0),
                "memory_episodic": data.get("memory", {}).get("episodic_memories", 0),
                "memory_semantic": data.get("memory", {}).get("semantic_memories", 0),
                "feature_n_features": final_snap.get("n_features", 0),
                "feature_avg_confidence": final_snap.get("avg_confidence", 0.0),
                "feature_low_confidence_count": final_snap.get("low_confidence_count", 0),
                "feature_elapsed_s": feature.get("elapsed_seconds", 0.0),
                "discussion_confidence": data.get("discussion", {}).get("confidence", 0.0),
                "discussion_n_proposals": data.get("discussion", {}).get("n_proposals", 0),
                "discussion_elapsed_s": data.get("discussion", {}).get("elapsed_seconds", 0.0),
                "total_elapsed_s": data.get("total_elapsed_seconds", 0.0),
            }

            # Conformal metrics from final snapshot
            if "conformal_avg_set_size" in final_snap:
                summary["comparison"][name]["conformal_avg_set_size"] = final_snap["conformal_avg_set_size"]
                summary["comparison"][name]["conformal_singletons"] = final_snap.get("conformal_singletons", 0)

        # Rank by feature prediction quality (higher avg confidence = better)
        ranked = sorted(
            summary["comparison"].items(),
            key=lambda x: x[1].get("feature_avg_confidence", 0.0),
            reverse=True,
        )
        summary["ranking_by_confidence"] = [name for name, _ in ranked]

        return summary


# =====================================================================
# CLI entry point
# =====================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Extended Ablation Study for SoulMatch",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python experiments/ablation_study.py\n"
            '  python experiments/ablation_study.py --configs "Full System" "w/o Bayesian Update"\n'
            "  python experiments/ablation_study.py --output experiments/ablation_custom.json\n"
        ),
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Ablation configs to run. Default: all configs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/ablation_results.json",
        help="Output JSON path. Default: experiments/ablation_results.json",
    )
    args = parser.parse_args()

    study = AblationStudy()
    results = await study.run_full_ablation(configs=args.configs)

    # Save results
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary table
    summary = results.get("_summary", {})
    comparison = summary.get("comparison", {})

    print("\n" + "=" * 80)
    print("Ablation Study Results Summary")
    print("=" * 80)
    print(f"{'Config':<30} {'Features':>8} {'Avg Conf':>10} {'Low Conf':>10} {'Time (s)':>10}")
    print("-" * 80)

    for name, metrics in comparison.items():
        print(
            f"{name:<30} "
            f"{metrics.get('feature_n_features', 0):>8} "
            f"{metrics.get('feature_avg_confidence', 0.0):>10.4f} "
            f"{metrics.get('feature_low_confidence_count', 0):>10} "
            f"{metrics.get('total_elapsed_s', 0.0):>10.1f}"
        )

    print("-" * 80)
    ranking = summary.get("ranking_by_confidence", [])
    if ranking:
        print(f"Ranking (by confidence): {' > '.join(ranking)}")
    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    asyncio.run(main())
