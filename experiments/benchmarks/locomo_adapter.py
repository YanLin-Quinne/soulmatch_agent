"""LoCoMo benchmark adapter for AI YOU evaluation.

LoCoMo (Maharana et al., 2024) evaluates long-term conversational memory
across 5 question types:
1. Single-hop factual recall
2. Multi-hop reasoning
3. Temporal reasoning (ordering events)
4. Open-domain questions
5. Adversarial (false premise) questions

This adapter prefers LLM-assisted dialogue generation and judging when the
project's router is configured, but keeps deterministic fallbacks so the
benchmark remains runnable in local development.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

from src.agents.llm_router import router, AgentRole, MODEL_ROUTING, MODELS


QUESTION_TYPES = [
    "single_hop",
    "multi_hop",
    "temporal",
    "open_domain",
    "adversarial",
]

FALSE_PREMISE_ANSWERS = {
    "they never said that",
    "not mentioned",
    "unknown",
    "false premise",
    "never mentioned",
}


class LoCoMoAdapter:
    """Adapter for LoCoMo-style evaluation."""

    def __init__(
        self,
        data_path: Optional[str] = None,
        llm_router=None,
        use_llm_assistance: bool = False,
    ):
        self.data_path = data_path
        self.router = llm_router or router
        self.use_llm_assistance = use_llm_assistance
        self.dataset = None

    def generate_synthetic_locomo(
        self,
        n_sessions: int = 5,
        turns_per_session: int = 20,
        n_dialogues: int = 3,
    ) -> List[Dict]:
        """Generate LoCoMo-style evaluation data."""
        dataset = []
        for dialogue_idx in range(max(1, n_dialogues)):
            story = self._build_story_seed(dialogue_idx)
            sample = self._build_dialogue_sample(story, n_sessions, turns_per_session)
            sample = self._maybe_rewrite_with_llm(sample, story)
            dataset.append(sample)

        self.dataset = dataset
        return dataset

    def evaluate_memory_system(self, memory_system, dataset=None) -> Dict:
        """Evaluate a memory system on LoCoMo-style questions."""
        dataset = dataset or self.dataset or self.generate_synthetic_locomo()

        per_type_scores = {q_type: [] for q_type in QUESTION_TYPES}
        details = []

        for sample_idx, sample in enumerate(dataset):
            system = self._prepare_memory_system(memory_system)
            self._feed_dialogue(system, sample["dialogue_sessions"])

            for question in sample["questions"]:
                predicted = self._query_memory_system(system, question["question"])
                judgement = self._score_answer(
                    question=question["question"],
                    predicted_answer=predicted,
                    gold_answer=question["answer"],
                    question_type=question["type"],
                )
                per_type_scores[question["type"]].append(judgement["score"])
                details.append(
                    {
                        "sample_index": sample_idx,
                        "question_type": question["type"],
                        "question": question["question"],
                        "gold_answer": question["answer"],
                        "predicted_answer": predicted,
                        "score": judgement["score"],
                        "correct": judgement["correct"],
                        "evidence_session": question.get("evidence_session"),
                        "evidence_turn": question.get("evidence_turn"),
                    }
                )

        summary = {
            q_type: self._mean(scores)
            for q_type, scores in per_type_scores.items()
        }
        all_scores = [score for scores in per_type_scores.values() for score in scores]
        summary["overall"] = self._mean(all_scores)
        summary["counts"] = {q_type: len(scores) for q_type, scores in per_type_scores.items()}
        summary["details"] = details
        return summary

    def load_dataset(self, path: str) -> List[Dict]:
        """Load actual LoCoMo dataset if available."""
        input_path = Path(path)
        if not input_path.exists():
            raise FileNotFoundError(f"LoCoMo dataset not found at {input_path}")

        if input_path.suffix == ".jsonl":
            dataset = [
                json.loads(line)
                for line in input_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payload = json.loads(input_path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                dataset = payload
            elif isinstance(payload, dict) and "data" in payload:
                dataset = payload["data"]
            else:
                dataset = [payload]

        self.dataset = dataset
        return dataset

    def _build_story_seed(self, dialogue_idx: int) -> Dict:
        names = ["Maya", "Jordan", "Elena", "Ravi", "Nina", "Lucas"]
        jobs = [
            "product designer",
            "data journalist",
            "urban planner",
            "wildlife photographer",
            "museum curator",
            "physical therapist",
        ]
        new_jobs = [
            "lead designer at a climate-tech startup",
            "features editor at a local magazine",
            "planning manager for the city council",
            "creative director at an outdoor brand",
            "head of exhibitions at a contemporary gallery",
            "sports rehab specialist at a community clinic",
        ]
        cities = ["Lisbon", "Edinburgh", "Kyoto", "Porto", "Vancouver", "Seville"]
        restaurants = [
            "Green Fig Bistro",
            "Olive & Ember",
            "Saffron Table",
            "Juniper Kitchen",
            "Harbor Spoon",
            "Maple Lane Cafe",
        ]
        diets = ["vegetarian", "vegan", "omnivore"]
        hobbies = [
            ("trail photography", "weekend hiking"),
            ("ceramics", "visiting design markets"),
            ("running", "meal prepping"),
            ("book collecting", "slow travel"),
            ("birdwatching", "sketching"),
            ("live music", "cycling"),
        ]
        favorite_foods = ["mushroom ramen", "falafel wraps", "spicy noodles", "tacos", "sushi", "dim sum"]

        idx = dialogue_idx % len(names)
        primary_hobby, secondary_hobby = hobbies[idx]
        diet = diets[dialogue_idx % len(diets)]
        story = {
            "dialogue_id": f"locomo_{dialogue_idx}",
            "name": names[idx],
            "friend_name": names[(idx + 1) % len(names)],
            "job": jobs[idx],
            "new_job": new_jobs[idx],
            "trip_city": cities[idx],
            "restaurant": restaurants[idx],
            "diet": diet,
            "favorite_food": favorite_foods[idx],
            "primary_hobby": primary_hobby,
            "secondary_hobby": secondary_hobby,
            "pet_name": ["Pico", "Miso", "Taro", "Luna", "Otis", "Penny"][idx],
            "volunteer_cause": ["river cleanups", "food banks", "animal rescue", "library tutoring"][dialogue_idx % 4],
        }
        story["gift_idea"] = self._gift_idea(story["job"], story["primary_hobby"], story["secondary_hobby"])
        story["restaurant_type"] = self._restaurant_type_for_diet(diet)
        story["temporal_answer"] = "before"
        return story

    def _build_dialogue_sample(self, story: Dict, n_sessions: int, turns_per_session: int) -> Dict:
        sessions = []
        facts = {}

        for session_id in range(n_sessions):
            session, session_facts = self._build_session(story, session_id, turns_per_session)
            sessions.append(session)
            facts.update(session_facts)

        questions = [
            {
                "question": f"What restaurant did {story['name']} mention for dinner plans?",
                "answer": story["restaurant"],
                "type": "single_hop",
                "evidence_session": facts["restaurant"]["session_id"],
                "evidence_turn": facts["restaurant"]["turn_id"],
            },
            {
                "question": (
                    f"Based on {story['name']}'s work as a {story['job']} and their hobbies, "
                    f"what gift would likely suit them?"
                ),
                "answer": story["gift_idea"],
                "type": "multi_hop",
                "evidence_session": facts["job"]["session_id"],
                "evidence_turn": facts["job"]["turn_id"],
                "additional_evidence": [
                    {"session_id": facts["primary_hobby"]["session_id"], "turn_id": facts["primary_hobby"]["turn_id"]},
                    {"session_id": facts["secondary_hobby"]["session_id"], "turn_id": facts["secondary_hobby"]["turn_id"]},
                ],
            },
            {
                "question": (
                    f"Did {story['name']} mention the trip to {story['trip_city']} before or after "
                    f"talking about the new job?"
                ),
                "answer": story["temporal_answer"],
                "type": "temporal",
                "evidence_session": facts["trip_city"]["session_id"],
                "evidence_turn": facts["trip_city"]["turn_id"],
            },
            {
                "question": (
                    f"Given that {story['name']} said they are {story['diet']}, "
                    f"what restaurant type would suit them for a celebration dinner?"
                ),
                "answer": story["restaurant_type"],
                "type": "open_domain",
                "evidence_session": facts["diet"]["session_id"],
                "evidence_turn": facts["diet"]["turn_id"],
            },
            {
                "question": f"When did {story['name']} say they hated hiking?",
                "answer": "They never said that.",
                "type": "adversarial",
                "evidence_session": facts["secondary_hobby"]["session_id"],
                "evidence_turn": facts["secondary_hobby"]["turn_id"],
            },
        ]

        return {
            "dialogue_id": story["dialogue_id"],
            "dialogue_sessions": sessions,
            "questions": questions,
            "planted_facts": story,
            "total_turns": n_sessions * turns_per_session,
        }

    def _build_session(self, story: Dict, session_id: int, turns_per_session: int) -> tuple[Dict, Dict]:
        turns = []
        facts = {}

        def add_turn(speaker: str, message: str, fact_key: Optional[str] = None):
            turn_id = len(turns)
            turns.append({"turn_id": turn_id, "speaker": speaker, "message": message})
            if fact_key:
                facts[fact_key] = {"session_id": session_id, "turn_id": turn_id}

        if session_id == 0:
            exchanges = [
                ("bot", f"How has your week been, {story['name']}?"),
                ("user", f"Busy but good. I'm a {story['job']} and work has been full on lately.", "job"),
                ("bot", "What do you do when work finally calms down?"),
                ("user", f"I usually go for {story['primary_hobby']} on Saturdays because it clears my head.", "primary_hobby"),
                ("bot", "That sounds fun. What kind of food is your default comfort meal?"),
                ("user", f"{story['favorite_food'].title()} every time, and I'm strictly {story['diet']} these days.", "diet"),
                ("bot", "Any pets keeping you company at home?"),
                ("user", f"Yeah, my cat {story['pet_name']} supervises everything I do.", "pet_name"),
                ("bot", "What do you have planned for the weekend?"),
                ("user", f"If the weather holds, probably some {story['secondary_hobby']} and a quiet Sunday."),
            ]
        elif session_id == 1:
            exchanges = [
                ("bot", "Did you end up making weekend plans?"),
                ("user", f"Yes, I booked a table at {story['restaurant']} for Friday.", "restaurant"),
                ("bot", "Nice. Anything else you are looking forward to?"),
                ("user", f"I'm also planning a trip to {story['trip_city']} next month, which should be a reset.", "trip_city"),
                ("bot", "Are you still fitting in hobbies around that?"),
                ("user", f"Definitely. {story['secondary_hobby'].title()} keeps me grounded after work.", "secondary_hobby"),
                ("bot", "Have you taken on anything outside work lately?"),
                ("user", f"I've been helping with {story['volunteer_cause']} on Sunday mornings.", "volunteer_cause"),
                ("bot", "That is a packed schedule."),
                ("user", "It is, but I like feeling that my week has some shape to it."),
            ]
        else:
            exchanges = [
                ("bot", "What has changed since we last talked?"),
                ("user", f"I accepted a new role as {story['new_job']}, which starts after the trip.", "new_job"),
                ("bot", "That is a big move. How are you feeling about it?"),
                ("user", f"Excited, mostly. It still fits with how much I enjoy {story['primary_hobby']}."),
                ("bot", "Did you still keep the dinner booking?"),
                ("user", f"Yes, and {story['restaurant']} is still the place I'm most excited about this month."),
                ("bot", "What are you hoping to do in the new season of life?"),
                ("user", f"Travel more, keep up with {story['secondary_hobby']}, and avoid dropping my volunteer work."),
                ("bot", "Sounds like a solid plan."),
                ("user", f"I just want the move to feel deliberate instead of rushed before {story['trip_city']}."),
            ]

        for speaker, message, *fact_key in exchanges:
            add_turn(speaker, message, fact_key[0] if fact_key else None)

        while len(turns) < turns_per_session:
            filler_idx = len(turns)
            if filler_idx % 2 == 0:
                add_turn("bot", f"Tell me one more small detail from session {session_id + 1}.")
            else:
                add_turn("user", f"I keep circling back to {story['primary_hobby']} and planning ahead for the next week.")

        return {"session_id": session_id, "turns": turns[:turns_per_session]}, facts

    def _maybe_rewrite_with_llm(self, sample: Dict, story: Dict) -> Dict:
        """Optionally naturalize the synthetic dialogue while preserving planted facts."""
        if not self._llm_enabled(AgentRole.PERSONA):
            return sample
        try:
            rendered = json.dumps(sample["dialogue_sessions"], ensure_ascii=False)
            prompt = f"""Rewrite the following multi-session dialogue into more natural conversation.

Constraints:
- Preserve the number of sessions and turns.
- Preserve every factual detail exactly: {json.dumps(story, ensure_ascii=False)}
- Keep the same speakers and turn order.
- Return JSON only in the original schema.

Dialogue JSON:
{rendered}
"""
            response = self.router.chat(
                role=AgentRole.PERSONA,
                system="You rewrite structured dialogue into natural conversation while preserving facts exactly.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=2000,
                json_mode=True,
                timeout=20.0,
            )
            rewritten = self._parse_json(response)
            if isinstance(rewritten, list):
                sample["dialogue_sessions"] = rewritten
        except Exception as exc:
            logger.debug(f"[LoCoMoAdapter] Falling back to deterministic dialogue generation: {exc}")
        return sample

    def _prepare_memory_system(self, memory_system):
        if isinstance(memory_system, type):
            return memory_system()

        if callable(memory_system) and not hasattr(memory_system, "predict_personality"):
            try:
                return memory_system()
            except TypeError:
                pass

        if hasattr(memory_system, "reset") and callable(memory_system.reset):
            memory_system.reset()
            return memory_system

        if hasattr(memory_system, "clear") and callable(memory_system.clear):
            memory_system.clear()
            return memory_system

        if hasattr(memory_system, "working_memory") and hasattr(memory_system, "episodic_memory"):
            memory_system.working_memory = []
            memory_system.episodic_memory = []
            if hasattr(memory_system, "semantic_memory"):
                memory_system.semantic_memory = []
            if hasattr(memory_system, "dialogue_archive"):
                memory_system.dialogue_archive = {}
            if hasattr(memory_system, "current_turn"):
                memory_system.current_turn = 0
            return memory_system

        try:
            return copy.deepcopy(memory_system)
        except Exception:
            return memory_system

    def _feed_dialogue(self, memory_system, dialogue_sessions: List[Dict]) -> None:
        for session in dialogue_sessions:
            for turn in session.get("turns", []):
                self._add_turn(memory_system, turn)

    def _add_turn(self, memory_system, turn: Dict) -> None:
        speaker = turn.get("speaker", "user")
        message = turn.get("message", "")

        if hasattr(memory_system, "add_turn"):
            try:
                memory_system.add_turn(turn)
                return
            except TypeError:
                memory_system.add_turn(speaker, message)
                return

        if hasattr(memory_system, "add_to_working_memory"):
            memory_system.add_to_working_memory(speaker, message)
            return

        if hasattr(memory_system, "append"):
            memory_system.append(turn)

    def _query_memory_system(self, memory_system, question: str) -> str:
        for method_name in ("query", "answer_question", "answer", "ask"):
            if hasattr(memory_system, method_name):
                method = getattr(memory_system, method_name)
                try:
                    response = method(question)
                except TypeError:
                    response = method(question=question)
                return self._coerce_text(response)

        context = self._extract_memory_context(memory_system, question)
        if not context:
            return ""

        if not self._llm_enabled(AgentRole.MEMORY):
            return ""

        try:
            response = self.router.chat(
                role=AgentRole.MEMORY,
                system=(
                    "Answer the user's memory question using only the provided memory context. "
                    "If the answer is not supported, say 'Not mentioned in memory'."
                ),
                messages=[{"role": "user", "content": f"Memory context:\n{context}\n\nQuestion: {question}"}],
                temperature=0.0,
                max_tokens=120,
                timeout=20.0,
            )
            return self._coerce_text(response)
        except Exception as exc:
            logger.debug(f"[LoCoMoAdapter] Memory question fallback failed: {exc}")
            return ""

    def _extract_memory_context(self, memory_system, question: str) -> str:
        if hasattr(memory_system, "get_memory_context"):
            try:
                return self._coerce_text(memory_system.get_memory_context(query=question))
            except TypeError:
                return self._coerce_text(memory_system.get_memory_context(question))

        sections = []
        for method_name in (
            "get_working_memory_context",
            "get_episodic_memory_context",
            "get_semantic_memory_context",
        ):
            if hasattr(memory_system, method_name):
                method = getattr(memory_system, method_name)
                try:
                    sections.append(self._coerce_text(method(question)))
                except TypeError:
                    sections.append(self._coerce_text(method()))
        return "\n".join(section for section in sections if section)

    def _score_answer(
        self,
        question: str,
        predicted_answer: str,
        gold_answer: str,
        question_type: str,
    ) -> Dict:
        predicted = self._normalize(predicted_answer)
        gold = self._normalize(gold_answer)

        if question_type == "adversarial":
            correct = any(phrase in predicted for phrase in FALSE_PREMISE_ANSWERS) or "never" in predicted
            return {"score": 1.0 if correct else 0.0, "correct": correct}

        if predicted == gold or gold in predicted or predicted in gold:
            return {"score": 1.0, "correct": True}

        if question_type in {"single_hop", "temporal"}:
            return {"score": 0.0, "correct": False}

        llm_judgement = self._llm_judge(question, gold_answer, predicted_answer, question_type)
        if llm_judgement is not None:
            return llm_judgement

        token_overlap = self._token_overlap(predicted, gold)
        score = 1.0 if token_overlap >= 0.5 else 0.0
        return {"score": score, "correct": bool(score)}

    def _llm_judge(
        self,
        question: str,
        gold_answer: str,
        predicted_answer: str,
        question_type: str,
    ) -> Optional[Dict]:
        if not self._llm_enabled(AgentRole.FEATURE):
            return None
        try:
            prompt = f"""Score whether the predicted answer matches the gold answer for a memory benchmark.

Question type: {question_type}
Question: {question}
Gold answer: {gold_answer}
Predicted answer: {predicted_answer}

Return JSON:
{{
  "score": 0.0 or 1.0,
  "correct": true or false
}}
"""
            response = self.router.chat(
                role=AgentRole.FEATURE,
                system="You are an evaluation judge. Be strict about false facts and false premises.",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=80,
                json_mode=True,
                timeout=15.0,
            )
            parsed = self._parse_json(response)
            if isinstance(parsed, dict):
                score = float(parsed.get("score", 0.0))
                return {"score": 1.0 if score >= 0.5 else 0.0, "correct": bool(parsed.get("correct", score >= 0.5))}
        except Exception as exc:
            logger.debug(f"[LoCoMoAdapter] LLM judge unavailable: {exc}")
        return None

    def _gift_idea(self, job: str, primary_hobby: str, secondary_hobby: str) -> str:
        job_hint = "design notebook" if "designer" in job or "planner" in job else "field journal"
        if "photography" in primary_hobby or "photographer" in job:
            return f"a compact camera bag and a {job_hint}"
        if "hiking" in secondary_hobby:
            return f"a lightweight trail daypack and a {job_hint}"
        return f"a thoughtful {job_hint} linked to {primary_hobby}"

    def _restaurant_type_for_diet(self, diet: str) -> str:
        mapping = {
            "vegetarian": "a vegetarian-friendly Mediterranean restaurant",
            "vegan": "a vegan cafe with hearty mains",
            "omnivore": "a casual bistro with varied options",
        }
        return mapping.get(diet, "a restaurant that matches their dietary needs")

    def _parse_json(self, text: str):
        cleaned = text.strip()
        cleaned = re.sub(r"^```json\s*|\s*```$", "", cleaned, flags=re.MULTILINE)
        return json.loads(cleaned)

    def _coerce_text(self, value) -> str:
        if isinstance(value, dict):
            for key in ("answer", "response", "text"):
                if key in value:
                    return str(value[key])
            return json.dumps(value, ensure_ascii=False)
        return str(value or "")

    def _normalize(self, value: str) -> str:
        return " ".join(re.sub(r"[^a-z0-9\s]", " ", str(value).lower()).split())

    def _token_overlap(self, predicted: str, gold: str) -> float:
        pred_tokens = set(predicted.split())
        gold_tokens = set(gold.split())
        if not pred_tokens or not gold_tokens:
            return 0.0
        return len(pred_tokens & gold_tokens) / len(gold_tokens)

    def _mean(self, values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    def _llm_enabled(self, role: AgentRole) -> bool:
        if not self.use_llm_assistance:
            return False
        availability_check = getattr(self.router, "_is_available", None)
        if not callable(availability_check):
            return False

        for model_key in MODEL_ROUTING.get(role, []):
            spec = MODELS.get(model_key)
            if spec is None:
                continue
            try:
                if availability_check(spec.provider):
                    return True
            except Exception:
                continue
        return False


def run_locomo_evaluation(memory_systems: Dict, n_sessions: int = 5) -> Dict:
    """Run LoCoMo-style evaluation on multiple memory systems."""
    adapter = LoCoMoAdapter()
    dataset = adapter.generate_synthetic_locomo(n_sessions=n_sessions)

    results = {}
    for name, system in memory_systems.items():
        results[name] = adapter.evaluate_memory_system(system, dataset)
    return results
