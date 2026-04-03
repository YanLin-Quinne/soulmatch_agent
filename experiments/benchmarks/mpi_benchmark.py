"""MPI benchmark adapter for Big Five personality inference.

Uses the public-domain IPIP-NEO-120 item bank as a practical stand-in for the
MPI questionnaire when exact MPI appendix wording is not bundled locally.

Sources:
- Jiang et al., NeurIPS 2023, "Machine Personality Inventory"
- Johnson (2014) IPIP-NEO-120 item bank: https://ipip.ori.org/30FacetNEO-PI-RItems.htm
"""

from __future__ import annotations

import math
import random
from statistics import mean
from typing import Callable, Dict, List, Optional, Sequence

from loguru import logger

from experiments.baselines.utils import parse_json_response
from src.agents.llm_router import AgentRole, router

BIG_FIVE_DIMENSIONS = (
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
)

QUESTIONNAIRE_SOURCE_URL = "https://ipip.ori.org/30FacetNEO-PI-RItems.htm"

INTERVIEWER_PROMPTS = [
    "What does your ideal weekend look like?",
    "How do you usually approach planning a project or a busy week?",
    "What is it like for you to meet new people at a party or event?",
    "How do you usually handle disagreement with a friend or coworker?",
    "How do you feel about trying unfamiliar food, places, or hobbies?",
    "What happens internally when your day goes badly or something stressful comes up?",
    "When someone close to you is struggling, how do you usually respond?",
    "If you suddenly have a free evening, how do you like to spend it?",
    "How do you deal with deadlines, commitments, and promises?",
    "What kinds of books, films, art, or ideas tend to draw you in?",
    "How do you make decisions when there is uncertainty?",
    "How do you feel about routines versus change in daily life?",
]

IPIP_NEO_120_ITEMS = [
    {"text": "Worry about things.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Fear for the worst.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Am afraid of many things.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Get stressed out easily.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Get angry easily.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Get irritated easily.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Lose my temper.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Am not easily annoyed.", "dimension": "neuroticism", "keyed": "negative"},
    {"text": "Often feel blue.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Dislike myself.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Am often down in the dumps.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Feel comfortable with myself.", "dimension": "neuroticism", "keyed": "negative"},
    {"text": "Find it difficult to approach others.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Am afraid to draw attention to myself.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Only feel comfortable with friends.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Am not bothered by difficult social situations.", "dimension": "neuroticism", "keyed": "negative"},
    {"text": "Go on binges.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Rarely overindulge.", "dimension": "neuroticism", "keyed": "negative"},
    {"text": "Easily resist temptations.", "dimension": "neuroticism", "keyed": "negative"},
    {"text": "Am able to control my cravings.", "dimension": "neuroticism", "keyed": "negative"},
    {"text": "Panic easily.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Become overwhelmed by events.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Feel that I'm unable to deal with things.", "dimension": "neuroticism", "keyed": "positive"},
    {"text": "Remain calm under pressure.", "dimension": "neuroticism", "keyed": "negative"},
    {"text": "Make friends easily.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Feel comfortable around people.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Avoid contacts with others.", "dimension": "extraversion", "keyed": "negative"},
    {"text": "Keep others at a distance.", "dimension": "extraversion", "keyed": "negative"},
    {"text": "Love large parties.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Talk to a lot of different people at parties.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Prefer to be alone.", "dimension": "extraversion", "keyed": "negative"},
    {"text": "Avoid crowds.", "dimension": "extraversion", "keyed": "negative"},
    {"text": "Take charge.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Try to lead others.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Take control of things.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Wait for others to lead the way.", "dimension": "extraversion", "keyed": "negative"},
    {"text": "Am always busy.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Am always on the go.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Do a lot in my spare time.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Like to take it easy.", "dimension": "extraversion", "keyed": "negative"},
    {"text": "Love excitement.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Seek adventure.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Enjoy being reckless.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Act wild and crazy.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Radiate joy.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Have a lot of fun.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Love life.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Look at the bright side of life.", "dimension": "extraversion", "keyed": "positive"},
    {"text": "Have a vivid imagination.", "dimension": "openness", "keyed": "positive"},
    {"text": "Enjoy wild flights of fantasy.", "dimension": "openness", "keyed": "positive"},
    {"text": "Love to daydream.", "dimension": "openness", "keyed": "positive"},
    {"text": "Like to get lost in thought.", "dimension": "openness", "keyed": "positive"},
    {"text": "Believe in the importance of art.", "dimension": "openness", "keyed": "positive"},
    {"text": "See beauty in things that others might not notice.", "dimension": "openness", "keyed": "positive"},
    {"text": "Do not like poetry.", "dimension": "openness", "keyed": "negative"},
    {"text": "Do not enjoy going to art museums.", "dimension": "openness", "keyed": "negative"},
    {"text": "Experience my emotions intensely.", "dimension": "openness", "keyed": "positive"},
    {"text": "Feel others' emotions.", "dimension": "openness", "keyed": "positive"},
    {"text": "Rarely notice my emotional reactions.", "dimension": "openness", "keyed": "negative"},
    {"text": "Don't understand people who get emotional.", "dimension": "openness", "keyed": "negative"},
    {"text": "Prefer variety to routine.", "dimension": "openness", "keyed": "positive"},
    {"text": "Prefer to stick with things that I know.", "dimension": "openness", "keyed": "negative"},
    {"text": "Dislike changes.", "dimension": "openness", "keyed": "negative"},
    {"text": "Am attached to conventional ways.", "dimension": "openness", "keyed": "negative"},
    {"text": "Love to read challenging material.", "dimension": "openness", "keyed": "positive"},
    {"text": "Avoid philosophical discussions.", "dimension": "openness", "keyed": "negative"},
    {"text": "Have difficulty understanding abstract ideas.", "dimension": "openness", "keyed": "negative"},
    {"text": "Am not interested in theoretical discussions.", "dimension": "openness", "keyed": "negative"},
    {"text": "Tend to vote for liberal political candidates.", "dimension": "openness", "keyed": "positive"},
    {"text": "Believe that there is no absolute right and wrong.", "dimension": "openness", "keyed": "positive"},
    {"text": "Tend to vote for conservative political candidates.", "dimension": "openness", "keyed": "negative"},
    {"text": "Believe that we should be tough on crime.", "dimension": "openness", "keyed": "negative"},
    {"text": "Trust others.", "dimension": "agreeableness", "keyed": "positive"},
    {"text": "Believe that others have good intentions.", "dimension": "agreeableness", "keyed": "positive"},
    {"text": "Trust what people say.", "dimension": "agreeableness", "keyed": "positive"},
    {"text": "Distrust people.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Use others for my own ends.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Cheat to get ahead.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Take advantage of others.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Obstruct others' plans.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Am concerned about others.", "dimension": "agreeableness", "keyed": "positive"},
    {"text": "Love to help others.", "dimension": "agreeableness", "keyed": "positive"},
    {"text": "Am indifferent to the feelings of others.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Take no time for others.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Love a good fight.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Yell at people.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Insult people.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Get back at others.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Believe that I am better than others.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Think highly of myself.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Have a high opinion of myself.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Boast about my virtues.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Sympathize with the homeless.", "dimension": "agreeableness", "keyed": "positive"},
    {"text": "Feel sympathy for those who are worse off than myself.", "dimension": "agreeableness", "keyed": "positive"},
    {"text": "Am not interested in other people's problems.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Try not to think about the needy.", "dimension": "agreeableness", "keyed": "negative"},
    {"text": "Complete tasks successfully.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Excel in what I do.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Handle tasks smoothly.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Know how to get things done.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Like to tidy up.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Often forget to put things back in their proper place.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Leave a mess in my room.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Leave my belongings around.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Keep my promises.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Tell the truth.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Break rules.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Break my promises.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Do more than what's expected of me.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Work hard.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Put little time and effort into my work.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Do just enough work to get by.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Am always prepared.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Carry out my plans.", "dimension": "conscientiousness", "keyed": "positive"},
    {"text": "Waste my time.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Have difficulty starting tasks.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Jump into things without thinking.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Make rash decisions.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Rush into things.", "dimension": "conscientiousness", "keyed": "negative"},
    {"text": "Act without thinking.", "dimension": "conscientiousness", "keyed": "negative"},
]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, float(value)))


def _safe_mean(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _pearson_correlation(xs: Sequence[float], ys: Sequence[float]) -> float:
    if len(xs) != len(ys):
        raise ValueError(f"Length mismatch: {len(xs)} != {len(ys)}")
    if len(xs) < 2:
        return 0.0

    mean_x = _safe_mean(xs)
    mean_y = _safe_mean(ys)
    centered_x = [x - mean_x for x in xs]
    centered_y = [y - mean_y for y in ys]
    denom_x = math.sqrt(sum(val * val for val in centered_x))
    denom_y = math.sqrt(sum(val * val for val in centered_y))
    if denom_x == 0.0 or denom_y == 0.0:
        return 0.0
    covariance = sum(x * y for x, y in zip(centered_x, centered_y))
    return float(covariance / (denom_x * denom_y))


class MPIBenchmark:
    def __init__(self):
        self.items = IPIP_NEO_120_ITEMS
        self.random = random.Random(42)

    def generate_persona_dialogue(
        self,
        target_scores: Dict[str, float],
        n_turns: int = 10,
    ) -> List[Dict]:
        """Generate a synthetic interview dialogue for a target Big Five profile."""
        if n_turns <= 0:
            raise ValueError("n_turns must be positive")

        prompts = [INTERVIEWER_PROMPTS[i % len(INTERVIEWER_PROMPTS)] for i in range(n_turns)]
        answers = self._generate_llm_persona_answers(target_scores, prompts)
        dialogue: List[Dict] = []

        for turn_idx, (prompt, answer) in enumerate(zip(prompts, answers), start=1):
            dialogue.append(
                {
                    "speaker": "interviewer",
                    "message": prompt,
                    "turn": turn_idx * 2 - 1,
                }
            )
            dialogue.append(
                {
                    "speaker": "persona",
                    "message": answer,
                    "turn": turn_idx * 2,
                }
            )

        return dialogue

    def score_responses(self, responses: List[int]) -> Dict[str, float]:
        """Score questionnaire responses into normalized [0, 1] Big Five values."""
        if len(responses) != len(self.items):
            raise ValueError(
                f"Expected {len(self.items)} responses, received {len(responses)}"
            )

        per_dimension: Dict[str, List[float]] = {dimension: [] for dimension in BIG_FIVE_DIMENSIONS}
        for response, item in zip(responses, self.items):
            likert = int(response)
            if likert < 1 or likert > 5:
                raise ValueError(f"Likert responses must be between 1 and 5, got {likert}")
            normalized = (likert - 1) / 4.0
            if item["keyed"] == "negative":
                normalized = 1.0 - normalized
            per_dimension[item["dimension"]].append(normalized)

        return {
            dimension: round(_safe_mean(scores), 4)
            for dimension, scores in per_dimension.items()
        }

    def evaluate_system(
        self,
        system_predict_fn,
        n_personas: int = 20,
        n_turns: int = 10,
    ) -> Dict:
        """Benchmark a personality inference system against questionnaire ground truth."""
        if n_personas <= 0:
            raise ValueError("n_personas must be positive")

        ground_truths: List[Dict[str, float]] = []
        predictions: List[Dict[str, float]] = []
        samples: List[Dict] = []
        failed_predictions = 0

        for persona_idx in range(n_personas):
            target_scores = self._sample_target_scores()
            questionnaire_responses = self._simulate_questionnaire_responses(target_scores)
            ground_truth = self.score_responses(questionnaire_responses)
            dialogue = self.generate_persona_dialogue(ground_truth, n_turns=n_turns)

            try:
                raw_prediction = self._call_predict_fn(system_predict_fn, dialogue)
                prediction = self._normalize_prediction(raw_prediction)
            except Exception as exc:
                failed_predictions += 1
                logger.warning(
                    "MPI benchmark prediction failed for persona {}: {}",
                    persona_idx,
                    exc,
                )
                prediction = {dimension: 0.5 for dimension in BIG_FIVE_DIMENSIONS}

            ground_truths.append(ground_truth)
            predictions.append(prediction)
            samples.append(
                {
                    "persona_id": persona_idx,
                    "target_scores": target_scores,
                    "ground_truth": ground_truth,
                    "predicted": prediction,
                    "responses": questionnaire_responses,
                    "dialogue": dialogue,
                }
            )

        mae_by_dim = {}
        rmse_by_dim = {}
        correlation_by_dim = {}

        for dimension in BIG_FIVE_DIMENSIONS:
            y_true = [sample[dimension] for sample in ground_truths]
            y_pred = [sample[dimension] for sample in predictions]
            abs_errors = [abs(true - pred) for true, pred in zip(y_true, y_pred)]
            squared_errors = [(true - pred) ** 2 for true, pred in zip(y_true, y_pred)]

            mae_by_dim[dimension] = round(_safe_mean(abs_errors), 4)
            rmse_by_dim[dimension] = round(math.sqrt(_safe_mean(squared_errors)), 4)
            correlation_by_dim[dimension] = round(_pearson_correlation(y_true, y_pred), 4)

        return {
            "benchmark": "mpi_ipip_neo_120",
            "questionnaire_source": QUESTIONNAIRE_SOURCE_URL,
            "item_count": len(self.items),
            "n_personas": n_personas,
            "n_turns": n_turns,
            "failed_predictions": failed_predictions,
            "mae": mae_by_dim,
            "rmse": rmse_by_dim,
            "correlation": correlation_by_dim,
            "mean_mae": round(_safe_mean(list(mae_by_dim.values())), 4),
            "mean_rmse": round(_safe_mean(list(rmse_by_dim.values())), 4),
            "mean_correlation": round(_safe_mean(list(correlation_by_dim.values())), 4),
            "samples": samples,
        }

    def _generate_llm_persona_answers(
        self,
        target_scores: Dict[str, float],
        prompts: Sequence[str],
    ) -> List[str]:
        prompt_block = "\n".join(f"{idx + 1}. {prompt}" for idx, prompt in enumerate(prompts))
        user_prompt = f"""Create first-person answers for a personality benchmark interview.

Target Big Five profile on a 0-1 scale:
- openness: {target_scores['openness']:.2f}
- conscientiousness: {target_scores['conscientiousness']:.2f}
- extraversion: {target_scores['extraversion']:.2f}
- agreeableness: {target_scores['agreeableness']:.2f}
- neuroticism: {target_scores['neuroticism']:.2f}

Write one natural answer per interviewer question. The speaker should sound like a real person, not a psychology summary. Show the personality indirectly through preferences, habits, emotional reactions, and tone. Keep each answer to 2-4 sentences.

Interviewer questions:
{prompt_block}

Return JSON only:
{{
  "responses": [
    "answer 1",
    "answer 2"
  ]
}}"""

        try:
            response = router.chat(
                role=AgentRole.PERSONA,
                system="You role-play consistent personas for synthetic benchmark conversations.",
                messages=[{"role": "user", "content": user_prompt}],
                temperature=0.9,
                max_tokens=1600,
                json_mode=True,
            )
            parsed = parse_json_response(response)
            answers = self._extract_answers(parsed)
            if len(answers) == len(prompts):
                return answers
            logger.warning(
                "MPI benchmark received {} LLM answers for {} prompts; using fallback for missing turns",
                len(answers),
                len(prompts),
            )
        except Exception as exc:
            logger.warning("MPI benchmark dialogue generation fell back to templates: {}", exc)
            answers = []

        completed_answers = list(answers)
        for idx in range(len(completed_answers), len(prompts)):
            completed_answers.append(
                self._fallback_persona_answer(target_scores, prompts[idx], idx)
            )
        return completed_answers

    def _extract_answers(self, parsed: Dict) -> List[str]:
        raw_answers = parsed.get("responses", []) if isinstance(parsed, dict) else []
        answers: List[str] = []
        for entry in raw_answers:
            if isinstance(entry, str) and entry.strip():
                answers.append(entry.strip())
            elif isinstance(entry, dict):
                text = entry.get("answer") or entry.get("response") or entry.get("text")
                if isinstance(text, str) and text.strip():
                    answers.append(text.strip())
        return answers

    def _fallback_persona_answer(
        self,
        target_scores: Dict[str, float],
        prompt: str,
        prompt_index: int,
    ) -> str:
        openness = target_scores["openness"]
        conscientiousness = target_scores["conscientiousness"]
        extraversion = target_scores["extraversion"]
        agreeableness = target_scores["agreeableness"]
        neuroticism = target_scores["neuroticism"]

        topic_index = prompt_index % len(INTERVIEWER_PROMPTS)
        if topic_index == 0:
            return (
                f"My ideal weekend usually {'includes seeing people and getting out of the house' if extraversion >= 0.6 else 'has a lot of quiet time and room to recharge'}. "
                f"I {'like trying somewhere new or doing something a little different' if openness >= 0.6 else 'usually stick with familiar places and routines that already feel good'}. "
                f"I {'leave some structure in place so the weekend does not disappear on me' if conscientiousness >= 0.6 else 'prefer to keep it loose and see what mood I am in' }."
            )
        if topic_index == 1:
            return (
                f"I usually {'make a plan, break things into steps, and keep track of deadlines' if conscientiousness >= 0.6 else 'start with the broad idea and figure out details as I go'}. "
                f"I {'enjoy building in space for experimentation if something better comes up' if openness >= 0.6 else 'prefer reliable methods over changing course too often'}. "
                f"When things get messy, I {'stay fairly even and adjust' if neuroticism < 0.5 else 'can overthink whether I am missing something important' }."
            )
        if topic_index == 2:
            return (
                f"Meeting new people is usually {'energizing for me and I can get into a conversation pretty quickly' if extraversion >= 0.6 else 'something I ease into slowly, especially in big groups'}. "
                f"I {'try to make people feel at ease and look for common ground' if agreeableness >= 0.6 else 'can be pretty direct and do not force chemistry if it is not there'}. "
                f"I {'rarely spiral about first impressions' if neuroticism < 0.5 else 'sometimes replay the interaction afterward more than I should' }."
            )
        if topic_index == 3:
            return (
                f"In a disagreement, I {'try to keep the tone respectful and actually understand the other person' if agreeableness >= 0.6 else 'would rather be blunt and get the real issue on the table'}. "
                f"I {'like to resolve it in a clear, structured way' if conscientiousness >= 0.6 else 'usually talk it through more organically'}. "
                f"I {'do not get too rattled by conflict' if neuroticism < 0.5 else 'feel the tension pretty strongly even when I am trying to stay composed' }."
            )
        if topic_index == 4:
            return (
                f"I am {'usually excited by unfamiliar places, ideas, or hobbies' if openness >= 0.6 else 'pretty selective and more comfortable with what I already know I like'}. "
                f"I {'see novelty as part of what keeps life interesting' if openness >= 0.75 else 'do not need constant novelty to feel engaged'}. "
                f"If it goes badly, I {'laugh it off and move on' if neuroticism < 0.5 else 'can get self-conscious about whether it was the wrong call' }."
            )
        if topic_index == 5:
            return (
                f"When a day goes badly, I {'stay relatively calm and focus on what I can do next' if neuroticism < 0.5 else 'feel it hard at first and can start running through worst-case scenarios'}. "
                f"I {'fall back on routines and a checklist' if conscientiousness >= 0.6 else 'need a little time before I can organize myself again'}. "
                f"It also helps that I {'will reach out to people instead of sitting in it alone' if extraversion >= 0.6 else 'usually need quiet space to settle my head' }."
            )
        if topic_index == 6:
            return (
                f"If someone close to me is struggling, I {'naturally move toward them and try to be useful' if agreeableness >= 0.6 else 'want to help, but I am more likely to focus on practical fixes than emotional cushioning'}. "
                f"I {'listen carefully before jumping in' if openness >= 0.5 else 'try to keep my advice grounded and straightforward'}. "
                f"I {'follow through consistently if I say I will help' if conscientiousness >= 0.6 else 'do better when support can be flexible rather than highly structured' }."
            )
        if topic_index == 7:
            return (
                f"With a free evening, I usually {'look for company or something lively to do' if extraversion >= 0.6 else 'gravitate toward a quieter, lower-key plan'}. "
                f"I {'like mixing in something new, creative, or mentally interesting' if openness >= 0.6 else 'am happiest with familiar comforts and simple routines'}. "
                f"I {'still like having a loose plan for the night' if conscientiousness >= 0.6 else 'prefer not to over-structure my downtime' }."
            )
        if topic_index == 8:
            return (
                f"I take deadlines and promises {'seriously and usually build in enough margin to deliver' if conscientiousness >= 0.6 else 'seriously, but I am definitely more last-minute than I would like'}. "
                f"I {'hate letting people down' if agreeableness >= 0.6 else 'care more about being honest than sounding polished if something slips'}. "
                f"Under pressure I {'stay fairly steady' if neuroticism < 0.5 else 'can get tense and mentally noisy even when I am still getting the work done' }."
            )
        if topic_index == 9:
            return (
                f"I am usually drawn to {'ideas and art that make me think or see something differently' if openness >= 0.6 else 'stories and media that feel grounded, clear, and familiar'}. "
                f"I {'like wrestling with ambiguity and layered meanings' if openness >= 0.75 else 'prefer things that connect quickly and concretely'}. "
                f"I also {'love talking about it with other people' if extraversion >= 0.6 else 'tend to sit with it privately before I say much' }."
            )
        if topic_index == 10:
            return (
                f"When things are uncertain, I {'can tolerate ambiguity and explore a few possibilities' if openness >= 0.6 else 'want enough information to choose the most reliable option'}. "
                f"I {'usually create a plan so I am not drifting' if conscientiousness >= 0.6 else 'keep the decision lighter and adapt as I learn more'}. "
                f"I {'do not spiral too much over unknowns' if neuroticism < 0.5 else 'can get stuck looping on what might go wrong' }."
            )
        return (
            f"I {'like some change because it keeps me engaged' if openness >= 0.6 else 'prefer a steady routine once I find something that works'}. "
            f"I {'build structure around my habits' if conscientiousness >= 0.6 else 'leave room to improvise from day to day'}. "
            f"Overall I {'adapt pretty calmly' if neuroticism < 0.5 else 'notice stress faster when life gets unsettled' }."
        )

    def _simulate_questionnaire_responses(
        self,
        target_scores: Dict[str, float],
    ) -> List[int]:
        responses: List[int] = []
        for item in self.items:
            trait_value = _clamp(target_scores[item["dimension"]])
            expected = 1.0 + 4.0 * trait_value
            if item["keyed"] == "negative":
                expected = 6.0 - expected

            noisy = self.random.gauss(expected, 0.45)
            responses.append(int(round(max(1.0, min(5.0, noisy)))))
        return responses

    def _sample_target_scores(self) -> Dict[str, float]:
        return {
            dimension: round(self.random.uniform(0.1, 0.9), 3)
            for dimension in BIG_FIVE_DIMENSIONS
        }

    def _call_predict_fn(self, system_predict_fn, dialogue: List[Dict]):
        if callable(system_predict_fn):
            return system_predict_fn(dialogue)
        if hasattr(system_predict_fn, "predict_personality"):
            return system_predict_fn.predict_personality(dialogue)
        raise TypeError(
            "system_predict_fn must be callable or expose a predict_personality(dialogue) method"
        )

    def _normalize_prediction(self, prediction: Optional[Dict]) -> Dict[str, float]:
        if not isinstance(prediction, dict):
            raise ValueError("Prediction must be a dict")

        if isinstance(prediction.get("big_five"), dict):
            candidate = prediction["big_five"]
        elif isinstance(prediction.get("features"), dict):
            features = prediction["features"]
            candidate = {
                dimension: features.get(f"big_five_{dimension}")
                for dimension in BIG_FIVE_DIMENSIONS
            }
        else:
            candidate = prediction

        normalized = {}
        for dimension in BIG_FIVE_DIMENSIONS:
            value = candidate.get(dimension)
            if value is None:
                value = candidate.get(f"big_five_{dimension}")
            normalized[dimension] = _clamp(0.5 if value is None else float(value))
        return normalized


def run_mpi_benchmark(
    predict_fn: Callable,
    n_personas: int = 20,
    n_turns: int = 10,
) -> Dict:
    """Standalone MPI benchmark runner."""
    benchmark = MPIBenchmark()
    return benchmark.evaluate_system(
        predict_fn,
        n_personas=n_personas,
        n_turns=n_turns,
    )
