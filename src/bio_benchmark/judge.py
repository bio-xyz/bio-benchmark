from __future__ import annotations

import json
import os
import random
import string
from dataclasses import dataclass
from typing import Any

from openai import OpenAI

from .config import JudgeConfig


@dataclass
class GradeResult:
    question_id: str
    correct: bool
    reasoning: str


@dataclass
class BatchThreeModeEvaluation:
    direct_by_question_id: dict[str, GradeResult]
    mcq_with_refusal_by_question_id: dict[str, GradeResult]
    mcq_without_refusal_by_question_id: dict[str, GradeResult]
    mapped_with_refusal: str
    mapped_without_refusal: str
    options_with_refusal_by_question_id: dict[str, list[str]]
    options_without_refusal_by_question_id: dict[str, list[str]]
    grader_model: str
    option_chooser_model: str
    judge_error: str = ""


def _resolve_openai_api_key(config: JudgeConfig) -> str:
    if config.api_key:
        return config.api_key
    key = os.getenv(config.api_key_env)
    if key:
        return key
    raise ValueError(
        f"Missing OpenAI key. Set judge.api_key or env var '{config.api_key_env}'."
    )


def _to_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    return bool(value)


def _build_question_text(questions_data: list[dict[str, Any]]) -> str:
    questions_text = ""
    for i, q_data in enumerate(questions_data, 1):
        questions_text += (
            f"{i}. {q_data['question']}\n   Ground Truth: {q_data['ground_truth']}\n\n"
        )
    return questions_text


def llm_as_judge_batch(
    *,
    client: OpenAI,
    questions_data: list[dict[str, Any]],
    agent_response: str,
    model: str,
) -> list[GradeResult]:
    """Copied prompt/logic pattern from bio-bixbench generate_trajectories.py."""
    questions_text = _build_question_text(questions_data)
    prompt = f"""You are evaluating an AI agent's answers to multiple biomedical research questions.

QUESTIONS AND GROUND TRUTHS:
{questions_text}

AGENT'S RESPONSE:
{agent_response}

TASK: Grade each question individually.

Guidelines:
- Match each numbered answer in the agent's response to the corresponding question
- Consider numerical equivalence (0.0 = 0)
- Consider semantic equivalence (Yes = True, No = False)
- If agent says "Cannot be determined", check if ground truth also indicates uncertainty
- If agent's answer is missing for a question, mark as incorrect

Respond in this exact JSON format:
{{
  "evaluations": [
    {{
      "question_number": 1,
      "correct": true/false,
      "reasoning": "Brief explanation"
    }},
    {{
      "question_number": 2,
      "correct": true/false,
      "reasoning": "Brief explanation"
    }}
  ]
}}"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        raw_content = response.choices[0].message.content or "{}"
        payload = json.loads(raw_content)
        graded_results: list[GradeResult] = []
        for i, evaluation in enumerate(payload.get("evaluations", [])):
            if i < len(questions_data):
                graded_results.append(
                    GradeResult(
                        question_id=str(questions_data[i]["question_id"]),
                        correct=_to_bool(evaluation.get("correct", False)),
                        reasoning=str(evaluation.get("reasoning", "")),
                    )
                )
        while len(graded_results) < len(questions_data):
            q = questions_data[len(graded_results)]
            graded_results.append(
                GradeResult(
                    question_id=str(q["question_id"]),
                    correct=False,
                    reasoning="Missing evaluation output",
                )
            )
        return graded_results
    except Exception as exc:
        return [
            GradeResult(
                question_id=str(q["question_id"]),
                correct=False,
                reasoning=f"Grading failed: {exc}",
            )
            for q in questions_data
        ]


def _llm_choose_single_option(
    *,
    client: OpenAI,
    question: dict[str, Any],
    agent_response: str,
    model: str,
    question_index: int,
    total_questions: int,
    include_refusal: bool = False,
) -> str:
    """Use tool calling to force the LLM to pick exactly one valid option."""
    options = question["options"]

    refusal_instruction = ""
    if include_refusal:
        refusal_instruction = (
            "\n- If the agent's answer indicates uncertainty, inability to answer, "
            'or says something like "Cannot be determined", choose the '
            "refusal/uncertainty option if available."
        )
    else:
        refusal_instruction = (
            "\n- Even if the agent expresses uncertainty, you MUST choose the best "
            "matching option."
            '\n- Do NOT choose a refusal or "cannot determine" option - always pick '
            "the closest actual answer."
        )

    options_str = "\n".join([f"  - {opt}" for opt in options])
    prompt = f"""You are mapping an AI agent's free-form answer to a multiple-choice option.

This is question {question_index} of {total_questions}.

QUESTION:
{question["question"]}

OPTIONS:
{options_str}

AGENT'S FULL RESPONSE (find the answer relevant to this question):
{agent_response}

TASK: Select the single MCQ option that best matches what the agent answered for this question.

Guidelines:
- Locate the agent's answer for this specific question within the full response
- Select the option that is semantically closest to what the agent answered
- Consider numerical equivalence (0.0 = 0, 3.14 ≈ 3.1415)
- Consider semantic equivalence (Yes = True, No = False)
- If the agent provides a specific value, find the option closest to that value{refusal_instruction}

Call the select_option tool with your choice."""

    tool_def = {
        "type": "function",
        "function": {
            "name": "select_option",
            "description": "Select the MCQ option that best matches the agent's answer",
            "parameters": {
                "type": "object",
                "properties": {
                    "selected_option": {
                        "type": "string",
                        "enum": options,
                        "description": "The selected option text, exactly as listed",
                    }
                },
                "required": ["selected_option"],
            },
        },
    }

    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        tools=[tool_def],
        tool_choice={"type": "function", "function": {"name": "select_option"}},
    )

    tool_call = response.choices[0].message.tool_calls[0]
    args = json.loads(tool_call.function.arguments)
    return args["selected_option"]


def llm_as_option_chooser(
    *,
    client: OpenAI,
    questions_data: list[dict[str, Any]],
    agent_response: str,
    model: str,
    include_refusal: bool = False,
) -> str:
    """Map agent free-form answers to MCQ options, one question at a time via tool calling."""
    parts: list[str] = []
    for i, q_data in enumerate(questions_data, 1):
        try:
            selected = _llm_choose_single_option(
                client=client,
                question=q_data,
                agent_response=agent_response,
                model=model,
                question_index=i,
                total_questions=len(questions_data),
                include_refusal=include_refusal,
            )
        except Exception:
            selected = ""
        parts.append(f"{i}. {selected}")
    return "\n".join(parts)


def _parse_distractors(raw_distractors: Any) -> list[str]:
    if isinstance(raw_distractors, list):
        return [str(item) for item in raw_distractors]
    if isinstance(raw_distractors, str):
        stripped = raw_distractors.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
            if isinstance(parsed, list):
                return [str(item) for item in parsed]
        except json.JSONDecodeError:
            pass
    return []


def _lettered_options(values: list[str]) -> list[str]:
    alphabet = string.ascii_uppercase
    options: list[str] = []
    for i, value in enumerate(values):
        letter = alphabet[i] if i < len(alphabet) else f"Opt{i+1}"
        options.append(f"({letter}) {value}")
    return options


def _grade_map(results: list[GradeResult]) -> dict[str, GradeResult]:
    return {grade.question_id: grade for grade in results}


def _disabled_grade_map(question_ids: list[str], reason: str) -> dict[str, GradeResult]:
    return {
        question_id: GradeResult(
            question_id=question_id,
            correct=False,
            reasoning=reason,
        )
        for question_id in question_ids
    }


def _randomize_choices(
    ideal: str,
    distractors: list[str],
    *,
    with_refusal: bool,
    refusal_choice: str,
) -> list[str]:
    alphabet = string.ascii_uppercase
    choices = [ideal, refusal_choice, *distractors] if with_refusal else [ideal, *distractors]
    n_choices = len(choices)
    if n_choices <= len(alphabet):
        perm = list(range(n_choices))
        random.shuffle(perm)
        return [f"({letter}) {choices[idx]}" for letter, idx in zip(alphabet, perm, strict=False)]
    return _lettered_options(choices)


def evaluate_batch_three_modes(
    *,
    config: JudgeConfig,
    questions: list[dict[str, Any]],
    agent_response: str,
) -> BatchThreeModeEvaluation:
    question_ids = [str(q["question_id"]) for q in questions]
    if not config.enabled:
        disabled_map = _disabled_grade_map(question_ids, "Judge disabled in config")
        return BatchThreeModeEvaluation(
            direct_by_question_id=disabled_map,
            mcq_with_refusal_by_question_id=disabled_map,
            mcq_without_refusal_by_question_id=disabled_map,
            mapped_with_refusal="",
            mapped_without_refusal="",
            options_with_refusal_by_question_id={qid: [] for qid in question_ids},
            options_without_refusal_by_question_id={qid: [] for qid in question_ids},
            grader_model=config.grader_model,
            option_chooser_model=config.option_chooser_model,
        )

    try:
        client = OpenAI(
            api_key=_resolve_openai_api_key(config),
            timeout=config.timeout_seconds,
        )

        direct_questions = [
            {
                "question_id": str(question["question_id"]),
                "question": str(question["question"]),
                "ground_truth": str(question["ground_truth"]),
            }
            for question in questions
        ]
        direct_grades = llm_as_judge_batch(
            client=client,
            questions_data=direct_questions,
            agent_response=agent_response,
            model=config.grader_model,
        )
        direct_map = _grade_map(direct_grades)

        options_without_refusal_by_qid: dict[str, list[str]] = {}
        options_with_refusal_by_qid: dict[str, list[str]] = {}
        questions_mcq_with_refusal: list[dict[str, Any]] = []
        questions_mcq_without_refusal: list[dict[str, Any]] = []
        for question in questions:
            question_id = str(question["question_id"])
            ground_truth = str(question["ground_truth"])
            distractor_list = _parse_distractors(question.get("distractors"))
            # Match bixbench behavior: remove duplicates and any accidental copies of ideal/refusal text.
            distractor_set = {
                item
                for item in distractor_list
                if item != ground_truth and item != config.refusal_option_text
            }
            base_options = _randomize_choices(
                ground_truth,
                sorted(distractor_set),
                with_refusal=False,
                refusal_choice=config.refusal_option_text,
            )
            options_without_refusal_by_qid[question_id] = base_options
            with_refusal = _randomize_choices(
                ground_truth,
                sorted(distractor_set),
                with_refusal=True,
                refusal_choice=config.refusal_option_text,
            )
            options_with_refusal_by_qid[question_id] = with_refusal
            questions_mcq_with_refusal.append(
                {
                    "question_id": question_id,
                    "question": str(question["question"]),
                    "ground_truth": ground_truth,
                    "options": with_refusal,
                }
            )
            questions_mcq_without_refusal.append(
                {
                    "question_id": question_id,
                    "question": str(question["question"]),
                    "ground_truth": ground_truth,
                    "options": base_options,
                }
            )

        if not config.include_mcq_modes:
            skipped_map = _disabled_grade_map(question_ids, "MCQ mode disabled in config")
            return BatchThreeModeEvaluation(
                direct_by_question_id=direct_map,
                mcq_with_refusal_by_question_id=skipped_map,
                mcq_without_refusal_by_question_id=skipped_map,
                mapped_with_refusal="",
                mapped_without_refusal="",
                options_with_refusal_by_question_id=options_with_refusal_by_qid,
                options_without_refusal_by_question_id=options_without_refusal_by_qid,
                grader_model=config.grader_model,
                option_chooser_model=config.option_chooser_model,
            )

        mapped_with_refusal = llm_as_option_chooser(
            client=client,
            questions_data=questions_mcq_with_refusal,
            agent_response=agent_response,
            model=config.option_chooser_model,
            include_refusal=True,
        )
        mapped_without_refusal = llm_as_option_chooser(
            client=client,
            questions_data=questions_mcq_without_refusal,
            agent_response=agent_response,
            model=config.option_chooser_model,
            include_refusal=False,
        )
        mcq_with_refusal_map = _grade_map(
            llm_as_judge_batch(
                client=client,
                questions_data=questions_mcq_with_refusal,
                agent_response=mapped_with_refusal,
                model=config.grader_model,
            )
        )
        mcq_without_refusal_map = _grade_map(
            llm_as_judge_batch(
                client=client,
                questions_data=questions_mcq_without_refusal,
                agent_response=mapped_without_refusal,
                model=config.grader_model,
            )
        )

        return BatchThreeModeEvaluation(
            direct_by_question_id=direct_map,
            mcq_with_refusal_by_question_id=mcq_with_refusal_map,
            mcq_without_refusal_by_question_id=mcq_without_refusal_map,
            mapped_with_refusal=mapped_with_refusal,
            mapped_without_refusal=mapped_without_refusal,
            options_with_refusal_by_question_id=options_with_refusal_by_qid,
            options_without_refusal_by_question_id=options_without_refusal_by_qid,
            grader_model=config.grader_model,
            option_chooser_model=config.option_chooser_model,
        )
    except Exception as exc:
        failed_map = _disabled_grade_map(question_ids, f"Judge failed: {exc}")
        return BatchThreeModeEvaluation(
            direct_by_question_id=failed_map,
            mcq_with_refusal_by_question_id=failed_map,
            mcq_without_refusal_by_question_id=failed_map,
            mapped_with_refusal="",
            mapped_without_refusal="",
            options_with_refusal_by_question_id={qid: [] for qid in question_ids},
            options_without_refusal_by_question_id={qid: [] for qid in question_ids},
            grader_model=config.grader_model,
            option_chooser_model=config.option_chooser_model,
            judge_error=str(exc),
        )
