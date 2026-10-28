"""
Autocouncil: Autonomous multi-round deliberation loop for LLM Council Plus.

Runs the standard 3-stage council repeatedly, feeding each round's synthesis
back as context to the next, until convergence is detected or max rounds reached.

Convergence detection criteria (any one triggers convergence):
  1. Same top-ranked model for 2 consecutive rounds
  2. Chairman answer length stable within 15% for 2 consecutive rounds
  3. Ranking consensus — top model gets 75%+ of aggregate rank points

CLI usage:
  python -m backend.autocouncil "What is the best approach to X?" \
      --models openai/gpt-4o,anthropic/claude-sonnet-4,google/gemini-2.5-pro \
      --chairman google/gemini-2.5-pro \
      --max-rounds 5 --json --quiet
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from .council import (
    calculate_aggregate_rankings,
    parse_ranking_from_text,
    query_model,
    query_models_parallel,
)
from .prompts import (
    STAGE1_PROMPT_DEFAULT,
    STAGE2_PROMPT_DEFAULT,
    STAGE3_PROMPT_DEFAULT,
    STAGE1_SEARCH_CONTEXT_TEMPLATE,
)
from .settings import get_settings

if TYPE_CHECKING:
    from .tracing import TraceContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Convergence detection
# ---------------------------------------------------------------------------


def _top_model(aggregate: List[Dict[str, Any]]) -> Optional[str]:
    """Return the model with the best (lowest) average rank."""
    if not aggregate:
        return None
    return aggregate[0]["model"]


def _check_length_stable(lengths: List[int], window: int = 2) -> bool:
    """True if the last 'window' lengths are all within 15% of each other."""
    if len(lengths) < window:
        return False
    recent = lengths[-window:]
    avg = sum(recent) / len(recent)
    if avg == 0:
        return all(v == 0 for v in recent)
    return all(abs(v - avg) / avg <= 0.15 for v in recent)


def _check_ranking_consensus(
    aggregate: List[Dict[str, Any]], threshold: float = 0.75
) -> bool:
    """
    True if the top model received >= threshold fraction of all ranking
    'points' (i.e., appeared in the parsed rankings of >= 75% of voters).
    """
    if not aggregate:
        return False
    top = aggregate[0]
    total_voters = sum(r["rankings_count"] for r in aggregate)
    if total_voters == 0:
        return False
    return (top["rankings_count"] / total_voters) >= threshold


def _detect_convergence(
    round_idx: int,
    top_models: List[Optional[str]],
    answer_lengths: List[int],
    aggregate: List[Dict[str, Any]],
) -> Tuple[bool, Optional[str]]:
    """Check all three convergence criteria. Returns (converged, reason)."""
    reasons = []

    # 1) Same top-ranked model for 2 consecutive rounds
    if len(top_models) >= 2 and top_models[-1] == top_models[-2]:
        reasons.append("same_top_model")

    # 2) Chairman answer length stable within 15% for 2 consecutive rounds
    if _check_length_stable(answer_lengths, window=2):
        reasons.append("length_stable")

    # 3) Ranking consensus: top model with 75%+ votes
    if _check_ranking_consensus(aggregate):
        reasons.append("ranking_consensus")

    if reasons:
        return True, reasons[0]
    return False, None


# ---------------------------------------------------------------------------
# Autocouncil stages (accept explicit model lists)
# ---------------------------------------------------------------------------

AUTOCUNCIL_STAGE1_PROMPT = """You are a helpful AI assistant.

{previous_context}
{search_context_block}
Question: {user_query}"""

AUTOCUNCIL_STAGE3_PROMPT = """You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then ranked each other's responses. This is part of a multi-round deliberation process.

Original Question: {user_query}

{previous_context}

{search_context_block}
STAGE 1 - Individual Responses:
{stage1_text}

STAGE 2 - Peer Rankings:
{stage2_text}

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their insights
- The peer rankings and what they reveal about response quality
- Any patterns of agreement or disagreement
- The evolution of thinking from previous rounds (if any)

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""


async def _autocouncil_stage1(
    user_query: str,
    models: List[str],
    previous_context: str = "",
    search_context: str = "",
    temperature: float = 0.7,
) -> List[Dict[str, Any]]:
    """Run Stage 1 with explicit model list, returning individual responses."""
    settings = get_settings()

    search_context_block = ""
    if search_context:
        search_context_block = STAGE1_SEARCH_CONTEXT_TEMPLATE.format(
            search_context=search_context
        )

    prev_block = ""
    if previous_context:
        prev_block = f"Previous Round Synthesis:\n{previous_context}\n"

    prompt = AUTOCUNCIL_STAGE1_PROMPT.format(
        user_query=user_query,
        previous_context=prev_block,
        search_context_block=search_context_block,
    )

    messages = [{"role": "user", "content": prompt}]
    council_temp = temperature if temperature else settings.council_temperature

    responses = await query_models_parallel(models, messages)
    results: List[Dict[str, Any]] = []

    for model, response in responses.items():
        if response is None or response.get("error"):
            results.append(
                {
                    "model": model,
                    "response": None,
                    "error": response.get("error") if response else "no_response",
                    "error_message": (
                        response.get("error_message", "Unknown error")
                        if response
                        else "No response received"
                    ),
                }
            )
        else:
            content = response.get("content", "")
            if not isinstance(content, str):
                content = str(content) if content is not None else ""
            results.append(
                {"model": model, "response": content, "error": None}
            )

    return results


async def _autocouncil_stage2(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    previous_context: str = "",
    search_context: str = "",
    temperature: float = 0.3,
) -> Tuple[Dict[str, str], List[Dict[str, Any]]]:
    """Run Stage 2 with explicit model list, returning label map and rankings."""
    settings = get_settings()

    # Only successful models participate in ranking
    successful = [r for r in stage1_results if not r.get("error")]
    if len(successful) < 2:
        return {}, []

    # Build anonymized labels
    labels = [chr(65 + i) for i in range(len(successful))]
    label_to_model = {
        f"Response {label}": r["model"]
        for label, r in zip(labels, successful)
    }

    responses_text = "\n\n".join(
        f"Response {label}:\n{r['response']}"
        for label, r in zip(labels, successful)
    )

    search_context_block = ""
    if search_context:
        search_context_block = f"Context from Web Search:\n{search_context}\n"

    prev_block = ""
    if previous_context:
        prev_block = f"Previous Round Synthesis:\n{previous_context}\n"

    prompt = STAGE2_PROMPT_DEFAULT.format(
        user_query=user_query,
        responses_text=responses_text,
        search_context_block=search_context_block,
    )

    # Inject previous round context by modifying the prompt
    if previous_context:
        prompt = (
            f"The following is from a previous round of council deliberation. "
            f"Consider it as additional context when evaluating responses:\n\n"
            f"{prev_block}\n\n{prompt}"
        )

    messages = [{"role": "user", "content": prompt}]
    stage2_temp = temperature if temperature else settings.stage2_temperature

    successful_models = [r["model"] for r in successful]
    responses = await query_models_parallel(successful_models, messages)

    ranking_results: List[Dict[str, Any]] = []
    for model, response in responses.items():
        if response is None or response.get("error"):
            ranking_results.append(
                {
                    "model": model,
                    "ranking": None,
                    "parsed_ranking": [],
                    "error": response.get("error") if response else "no_response",
                    "error_message": (
                        response.get("error_message", "Unknown error")
                        if response
                        else "No response received"
                    ),
                }
            )
        else:
            full_text = response.get("content", "")
            if not isinstance(full_text, str):
                full_text = str(full_text) if full_text is not None else ""
            parsed = parse_ranking_from_text(full_text, expected_count=len(successful))
            ranking_results.append(
                {
                    "model": model,
                    "ranking": full_text,
                    "parsed_ranking": parsed,
                    "error": None,
                }
            )

    return label_to_model, ranking_results


async def _autocouncil_stage3(
    user_query: str,
    stage1_results: List[Dict[str, Any]],
    stage2_results: List[Dict[str, Any]],
    chairman: str,
    previous_context: str = "",
    search_context: str = "",
    temperature: float = 0.3,
) -> Dict[str, Any]:
    """Run Stage 3 with explicit chairman model, returning synthesis."""
    settings = get_settings()
    chairman_temp = temperature if temperature else settings.chairman_temperature

    stage1_text = "\n\n".join(
        f"Model: {r['model']}\nResponse: {r.get('response', 'No response')}"
        for r in stage1_results
        if r.get("response") is not None
    )

    stage2_text = "\n\n".join(
        f"Model: {r['model']}\nRanking: {r.get('ranking', 'No ranking')}"
        for r in stage2_results
        if r.get("ranking") is not None
    )

    search_context_block = ""
    if search_context:
        search_context_block = f"Context from Web Search:\n{search_context}\n"

    prev_block = ""
    if previous_context:
        prev_block = f"Previous Round Synthesis:\n{previous_context}\n"

    prompt = AUTOCUNCIL_STAGE3_PROMPT.format(
        user_query=user_query,
        previous_context=prev_block,
        stage1_text=stage1_text,
        stage2_text=stage2_text,
        search_context_block=search_context_block,
    )

    messages = [
        {
            "role": "system",
            "content": (
                "You are the Chairman of an LLM Council. Your task is to "
                "synthesize the provided model responses into a single, "
                "comprehensive answer."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    try:
        response = await query_model(chairman, messages, temperature=chairman_temp)

        if response is None or response.get("error"):
            error_msg = (
                response.get("error_message", "Unknown error")
                if response
                else "No response received"
            )
            return {
                "model": chairman,
                "response": f"Error synthesizing final answer: {error_msg}",
                "error": True,
                "error_message": error_msg,
            }

        content = response.get("content") or ""
        reasoning = (
            response.get("reasoning") or response.get("reasoning_details") or ""
        )

        final_response = content
        if reasoning and not content:
            final_response = f"**Reasoning:**\n{reasoning}"
        elif reasoning and content:
            final_response = f"<think>\n{reasoning}\n</think>\n\n{content}"

        if not final_response:
            final_response = "No response generated by the Chairman."

        return {
            "model": chairman,
            "response": final_response,
            "error": False,
        }

    except Exception as e:
        logger.error(f"Unexpected error in autocouncil Stage 3: {e}")
        return {
            "model": chairman,
            "response": f"Error: Unable to generate final synthesis due to unexpected error.",
            "error": True,
            "error_message": str(e),
        }


# ---------------------------------------------------------------------------
# Main autocouncil loop
# ---------------------------------------------------------------------------


async def run_autocouncil(
    query: str,
    models: List[str],
    chairman: str,
    max_rounds: int = 5,
    search_context: str = "",
    progress_callback: Optional[callable] = None,
) -> Dict[str, Any]:
    """
    Run the multi-round autocouncil deliberation loop.

    Each round:
      1. Stage 1: All models answer the query (enriched with previous synthesis)
      2. Stage 2: Models rank each other's responses
      3. Stage 3: Chairman synthesizes a final answer

    After each round, convergence is checked. If converged, the loop stops.

    Args:
        query: The user's question.
        models: List of model IDs for the council.
        chairman: Chairman model ID.
        max_rounds: Maximum number of deliberation rounds (default 5).
        search_context: Optional web search context.
        progress_callback: Optional async callable(phase: str, data: dict).

    Returns:
        Structured result dict.
    """
    if not models:
        return {
            "rounds": 0,
            "converged": False,
            "convergence_reason": None,
            "final_answer": "No council models provided.",
            "answer_history": [],
            "final_rankings": [],
        }

    # --- Langfuse tracing ---
    from .tracing import (
        trace_council_query,
        trace_stage,
        trace_convergence,
        end_trace,
    )
    trace_ctx = trace_council_query(
        query=query,
        models=models,
        chairman=chairman,
        metadata={"max_rounds": max_rounds},
    )

    try:
        # Track convergence state
        top_models: List[Optional[str]] = []
        answer_lengths: List[int] = []
        answer_history: List[Dict[str, Any]] = []
        previous_synthesis = ""

        round_idx = 0
        converged = False
        convergence_reason = None

        await _emit(progress_callback, "autocouncil_start", {"max_rounds": max_rounds, "models": models, "chairman": chairman})

        while round_idx < max_rounds:
            round_idx += 1
            logger.info(
                f"Autocouncil round {round_idx}/{max_rounds} with {len(models)} models"
            )
            await _emit(progress_callback, "round_start", {"round": round_idx})

            # --- Stage 1 ---
            await _emit(progress_callback, "stage1_start", {"round": round_idx})
            stage1 = await _autocouncil_stage1(
                query, models, previous_synthesis, search_context
            )
            await _emit(progress_callback, "stage1_complete", {"round": round_idx, "results": len(stage1)})

            # Trace round's Stage 1
            if trace_ctx is not None:
                for r in stage1:
                    trace_stage(
                        trace_ctx,
                        f"round.{round_idx}.stage1.{r['model']}",
                        r["model"],
                        input_data={"query": query, "previous_context": previous_synthesis},
                        output_data=r,
                    )

            successful_s1 = [r for r in stage1 if not r.get("error")]
            if not successful_s1:
                logger.warning(f"Round {round_idx}: all models failed in Stage 1")
                await _emit(progress_callback, "round_complete", {"round": round_idx, "error": "all_models_failed"})
                break

            # --- Stage 2 ---
            await _emit(progress_callback, "stage2_start", {"round": round_idx})
            label_map, stage2 = await _autocouncil_stage2(
                query, stage1, previous_synthesis, search_context
            )
            await _emit(progress_callback, "stage2_complete", {"round": round_idx, "results": len(stage2)})

            # Trace round's Stage 2
            if trace_ctx is not None:
                for r in stage2:
                    trace_stage(
                        trace_ctx,
                        f"round.{round_idx}.stage2.{r['model']}",
                        r["model"],
                        input_data={"query": query, "label_map": label_map},
                        output_data=r,
                    )

            # --- Aggregate rankings ---
            aggregate = calculate_aggregate_rankings(stage2, label_map)
            top = _top_model(aggregate)
            top_models.append(top)

            # --- Stage 3 ---
            await _emit(progress_callback, "stage3_start", {"round": round_idx})
            stage3 = await _autocouncil_stage3(
                query, stage1, stage2, chairman, previous_synthesis, search_context
            )
            await _emit(progress_callback, "stage3_complete", {"round": round_idx})

            synthesis = stage3.get("response", "")

            # Trace round's Stage 3
            if trace_ctx is not None:
                trace_stage(
                    trace_ctx,
                    f"round.{round_idx}.stage3.chairman",
                    chairman,
                    input_data={"query": query, "previous_context": previous_synthesis},
                    output_data={"response": synthesis, "error": stage3.get("error", False)},
                )
            answer_lengths.append(len(synthesis))
            previous_synthesis = synthesis

            # Record round
            answer_history.append(
                {
                    "round": round_idx,
                    "stage1_results": stage1,
                    "stage2_results": stage2,
                    "label_to_model": label_map,
                    "aggregate_rankings": aggregate,
                    "synthesis": synthesis,
                    "chairman_model": chairman,
                }
            )

            await _emit(progress_callback, "round_complete", {"round": round_idx, "answer_length": len(synthesis)})

            # --- Convergence check ---
            converged, convergence_reason = _detect_convergence(
                round_idx, top_models, answer_lengths, aggregate
            )

            if converged:
                logger.info(
                    f"Autocouncil converged at round {round_idx}: {convergence_reason}"
                )
                await _emit(progress_callback, "converged", {"round": round_idx, "reason": convergence_reason})
                break

        if not converged:
            logger.info(
                f"Autocouncil reached max rounds ({max_rounds}) without convergence"
            )

        # Build final response from the last round
        final_answer = previous_synthesis if answer_history else ""
        final_rankings = answer_history[-1]["aggregate_rankings"] if answer_history else []

        # Record the outcome for both convergence and max-round/all-model failure exits.
        trace_convergence(
            trace_ctx,
            round_idx,
            converged=converged,
            reason=convergence_reason,
            final_answer=final_answer,
        )

        return {
            "rounds": round_idx,
            "converged": converged,
            "convergence_reason": convergence_reason,
            "final_answer": final_answer,
            "answer_history": answer_history,
            "final_rankings": final_rankings,
        }
    finally:
        # Preserve partial traces on exceptions and asyncio cancellation as well.
        # The SDK flush is synchronous; keep exporter I/O off the event loop.
        if trace_ctx is not None:
            await asyncio.to_thread(end_trace, trace_ctx)


async def _emit(callback, phase: str, data: dict):
    """Safely invoke the progress callback."""
    if callback:
        try:
            if asyncio.iscoroutinefunction(callback):
                await callback(phase, data)
            else:
                callback(phase, data)
        except Exception as e:
            logger.error(f"Progress callback error: {e}")


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------


def _build_cli_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autocouncil — autonomous multi-round deliberation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m backend.autocouncil "What is the meaning of life?" --json
  python -m backend.autocouncil "Best Python web framework?" \\
      --models openai/gpt-4o,anthropic/claude-sonnet-4 \\
      --chairman google/gemini-2.5-pro --max-rounds 3 --quiet
        """,
    )
    p.add_argument("query", help="The question to deliberate")
    p.add_argument(
        "--models",
        default="",
        help="Comma-separated council model IDs (default: from settings)",
    )
    p.add_argument(
        "--chairman",
        default="",
        help="Chairman model ID (default: from settings)",
    )
    p.add_argument(
        "--max-rounds",
        type=int,
        default=5,
        help="Maximum deliberation rounds (default: 5)",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Output full structured result as JSON",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )
    return p


async def _cli_main() -> None:
    parser = _build_cli_parser()
    args = parser.parse_args()

    from .config import get_council_models, get_chairman_model

    # Resolve models
    if args.models:
        models = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        models = get_council_models()

    chairman = args.chairman or get_chairman_model()

    if not models:
        print("Error: no council models available. Provide --models or configure settings.", file=sys.stderr)
        sys.exit(1)

    if not chairman:
        print("Error: no chairman model available. Provide --chairman or configure settings.", file=sys.stderr)
        sys.exit(1)

    # Progress handler for CLI
    if not args.quiet:

        def _progress(phase: str, data: dict) -> None:
            friendly = {
                "autocouncil_start": f"Starting autocouncil: {len(data.get('models',[]))} models, {data.get('max_rounds',0)} max rounds",
                "round_start": f"\n--- Round {data.get('round')} ---",
                "stage1_start": f"  Stage 1: Collecting responses...",
                "stage1_complete": f"  Stage 1: {data.get('results',0)} responses received",
                "stage2_start": f"  Stage 2: Collecting rankings...",
                "stage2_complete": f"  Stage 2: {data.get('results',0)} rankings received",
                "stage3_start": f"  Stage 3: Chairman synthesizing...",
                "stage3_complete": f"  Stage 3: Complete",
                "round_complete": f"  Round complete (answer: {data.get('answer_length',0)} chars)",
                "converged": f"\n=== CONVERGED at round {data.get('round')}: {data.get('reason')} ===",
            }
            msg = friendly.get(phase, f"{phase}: {json.dumps(data)}")
            print(msg, flush=True)

        progress_callback = _progress
    else:
        progress_callback = None

    result = await run_autocouncil(
        query=args.query,
        models=models,
        chairman=chairman,
        max_rounds=args.max_rounds,
        progress_callback=progress_callback,
    )

    if args.json:
        # Strip large intermediate data for CLI readability, but include full history
        serializable = {
            "rounds": result["rounds"],
            "converged": result["converged"],
            "convergence_reason": result["convergence_reason"],
            "final_answer": result["final_answer"],
            "final_rankings": result["final_rankings"],
            # Include truncated history summaries
            "answer_history": [
                {
                    "round": h["round"],
                    "synthesis": h["synthesis"],
                    "aggregate_rankings": h["aggregate_rankings"],
                }
                for h in result["answer_history"]
            ],
        }
        print(json.dumps(serializable, indent=2))
    else:
        print(f"\n{'='*60}")
        print(f"Autocouncil complete — {result['rounds']} round(s)")
        print(f"Converged: {result['converged']}")
        if result["converged"]:
            print(f"Reason: {result['convergence_reason']}")
        print(f"\n--- Final Rankings ---")
        for r in result["final_rankings"]:
            print(f"  {r['model']}: avg rank {r['average_rank']} ({r['rankings_count']} voters)")
        print(f"\n--- Final Answer ---")
        print(result["final_answer"])


if __name__ == "__main__":
    asyncio.run(_cli_main())

