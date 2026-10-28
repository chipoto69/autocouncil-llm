"""
Langfuse observability wrapper for autocouncil.

Fail-open design: every function silently no-ops if Langfuse is unavailable
or raises an exception. Tracing must never block or break council deliberation.

Uses Langfuse v4 OpenTelemetry-native API: start_observation() creates a trace
and root span; child observations (spans/generations) are created on the root.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_client: Optional[Any] = None  # Langfuse client
_initialized: bool = False


def init_langfuse() -> Optional[Any]:
    """Initialise the Langfuse client from environment variables.

    Reads (in order):
      HERMES_LANGFUSE_SECRET_KEY / LANGFUSE_SECRET_KEY
      HERMES_LANGFUSE_PUBLIC_KEY  / LANGFUSE_PUBLIC_KEY
      HERMES_LANGFUSE_BASE_URL    / LANGFUSE_BASE_URL

    Returns the client, or None if credentials are missing or the import fails.
    """
    global _client, _initialized

    if _initialized:
        return _client

    _initialized = True

    try:
        from langfuse import Langfuse  # type: ignore[import-untyped]
    except ImportError:
        logger.debug("langfuse package not installed – tracing disabled")
        return None

    secret_key = os.environ.get("HERMES_LANGFUSE_SECRET_KEY") or os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = os.environ.get("HERMES_LANGFUSE_PUBLIC_KEY") or os.environ.get("LANGFUSE_PUBLIC_KEY")
    base_url = os.environ.get("HERMES_LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_BASE_URL") or "https://cloud.langfuse.com"

    if not secret_key or not public_key:
        logger.debug("Langfuse credentials not set – tracing disabled")
        return None

    try:
        _client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            base_url=base_url,
        )
        logger.info("Langfuse tracing initialised (base_url=%s)", base_url)
        return _client
    except Exception as exc:
        logger.warning("Langfuse initialisation failed (fail-open): %s", exc)
        return None


def get_langfuse() -> Optional[Any]:
    """Return the cached client, initialising on first call if needed."""
    global _client
    if _client is None and not _initialized:
        init_langfuse()
    return _client


# ---------------------------------------------------------------------------
# Trace context – a lightweight handle wrapping the root observation
# ---------------------------------------------------------------------------


class TraceContext:
    """Holds a Langfuse root observation and the client reference."""

    __slots__ = ("client", "root", "trace_id")

    def __init__(self, client: Any, root: Any) -> None:
        self.client = client
        self.root = root
        self.trace_id: str = root.trace_id


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def trace_council_query(
    query: str,
    models: List[str],
    chairman: str,
    *,
    tags: Optional[List[str]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[TraceContext]:
    """Create a Langfuse trace for a full council deliberation.

    Returns a TraceContext handle, or None if tracing is unavailable.
    """
    client = get_langfuse()
    if client is None:
        return None

    try:
        meta: Dict[str, Any] = {
            "models": models,
            "chairman": chairman,
            "model_count": len(models),
        }
        if metadata:
            meta.update(metadata)

        root = client.start_observation(
            name="autocouncil",
            as_type="chain",
            input={"query": query},
            metadata=meta,
            level="DEFAULT",
        )

        # Attempt to set trace-level tags (private API; safe to fail)
        if tags:
            try:
                all_tags = ["autocouncil", "council"] + tags
                client._create_trace_tags_via_ingestion(
                    trace_id=root.trace_id, tags=all_tags
                )
            except Exception:
                logger.debug("Could not set Langfuse trace tags", exc_info=True)

        return TraceContext(client, root)
    except Exception as exc:
        logger.debug("Failed to create Langfuse trace: %s", exc)
        return None


def trace_stage(
    ctx: Optional[TraceContext],
    stage_name: str,
    model: str,
    input_data: Any = None,
    output_data: Any = None,
    *,
    as_type: str = "generation",
    metadata: Optional[Dict[str, Any]] = None,
) -> None:
    """Record a span for one model's work inside a council stage.

    Args:
        ctx: TraceContext from trace_council_query (None → no-op).
        stage_name: e.g. ``"stage1.openai/gpt-4o"``.
        model: The model ID used for this call.
        input_data: The prompt / messages sent.
        output_data: The response received.
        as_type: Langfuse observation type ('generation', 'span', …).
    """
    if ctx is None:
        return
    try:
        observation = ctx.root.start_observation(
            name=stage_name,
            as_type=as_type,
            input=input_data,
            output=output_data,
            model=model,
            metadata=metadata,
        )
        # These stage results are recorded after the model call has completed.
        # Manual observations are only exportable once they are ended.
        observation.end()
    except Exception as exc:
        logger.debug("Langfuse span creation failed (%s): %s", stage_name, exc)


def trace_convergence(
    ctx: Optional[TraceContext],
    round_num: int,
    converged: bool,
    reason: Optional[str] = None,
    *,
    final_answer: Optional[str] = None,
) -> None:
    """Record convergence result on the root trace."""
    if ctx is None:
        return
    try:
        output_payload: Dict[str, Any] = {
            "converged": converged,
            "total_rounds": round_num,
            "convergence_reason": reason,
        }
        if final_answer:
            output_payload["final_answer"] = final_answer[:500]  # safety truncate

        ctx.root.update(output=output_payload)
        ctx.root.set_trace_io(output=output_payload)
    except Exception as exc:
        logger.debug("Langfuse convergence update failed: %s", exc)


def end_trace(ctx: Optional[TraceContext]) -> None:
    """End the root observation and flush all pending events.

    Call this once per council deliberation after all spans are recorded.
    """
    if ctx is None:
        return
    try:
        ctx.root.end()
    except Exception as exc:
        logger.debug("Langfuse root observation end failed: %s", exc)
    try:
        # Still flush completed children if ending the root failed.
        ctx.client.flush()
        logger.debug("Langfuse trace %s flushed", ctx.trace_id)
    except Exception as exc:
        logger.debug("Langfuse trace flush failed: %s", exc)


def flush() -> None:
    """Flush all pending Langfuse events (non-blocking best-effort)."""
    client = get_langfuse()
    if client is None:
        return
    try:
        client.flush()
    except Exception as exc:
        logger.debug("Langfuse flush failed: %s", exc)

