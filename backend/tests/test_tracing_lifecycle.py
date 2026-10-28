"""Offline lifecycle tests; no Langfuse credentials or model calls are used."""

import asyncio
import importlib.util
from pathlib import Path
import sys
import threading
import types
import unittest
from unittest.mock import AsyncMock, patch


BACKEND = Path(__file__).resolve().parents[1]
PACKAGE = "_autocouncil_lifecycle_tests"


def load_module(name):
    spec = importlib.util.spec_from_file_location(
        f"{PACKAGE}.{name}", BACKEND / f"{name}.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class Observation:
    def __init__(self, client, **data):
        self.client = client
        self.data = data
        self.trace_id = "offline-trace"
        self.ended = False
        self.end_calls = 0
        self.output = None
        self.trace_output = None

    def start_observation(self, **data):
        return self.client.start_observation(**data)

    def update(self, **data):
        self.output = data.get("output", self.output)

    def set_trace_io(self, **data):
        self.trace_output = data.get("output", self.trace_output)

    def end(self):
        self.end_calls += 1
        self.ended = True


class Client:
    def __init__(self):
        self.observations = []
        self.exported = []
        self.flush_calls = 0

    def start_observation(self, **data):
        observation = Observation(self, **data)
        self.observations.append(observation)
        return observation

    def flush(self):
        self.flush_calls += 1
        # Match the SDK lifecycle: unended observations are not exportable.
        self.exported = [item for item in self.observations if item.ended]


class TracingLifecycleTests(unittest.TestCase):
    def setUp(self):
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(BACKEND)]
        council = types.ModuleType(f"{PACKAGE}.council")
        council.calculate_aggregate_rankings = lambda *_: []
        council.parse_ranking_from_text = lambda *_a, **_k: []
        council.query_model = AsyncMock()
        council.query_models_parallel = AsyncMock()
        settings = types.ModuleType(f"{PACKAGE}.settings")
        settings.get_settings = lambda: None
        # Isolate provider imports so these tests are runnable with stdlib only.
        self.modules = patch.dict(sys.modules, {
            PACKAGE: package,
            f"{PACKAGE}.council": council,
            f"{PACKAGE}.settings": settings,
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.tracing = load_module("tracing")
        self.autocouncil = load_module("autocouncil")
        self.client = Client()
        self.client_patch = patch.object(
            self.tracing, "get_langfuse", return_value=self.client
        )
        self.client_patch.start()
        self.addCleanup(self.client_patch.stop)

    def context(self):
        return self.tracing.trace_council_query("question", ["model"], "chairman")

    def run_council(self, *, rounds=1, converge=False, stage1_error=None):
        async def run():
            with patch.object(
                self.autocouncil, "_autocouncil_stage1",
                new=AsyncMock(
                    return_value=[{"model": "model", "response": "answer", "error": None}],
                    side_effect=stage1_error,
                ),
            ), patch.object(
                self.autocouncil, "_autocouncil_stage2",
                new=AsyncMock(return_value=(
                    {"Response A": "model"},
                    [{"model": "model", "ranking": "Response A", "error": None}],
                )),
            ), patch.object(
                self.autocouncil, "_autocouncil_stage3",
                new=AsyncMock(return_value={
                    "model": "chairman", "response": "final answer", "error": False,
                }),
            ), patch.object(
                self.autocouncil, "_detect_convergence",
                return_value=(converge, "same_top_model" if converge else None),
            ):
                return await self.autocouncil.run_autocouncil(
                    "question", ["model"], "chairman", max_rounds=rounds
                )
        return asyncio.run(run())

    def test_completed_child_is_exportable(self):
        context = self.context()
        self.tracing.trace_stage(
            context, "round.1.stage1.model", "model",
            input_data={"question": "test"}, output_data={"response": "answer"},
        )
        self.tracing.end_trace(context)
        self.assertEqual(len(self.client.exported), 2)
        child = self.client.observations[1]
        self.assertEqual(child.end_calls, 1)
        self.assertEqual(child.data["output"], {"response": "answer"})

    def test_flush_is_attempted_even_when_root_end_fails(self):
        context = self.context()
        self.tracing.trace_stage(context, "completed", "model")
        with patch.object(context.root, "end", side_effect=RuntimeError("end failed")):
            self.tracing.end_trace(context)
        self.assertEqual(self.client.flush_calls, 1)
        self.assertIn(self.client.observations[1], self.client.exported)

    def test_success_exports_all_observations_and_final_output(self):
        result = self.run_council()
        self.assertEqual(result["final_answer"], "final answer")
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(len(self.client.exported), 4)  # root + three stages
        self.assertTrue(all(o.end_calls == 1 for o in self.client.observations))
        self.assertEqual(self.client.flush_calls, 1)
        root = self.client.observations[0]
        self.assertEqual(root.trace_output["final_answer"], result["final_answer"])
        self.assertFalse(root.trace_output["converged"])

    def test_convergence_keeps_result_and_closes_trace(self):
        result = self.run_council(rounds=5, converge=True)
        self.assertTrue(result["converged"])
        self.assertEqual(result["rounds"], 1)
        self.assertEqual(self.client.observations[0].trace_output, {
            "converged": True,
            "total_rounds": 1,
            "convergence_reason": "same_top_model",
            "final_answer": "final answer",
        })
        self.assertEqual(self.client.flush_calls, 1)

    def test_exception_closes_trace_without_hiding_exception(self):
        with self.assertRaisesRegex(RuntimeError, "provider failed"):
            self.run_council(stage1_error=RuntimeError("provider failed"))
        self.assertEqual(self.client.observations[0].end_calls, 1)
        self.assertEqual(self.client.flush_calls, 1)

    def test_cancellation_closes_trace_and_propagates(self):
        with self.assertRaises(asyncio.CancelledError):
            self.run_council(stage1_error=asyncio.CancelledError())
        self.assertEqual(self.client.observations[0].end_calls, 1)
        self.assertEqual(self.client.flush_calls, 1)

    def test_all_models_failed_exports_failure_observations(self):
        with patch.object(
            self.autocouncil, "_autocouncil_stage1",
            new=AsyncMock(return_value=[{
                "model": "model", "response": None, "error": "no_response",
            }]),
        ):
            result = asyncio.run(
                self.autocouncil.run_autocouncil("question", ["model"], "chairman")
            )
        self.assertEqual(result["final_answer"], "")
        self.assertFalse(result["converged"])
        self.assertEqual(len(self.client.exported), 2)
        self.assertEqual(self.client.observations[1].data["output"]["error"], "no_response")

    def test_export_failure_does_not_break_deliberation(self):
        with patch.object(self.client, "flush", side_effect=RuntimeError("offline")):
            result = self.run_council()
        self.assertEqual(result["final_answer"], "final answer")
        self.assertTrue(all(o.ended for o in self.client.observations))

    def test_flush_does_not_block_the_event_loop(self):
        release = threading.Event()
        flush_saw_callback = []

        async def run():
            loop = asyncio.get_running_loop()

            def flush():
                # This callback must run while the synchronous exporter waits.
                loop.call_soon_threadsafe(release.set)
                flush_saw_callback.append(release.wait(timeout=1))

            with patch.object(self.client, "flush", side_effect=flush):
                await self.autocouncil.run_autocouncil(
                    "question", ["model"], "chairman", max_rounds=0
                )

        asyncio.run(run())
        self.assertEqual(flush_saw_callback, [True])

    def test_export_failure_does_not_hide_provider_exception(self):
        with patch.object(self.client, "flush", side_effect=RuntimeError("offline")):
            with self.assertRaisesRegex(ValueError, "provider failed"):
                self.run_council(stage1_error=ValueError("provider failed"))

    def test_empty_models_does_not_create_a_trace(self):
        result = asyncio.run(
            self.autocouncil.run_autocouncil("question", [], "chairman")
        )
        self.assertEqual(result["rounds"], 0)
        self.assertEqual(self.client.observations, [])

    def test_disabled_tracing_preserves_council_result(self):
        with patch.object(self.tracing, "get_langfuse", return_value=None):
            result = self.run_council()
        self.assertEqual(result["final_answer"], "final answer")
        self.assertEqual(self.client.observations, [])


if __name__ == "__main__":
    unittest.main()

