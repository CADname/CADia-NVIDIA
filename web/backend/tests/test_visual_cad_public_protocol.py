from __future__ import annotations

import pytest

from standalonecad_web.visual_cad_reliability import preflight_plan


class _Executor:
    def validate_plan(self, plan):
        calls = plan.get("calls")
        if not isinstance(calls, list) or not calls:
            raise ValueError("Planner returned no CAD calls")
        return calls


def _plan(*calls):
    return {"calls": list(calls), "note": "test"}


def _call(tool, **arguments):
    return {"tool": tool, "arguments": arguments}


def test_parameter_chain_must_resolve_in_execution_order():
    plan = _plan(
        _call("inventor_new_part"),
        _call("inventor_create_parameter", name="WIDTH", expression="100", unit="mm"),
        _call("inventor_create_parameter", name="HALF", expression="WIDTH / 2", unit="mm"),
    )
    canonical, report = preflight_plan(_Executor(), plan, {"parameters": {}})
    assert canonical["calls"] == plan["calls"]
    assert report.resolved_parameters["WIDTH"] == 100.0
    assert report.resolved_parameters["HALF"] == 50.0


def test_self_reference_is_rejected_before_cad_mutation():
    plan = _plan(
        _call("inventor_new_part"),
        _call("inventor_create_parameter", name="P", expression="P + 1", unit="mm"),
    )
    with pytest.raises(ValueError, match="cannot be resolved"):
        preflight_plan(_Executor(), plan, {"parameters": {}})


def test_forward_reference_is_rejected_before_cad_mutation():
    plan = _plan(
        _call("inventor_new_part"),
        _call("inventor_create_parameter", name="A", expression="B + 1", unit="mm"),
        _call("inventor_create_parameter", name="B", expression="10", unit="mm"),
    )
    with pytest.raises(ValueError, match="cannot be resolved"):
        preflight_plan(_Executor(), plan, {"parameters": {}})


def test_unknown_set_parameter_is_rejected():
    plan = _plan(
        _call("inventor_new_part"),
        _call("inventor_set_parameter", name="MISSING", value="5"),
    )
    with pytest.raises(ValueError, match="unknown parameter"):
        preflight_plan(_Executor(), plan, {"parameters": {}})


def test_existing_parameters_can_be_used_until_new_document_resets_scope():
    state = {
        "parameters": {
            "WIDTH": {"expression": "80 mm", "unit": "mm"},
            "HALF": {"expression": "WIDTH / 2", "unit": "mm"},
        }
    }
    plan = _plan(_call("inventor_set_parameter", name="WIDTH", value="HALF * 3"))
    _, report = preflight_plan(_Executor(), plan, state)
    assert report.resolved_parameters["WIDTH"] == 120.0

    reset_plan = _plan(
        _call("inventor_new_part"),
        _call("inventor_create_parameter", name="X", expression="WIDTH", unit="mm"),
    )
    with pytest.raises(ValueError, match="cannot be resolved"):
        preflight_plan(_Executor(), reset_plan, state)
