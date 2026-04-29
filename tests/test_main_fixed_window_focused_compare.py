import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pytest

from main_fixed_window_focused_compare import parse_config_specs


def test_parse_config_specs_accepts_baseline_and_hierarchical_entries():
    specs = parse_config_specs(
        "baseline_sw3|baseline|3|-|-|-|;"
        "cost_selective_sw3_mw4|hierarchical|3|4|cost_aware_selective_route_refill_on_full_accept|utility|baseline_sw3"
    )

    assert len(specs) == 2
    assert specs[0]["run_type"] == "baseline"
    assert specs[0]["middle_window"] is None
    assert specs[1]["run_type"] == "hierarchical"
    assert specs[1]["middle_window"] == 4
    assert specs[1]["hierarchical_variant"] == "cost_aware_selective_route_refill_on_full_accept"
    assert specs[1]["window_policy"] == "utility"
    assert specs[1]["baseline_label"] == "baseline_sw3"


def test_parse_config_specs_rejects_missing_hierarchical_fields():
    with pytest.raises(ValueError):
        parse_config_specs("bad|hierarchical|3|-|selective_route_refill_on_full_accept|utility|baseline_sw3")


def test_parse_config_specs_rejects_duplicate_labels():
    with pytest.raises(ValueError):
        parse_config_specs(
            "dup|baseline|3|-|-|-|;"
            "dup|hierarchical|3|4|double_layer|fixed|dup"
        )
