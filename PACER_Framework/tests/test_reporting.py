import pytest

from pacer_framework.reporting import (
    candidate_ranking_table,
    paired_component_bootstrap,
    success_table,
    wilson_interval,
)

# Synthetic counts only: these are documentation-quality checks of the reporting
# formulas, not paper-dataset result reproduction.
SYNTHETIC_WILSON_CASES = {
    (0, 10): (0.0, 27.8),
    (5, 10): (23.7, 76.3),
    (8, 10): (49.0, 94.3),
    (18, 24): (55.1, 88.0),
    (30, 40): (59.8, 85.8),
}

PACER_SYNTH = {"task_a": (9, 10), "task_b": (7, 10), "task_c": (8, 10)}
VANILLA_SYNTH = {"task_a": (5, 10), "task_b": (6, 10), "task_c": (5, 10)}
UNIFORM_SYNTH = {"task_a": (6, 10), "task_b": (6, 10), "task_c": (7, 10)}


def test_wilson_interval_matches_independent_synthetic_cases():
    for (successes, trials), expected in SYNTHETIC_WILSON_CASES.items():
        lo, hi = wilson_interval(successes, trials)
        assert (round(lo * 100, 1), round(hi * 100, 1)) == expected, (successes, trials)


def test_wilson_interval_input_validation():
    with pytest.raises(ValueError):
        wilson_interval(1, 0)
    with pytest.raises(ValueError):
        wilson_interval(5, 4)


def test_success_table_aggregates_components_and_attaches_wilson_ci():
    table = success_table({"pacer_selected": PACER_SYNTH, "vanilla_post_sft": VANILLA_SYNTH})
    by_method = {row["method"]: row for row in table}
    pacer = by_method["pacer_selected"]
    assert pacer["successes"] == 24 and pacer["trials"] == 30
    assert pacer["ci95_pct"] == (62.7, 90.5)
    assert pacer["per_component"]["task_b"] == "7/10"
    assert by_method["vanilla_post_sft"]["successes"] == 16


def test_paired_component_bootstrap_reports_synthetic_point_differences():
    result = paired_component_bootstrap(PACER_SYNTH, VANILLA_SYNTH, n_boot=2000, seed=11)
    assert result["difference_pp"] == pytest.approx(26.666667, abs=1e-6)
    lo, hi = result["ci95_pp"]
    assert lo <= result["difference_pp"] <= hi

    weaker_gap = paired_component_bootstrap(PACER_SYNTH, UNIFORM_SYNTH, n_boot=2000, seed=11)
    assert weaker_gap["difference_pp"] == pytest.approx(16.666667, abs=1e-6)


def test_paired_component_bootstrap_is_deterministic_and_paired():
    a = paired_component_bootstrap(PACER_SYNTH, VANILLA_SYNTH, n_boot=500, seed=3)
    b = paired_component_bootstrap(PACER_SYNTH, VANILLA_SYNTH, n_boot=500, seed=3)
    assert a == b
    same = paired_component_bootstrap(PACER_SYNTH, PACER_SYNTH, n_boot=500, seed=3)
    assert same["difference_pp"] == 0.0
    assert same["ci95_pp"] == (0.0, 0.0)  # pairing removes cluster variance
    with pytest.raises(ValueError):
        paired_component_bootstrap(PACER_SYNTH, {"task_a": (1, 10)})


def test_candidate_ranking_table_shape_and_order():
    j_vals = {"a": 0.70, "b": 0.90, "c": 0.90}
    feasible = {"a": True, "b": True, "c": False}
    table = candidate_ranking_table(j_vals, feasible, selected="b")
    assert [row["candidate"] for row in table] == ["b", "c", "a"]  # tie broken by id
    assert table[0]["selected"] is True and table[0]["feasible"] is True
    assert table[1]["selected"] is False and table[1]["feasible"] is False
