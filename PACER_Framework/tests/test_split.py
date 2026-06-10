import pytest

from pacer_framework.split import assign_case_group_split


def buffer_rows():
    rows = []
    for component, n_configs in (("widget", 4), ("socket", 3)):
        for k in range(n_configs):
            config = f"{component}_cfg{k}"
            for role in ("clean", "correction", "partial"):
                rows.append(
                    {
                        "row_id": f"{config}:{role}",
                        "component": component,
                        "phase": "approach",
                        "role": role,
                        "collection_config": config,
                        "mask": [1] * 5,
                    }
                )
    return rows


def test_split_is_configuration_disjoint_and_whole_group():
    out, manifest = assign_case_group_split(buffer_rows(), seed=7)
    split_by_config = {}
    for row in out:
        split_by_config.setdefault(row["collection_config"], set()).add(row["split"])
    # Every trace under a configuration lands whole on one side.
    assert all(len(splits) == 1 for splits in split_by_config.values())
    for component in ("widget", "socket"):
        plan = manifest["plan"][component]
        assert len(plan["val_configs"]) == 1
        assert set(plan["train_configs"]) | set(plan["val_configs"]) == set(plan["configs"])
        assert not set(plan["train_configs"]) & set(plan["val_configs"])


def test_split_is_deterministic_in_the_seed_and_row_order():
    rows = buffer_rows()
    _out_a, manifest_a = assign_case_group_split(rows, seed=20260610)
    _out_b, manifest_b = assign_case_group_split(list(reversed(rows)), seed=20260610)
    assert manifest_a["plan"] == manifest_b["plan"]


def test_frozen_configs_become_heldout_rows():
    out, manifest = assign_case_group_split(
        buffer_rows(), seed=7, frozen_configs={"widget": ["widget_cfg3"]}
    )
    plan = manifest["plan"]["widget"]
    assert plan["heldout_configs"] == ["widget_cfg3"]
    assert "widget_cfg3" not in plan["val_configs"] + plan["train_configs"]
    frozen_rows = [r for r in out if r["collection_config"] == "widget_cfg3"]
    assert frozen_rows and all(r["split"] == "heldout" for r in frozen_rows)


def test_split_errors():
    rows = buffer_rows()
    rows[0] = dict(rows[0])
    rows[0].pop("collection_config")
    with pytest.raises(ValueError):
        assign_case_group_split(rows)

    single = [
        {"row_id": "only", "component": "widget", "collection_config": "cfg0", "mask": [1]}
    ]
    with pytest.raises(ValueError):
        assign_case_group_split(single)  # cannot leave a training config

    with pytest.raises(ValueError):
        assign_case_group_split(buffer_rows(), frozen_configs={"widget": ["nope"]})
