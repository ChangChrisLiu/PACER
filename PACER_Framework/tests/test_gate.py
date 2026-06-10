from pacer_framework.gate import eligibility_gate, mask_has_mass


def valid_row(**over):
    row = {
        "row_id": "gate",
        "component": "ram",
        "phase": "approach",
        "split": "train",
        "role": "partial",
        "mask": [1] * 10,
        "safety": {"unsafe": False, "manual_safety_stop": False, "quarantined": False},
        "provenance": {
            "manifest_valid": True,
            "timing_aligned": True,
            "component_identity_valid": True,
            "action_mask_valid": True,
        },
    }
    row.update(over)
    return row


def test_fully_valid_row_passes_all_factors():
    ok, reasons = eligibility_gate(valid_row())
    assert ok is True
    assert reasons == []


def test_split_factor():
    ok, reasons = eligibility_gate(valid_row(split="val"))
    assert not ok and "gate_split_not_train" in reasons
    ok, reasons = eligibility_gate(valid_row(split="val"), require_train_split=False)
    assert ok


def test_mask_factor_requires_nonzero_valid_mass():
    assert mask_has_mass([1, 0, 1]) is True
    assert mask_has_mass([]) is False
    assert mask_has_mass(None) is False
    assert mask_has_mass([0] * 10) is False
    assert mask_has_mass([1, -1, 1]) is False
    for bad_mask in ([], [0] * 10, [1, -1]):
        ok, reasons = eligibility_gate(valid_row(mask=bad_mask))
        assert not ok and "gate_mask_no_valid_mass" in reasons


def test_provenance_factor():
    row = valid_row()
    row["provenance"]["timing_aligned"] = False
    ok, reasons = eligibility_gate(row)
    assert not ok and "gate_provenance_untrusted" in reasons


def test_safety_factor():
    for key in ("unsafe", "manual_safety_stop", "quarantined"):
        row = valid_row()
        row["safety"][key] = True
        ok, reasons = eligibility_gate(row)
        assert not ok and "gate_unsafe" in reasons, key


def test_role_factor_blocks_failure_and_excluded_in_both_vocabularies():
    for role in ("failure", "model_failure", "excluded"):
        row = valid_row()
        row.pop("role")
        row["sample_role"] = role
        ok, reasons = eligibility_gate(row)
        assert not ok and "gate_role_blocked_from_positive_imitation" in reasons, role
