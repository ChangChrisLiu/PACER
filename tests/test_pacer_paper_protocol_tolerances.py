from pathlib import Path

from tracevla.collection.reference_cache import PlannerReferenceCache
from tracevla.collection.schemas import TargetPoint, TargetRegion
from tracevla.weighting.action_chunk_compiler import _pose_error_to_region


def _target_point():
    return TargetPoint(
        tcp_pose=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        joint_positions=[0.0] * 6,
        gripper_obs=0,
        timestamp=0.0,
    )


def test_target_region_default_tolerance_matches_paper_by_component():
    cpu = TargetRegion(config_id="cfg", component="CPU", points=[_target_point()])
    ram = TargetRegion(config_id="cfg", component="RAM", points=[_target_point()])

    assert cpu.position_tolerance_m == 0.005
    assert ram.position_tolerance_m == 0.010
    assert cpu.rotation_tolerance_rad == 0.35
    assert ram.rotation_tolerance_rad == 0.35


def test_reference_cache_backfills_missing_position_tolerance_by_component(tmp_path: Path):
    cache_path = tmp_path / "planner_reference_cache.json"
    cache_path.write_text(
        '{"schema_version": "planner_reference_cache.v1", "config_id": "cfg", "components": {'
        '"CPU": {"groups": {"group_000": {"target_region": {'
        '"config_id": "cfg", "component": "CPU", "points": []}}}},'
        '"RAM": {"groups": {"group_000": {"target_region": {'
        '"config_id": "cfg", "component": "RAM", "points": []}}}}'
        '}}'
    )

    cache = PlannerReferenceCache(tmp_path)

    assert cache.get("CPU", "group_000").target_region.position_tolerance_m == 0.005
    assert cache.get("RAM", "group_000").target_region.position_tolerance_m == 0.010


def test_action_chunk_compiler_missing_tolerance_uses_component_protocol_default():
    target = {
        "component": "CPU",
        "points": [{"tcp_pose": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}],
    }

    inside_cpu = _pose_error_to_region([0.0049, 0.0, 0.0, 0.0, 0.0, 0.0], target)
    outside_cpu = _pose_error_to_region([0.0051, 0.0, 0.0, 0.0, 0.0, 0.0], target)

    assert inside_cpu["inside_region"] is True
    assert outside_cpu["inside_region"] is False

    target["component"] = "RAM"
    inside_ram = _pose_error_to_region([0.0099, 0.0, 0.0, 0.0, 0.0, 0.0], target)
    outside_ram = _pose_error_to_region([0.0101, 0.0, 0.0, 0.0, 0.0, 0.0], target)

    assert inside_ram["inside_region"] is True
    assert outside_ram["inside_region"] is False
