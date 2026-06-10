from pacer_framework import TargetRegion, validate_target_region
from pacer_framework.hooks import RobotHooks, TrainerHooks, VLAHooks


class DummyRobot:
    def target_region(self, component, collection_config):
        return TargetRegion(component=component, target_id=f"{collection_config}:{component}", point=(0.1, 0.0, 0.2), radius=0.01)

    def integrate_action_chunk(self, *, start_tcp, action_chunk, observation):
        x, y, z = start_tcp
        out = [(x, y, z)]
        for dx, dy, dz in action_chunk:
            x, y, z = x + dx, y + dy, z + dz
            out.append((x, y, z))
        return out

    def safety_flags(self, trace_or_prediction):
        return {"unsafe": False, "manual_safety_stop": False, "quarantined": False}


class DummyVLA:
    def predict_action_chunk(self, *, model_id, observation, prompt=None, deterministic=True):
        return [(0.01, 0.0, 0.0), (0.01, 0.0, 0.0)]

    def stop_emitted(self, prediction):
        return True


class DummyTrainer:
    def train_weighted_view(self, *, view_dir, starting_checkpoint, output_model_id, recipe):
        return output_model_id


def test_hook_protocols_define_the_robot_vla_trainer_boundary():
    robot = DummyRobot()
    vla = DummyVLA()
    trainer = DummyTrainer()

    assert isinstance(robot, RobotHooks)
    assert isinstance(vla, VLAHooks)
    assert isinstance(trainer, TrainerHooks)

    region = robot.target_region("widget", "scene_001")
    assert validate_target_region(region) == []
    assert region.as_geometry()["d_ref"] == 0.02

    actions = vla.predict_action_chunk(model_id="candidate_a", observation={"image_id": "obs"})
    positions = robot.integrate_action_chunk(start_tcp=(0.0, 0.0, 0.0), action_chunk=actions, observation={})
    assert positions == [(0.0, 0.0, 0.0), (0.01, 0.0, 0.0), (0.02, 0.0, 0.0)]
    assert vla.stop_emitted(actions) is True
    assert trainer.train_weighted_view(view_dir="views/pacer", starting_checkpoint="base", output_model_id="candidate_a", recipe={}) == "candidate_a"


def test_target_region_validation_rejects_bad_declarations():
    bad = TargetRegion(target_id="", component="", point=(0.0, 0.0), radius=0.0)  # type: ignore[arg-type]
    errors = validate_target_region(bad)
    assert "target_id is required" in errors
    assert "component is required" in errors
    assert "point must be three numeric base-frame coordinates" in errors
    assert "radius must be positive" in errors
