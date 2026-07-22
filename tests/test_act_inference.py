from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot_robot_fafu_arm.inference.act import (
    ActPolicyRuntime,
    build_synthetic_observation,
    derive_observation_settings,
    run_control_loop,
    validate_observation,
    validate_policy_schema,
    validate_robot_schema,
)
from lerobot_robot_fafu_arm.inference.manifest import InferenceManifest, InferenceManifestError
from lerobot_robot_fafu_arm.kinematics import default_urdf_path
from lerobot_robot_fafu_arm.representation import EE_COMPONENTS, JOINT_NAMES, action_features


def make_manifest(mode="joint", state="joint"):
    joint_pos = [f"{name}.pos" for name in JOINT_NAMES] + ["gripper.pos"]
    joint_vel = [f"{name}.vel" for name in JOINT_NAMES] + ["gripper.vel"]
    ee_pose = [f"ee.{name}" for name in EE_COMPONENTS]
    ee_delta = [f"ee_delta.{name}" for name in EE_COMPONENTS]
    state_names = {
        "joint": joint_pos + joint_vel,
        "ee_pose": ee_pose + ["gripper.pos"],
        "all": joint_pos + joint_vel + ee_pose + ee_delta,
    }[state]
    return InferenceManifest(
        action_mode=mode,
        robot_type="fafu_follower",
        fps=30,
        features={
            "observation.state": {
                "dtype": "float32",
                "shape": [len(state_names)],
                "names": state_names,
            },
            "observation.images.front": {
                "dtype": "video",
                "shape": [8, 10, 3],
                "names": ["height", "width", "channels"],
            },
            "action": {
                "dtype": "float32",
                "shape": [7],
                "names": list(action_features(mode)),
            },
        },
    )


@pytest.mark.parametrize(
    ("state", "expected"),
    [
        ("joint", ("joint", True, False)),
        ("ee_pose", ("ee_pose", False, False)),
        ("all", ("all", True, False)),
    ],
)
def test_derive_observation_settings(state, expected):
    assert derive_observation_settings(make_manifest(state=state)) == expected


def test_policy_schema_requires_exact_keys_and_shapes():
    manifest = make_manifest()
    config = SimpleNamespace(
        input_features={
            "observation.state": SimpleNamespace(shape=(14,)),
            "observation.images.front": SimpleNamespace(shape=(3, 8, 10)),
        },
        output_features={"action": SimpleNamespace(shape=(7,))},
    )

    validate_policy_schema(config, manifest)
    config.input_features["observation.state"] = SimpleNamespace(shape=(7,))
    with pytest.raises(ValueError, match="input features differ"):
        validate_policy_schema(config, manifest)


def fake_robot(manifest):
    observation_features = {name: float for name in manifest.state_names}
    observation_features["front"] = (8, 10, 3)
    config = SimpleNamespace(
        action_mode=manifest.action_mode,
        strict_action_features=True,
        use_servo=True,
        servo_rate_hz=30,
        cameras={"front": SimpleNamespace(fps=30)},
    )
    return SimpleNamespace(
        name="fafu_follower",
        action_features=action_features(manifest.action_mode),
        observation_features=observation_features,
        config=config,
        kinematics=SimpleNamespace(
            urdf_path=default_urdf_path(), base_link="base_link", tip_link="tool_link"
        ),
    )


def test_robot_schema_checks_semantics_cameras_fps_and_robot_type():
    manifest = make_manifest(mode="ee_delta", state="all")
    robot = fake_robot(manifest)

    validate_robot_schema(robot, manifest, fps=30)

    robot.observation_features["front"] = (10, 8, 3)
    with pytest.raises(ValueError, match="camera keys/shapes"):
        validate_robot_schema(robot, manifest, fps=30)
    robot.observation_features["front"] = (8, 10, 3)
    robot.name = "fafu_follower"
    robot.config.cameras["front"].fps = None
    with pytest.raises(ValueError, match="camera 'front' fps must equal"):
        validate_robot_schema(robot, manifest, fps=30)

    robot.name = "different_robot"
    with pytest.raises(ValueError, match="robot type"):
        validate_robot_schema(robot, manifest, fps=30)
    robot.name = "fafu_follower"
    with pytest.raises(ValueError, match="must equal training fps"):
        validate_robot_schema(robot, manifest, fps=20)


def test_runtime_uses_processors_and_manifest_action_order():
    torch = pytest.importorskip("torch")
    manifest = make_manifest()

    class Resettable:
        def __init__(self, function=lambda value: value):
            self.function = function
            self.resets = 0

        def __call__(self, value):
            return self.function(value)

        def reset(self):
            self.resets += 1

    class Policy(Resettable):
        config = SimpleNamespace(use_amp=False, type="act")

        def select_action(self, observation):
            assert observation["prepared"]
            return torch.arange(7, dtype=torch.float32).unsqueeze(0)

    policy = Policy()
    preprocessor = Resettable()
    postprocessor = Resettable(lambda value: value + 0.5)

    def build_frame(**kwargs):
        assert kwargs["task"] == "pick"
        return {"prepared": True}

    def make_action(tensor, features):
        return {name: float(tensor[0, index]) for index, name in enumerate(features["action"]["names"])}

    runtime = ActPolicyRuntime(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        manifest=manifest,
        device=torch.device("cpu"),
        task="pick",
        torch_module=torch,
        build_inference_frame=build_frame,
        make_robot_action=make_action,
    )

    action = runtime.predict(build_synthetic_observation(manifest))

    assert tuple(action) == manifest.action_names
    assert list(action.values()) == pytest.approx([0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    runtime.reset()
    assert policy.resets == preprocessor.resets == postprocessor.resets == 1


def test_runtime_load_rejects_an_unbound_checkpoint_before_loading_weights(tmp_path):
    checkpoint = tmp_path / "pretrained_model"
    checkpoint.mkdir()

    with pytest.raises(InferenceManifestError, match="cryptographic file binding"):
        ActPolicyRuntime.load(checkpoint, make_manifest(), device="cpu")


class FakeTime:
    def __init__(self):
        self.now = 0.0

    def clock(self):
        return self.now

    def sleep(self, duration):
        self.now += duration


class FakeRuntime:
    def __init__(self, manifest, *, fail=False):
        self.manifest = manifest
        self.fail = fail
        self.resets = 0

    def reset(self):
        self.resets += 1

    def predict(self, observation):
        if self.fail:
            raise RuntimeError("policy failed")
        return {name: 0.0 for name in self.manifest.action_names}


class FakeRobot:
    name = "fafu_follower"

    def __init__(self, manifest):
        template = fake_robot(manifest)
        self.action_features = template.action_features
        self.observation_features = template.observation_features
        self.config = template.config
        self.kinematics = template.kinematics
        self.manifest = manifest
        self.is_connected = False
        self.disconnects = 0
        self.actions = []

    def connect(self):
        self.is_connected = True

    def get_observation(self):
        return build_synthetic_observation(self.manifest)

    def send_action(self, action):
        self.actions.append(action)
        return action

    def disconnect(self):
        self.is_connected = False
        self.disconnects += 1


def test_control_loop_is_finite_and_always_disconnects():
    manifest = make_manifest()
    runtime = FakeRuntime(manifest)
    robot = FakeRobot(manifest)
    timer = FakeTime()

    report = run_control_loop(
        robot,
        runtime,
        fps=30,
        duration_s=0.1,
        clock=timer.clock,
        sleeper=timer.sleep,
    )

    assert report.steps == 3
    assert len(robot.actions) == 3
    assert robot.disconnects == 1
    assert not robot.is_connected
    assert runtime.resets == 2


def test_control_loop_disconnects_when_policy_fails():
    manifest = make_manifest()
    runtime = FakeRuntime(manifest, fail=True)
    robot = FakeRobot(manifest)

    with pytest.raises(RuntimeError, match="policy failed"):
        run_control_loop(robot, runtime, fps=30, duration_s=1)

    assert robot.disconnects == 1
    assert not robot.actions


@pytest.mark.parametrize(
    ("shape", "nonfinite", "message"),
    [
        ((7,), False, "must have shape"),
        ((1, 7), True, "contains NaN or infinity"),
    ],
)
def test_runtime_rejects_invalid_policy_output(shape, nonfinite, message):
    torch = pytest.importorskip("torch")
    manifest = make_manifest()

    class Resettable:
        def __call__(self, value):
            return value

        def reset(self):
            pass

    class Policy(Resettable):
        config = SimpleNamespace(use_amp=False, type="act")

        def select_action(self, observation):
            action = torch.zeros(shape, dtype=torch.float32)
            if nonfinite:
                action[0, 0] = float("nan")
            return action

    def make_action(tensor, features):
        return {name: float(tensor[0, index]) for index, name in enumerate(features["action"]["names"])}

    runtime = ActPolicyRuntime(
        policy=Policy(),
        preprocessor=Resettable(),
        postprocessor=Resettable(),
        manifest=manifest,
        device=torch.device("cpu"),
        task="",
        torch_module=torch,
        build_inference_frame=lambda **kwargs: {},
        make_robot_action=make_action,
    )

    with pytest.raises(RuntimeError, match=message):
        runtime.predict(build_synthetic_observation(manifest))


def test_control_loop_rejects_initial_state_before_sending_any_action():
    manifest = make_manifest()
    runtime = FakeRuntime(manifest)
    robot = FakeRobot(manifest)

    def reject_start(observation):
        raise RuntimeError("initial state is outside the authorized envelope")

    with pytest.raises(RuntimeError, match="outside the authorized envelope"):
        run_control_loop(
            robot,
            runtime,
            fps=30,
            duration_s=1,
            initial_observation_validator=reject_start,
        )

    assert not robot.actions
    assert robot.disconnects == 1


def test_control_loop_stops_after_consecutive_overruns():
    manifest = make_manifest()
    robot = FakeRobot(manifest)
    timer = FakeTime()

    class SlowRuntime(FakeRuntime):
        def predict(self, observation):
            timer.now += 0.04
            return super().predict(observation)

    with pytest.raises(RuntimeError, match="deadline"):
        run_control_loop(
            robot,
            SlowRuntime(manifest),
            fps=30,
            duration_s=1,
            max_consecutive_overruns=2,
            clock=timer.clock,
            sleeper=timer.sleep,
        )

    assert robot.disconnects == 1
    assert len(robot.actions) == 2


def test_control_loop_disconnects_on_keyboard_interrupt():
    manifest = make_manifest()
    robot = FakeRobot(manifest)

    class InterruptedRuntime(FakeRuntime):
        def predict(self, observation):
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_control_loop(robot, InterruptedRuntime(manifest), fps=30, duration_s=1)

    assert robot.disconnects == 1
    assert not robot.actions


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("state_nan", "contains NaN or infinity"),
        ("camera_shape", "returned shape"),
        ("camera_integer_dtype", "dtype must be"),
        ("camera_float_range", "finite and in"),
        ("extra_field", "fields differ"),
    ],
)
def test_observation_validation_rejects_bad_live_frames(mutation, message):
    manifest = make_manifest()
    observation = build_synthetic_observation(manifest)
    if mutation == "state_nan":
        observation[manifest.state_names[0]] = float("nan")
    elif mutation == "camera_shape":
        observation["front"] = observation["front"][:-1]
    elif mutation == "camera_integer_dtype":
        observation["front"] = observation["front"].astype("int16")
    elif mutation == "camera_float_range":
        observation["front"] = observation["front"].astype("float32")
        observation["front"][0, 0, 0] = 2.0
    else:
        observation["unexpected"] = 0.0

    with pytest.raises(RuntimeError, match=message):
        validate_observation(observation, manifest)


def test_control_loop_stops_on_nonconsecutive_overruns_within_one_second():
    manifest = make_manifest()
    robot = FakeRobot(manifest)
    timer = FakeTime()

    class IntermittentRuntime(FakeRuntime):
        def __init__(self, runtime_manifest):
            super().__init__(runtime_manifest)
            self.calls = 0

        def predict(self, observation):
            self.calls += 1
            if self.calls % 2:
                timer.now += 0.04
            return super().predict(observation)

    with pytest.raises(RuntimeError, match="within one second"):
        run_control_loop(
            robot,
            IntermittentRuntime(manifest),
            fps=30,
            duration_s=1,
            max_consecutive_overruns=10,
            max_overruns_per_second=3,
            clock=timer.clock,
            sleeper=timer.sleep,
        )

    assert robot.disconnects == 1
    assert len(robot.actions) == 5


def test_rollout_duration_starts_after_connection():
    manifest = make_manifest()
    timer = FakeTime()

    class SlowConnectRobot(FakeRobot):
        def connect(self):
            super().connect()
            timer.now += 5.0

    robot = SlowConnectRobot(manifest)
    report = run_control_loop(
        robot,
        FakeRuntime(manifest),
        fps=30,
        duration_s=0.1,
        clock=timer.clock,
        sleeper=timer.sleep,
    )

    assert report.steps == 3
    assert report.elapsed_s == pytest.approx(0.1)
    assert robot.disconnects == 1


def test_rolling_overrun_budget_uses_elapsed_time_not_frame_count():
    manifest = make_manifest()
    robot = FakeRobot(manifest)
    timer = FakeTime()

    class SpacedOverrunRuntime(FakeRuntime):
        def __init__(self, runtime_manifest):
            super().__init__(runtime_manifest)
            self.calls = 0

        def predict(self, observation):
            self.calls += 1
            if self.calls % 2:
                timer.now += 0.04
            return super().predict(observation)

    def oversleep(delay):
        timer.now += delay + 1.1

    report = run_control_loop(
        robot,
        SpacedOverrunRuntime(manifest),
        fps=30,
        duration_s=4,
        max_consecutive_overruns=10,
        max_overruns_per_second=3,
        clock=timer.clock,
        sleeper=oversleep,
    )

    assert report.steps >= 5
    assert report.overruns >= 3
    assert robot.disconnects == 1
