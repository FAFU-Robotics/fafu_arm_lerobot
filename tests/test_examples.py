from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _load_demo():
    path = Path(__file__).parents[1] / "examples" / "kinematics_demo.py"
    spec = spec_from_file_location("kinematics_demo", path)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_kinematics_demo_default_model(capsys):
    demo = _load_demo()

    assert demo.main([]) == 0
    output = capsys.readouterr().out
    assert "position error:" in output
    assert "rotation error:" in output
    assert "distance from independent IK seed:" in output
    assert "[OK] FK/IK round trip" in output


def test_kinematics_demo_reports_ik_failure(monkeypatch, tmp_path, capsys):
    demo = _load_demo()
    custom_urdf = tmp_path / "custom.urdf"

    class NoSolutionKinematics:
        def __init__(self, urdf):
            assert urdf == custom_urdf

        def forward(self, joints):
            return SimpleNamespace(position=np.zeros(3), rotation=np.eye(3))

        def inverse(self, position, rotation, *, seed):
            return None

    monkeypatch.setattr(demo, "FafuArmKinematics", NoSolutionKinematics)

    assert demo.main(["--urdf", str(custom_urdf)]) == 1
    assert "IK did not find a solution" in capsys.readouterr().out


def test_kinematics_demo_rejects_large_round_trip_error(monkeypatch, capsys):
    demo = _load_demo()

    class InaccurateKinematics:
        def __init__(self, urdf):
            self.forward_calls = 0

        @property
        def joint_limits(self):
            return -np.ones(6), np.ones(6)

        def forward(self, joints):
            self.forward_calls += 1
            position = np.zeros(3) if self.forward_calls == 1 else np.array([1e-3, 0.0, 0.0])
            return SimpleNamespace(position=position, rotation=np.eye(3))

        def inverse(self, position, rotation, *, seed):
            return np.zeros(6)

    monkeypatch.setattr(demo, "FafuArmKinematics", InaccurateKinematics)

    assert demo.main([]) == 1
    output = capsys.readouterr().out
    assert "position error:" in output
    assert "rotation error:" in output
    assert "exceeds tolerance" in output


def test_kinematics_demo_rejects_non_finite_ik_solution(monkeypatch, capsys):
    demo = _load_demo()

    class NonFiniteKinematics:
        def __init__(self, urdf):
            pass

        def forward(self, joints):
            return SimpleNamespace(position=np.zeros(3), rotation=np.eye(3))

        def inverse(self, position, rotation, *, seed):
            assert np.array_equal(seed, np.zeros(6))
            return np.array([np.nan, 0.0, 0.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(demo, "FafuArmKinematics", NonFiniteKinematics)

    assert demo.main([]) == 1
    assert "IK returned NaN or infinity" in capsys.readouterr().out


def test_kinematics_demo_rejects_solution_outside_finite_limits(monkeypatch, capsys):
    demo = _load_demo()

    class OutOfBoundsKinematics:
        def __init__(self, urdf):
            pass

        @property
        def joint_limits(self):
            # Joint 1 is continuous/unbounded; joint 2 has a finite upper bound.
            lower = np.array([-np.inf, -1.0, -1.0, -1.0, -1.0, -1.0])
            upper = np.array([np.inf, 1.0, 1.0, 1.0, 1.0, 1.0])
            return lower, upper

        def forward(self, joints):
            return SimpleNamespace(position=np.zeros(3), rotation=np.eye(3))

        def inverse(self, position, rotation, *, seed):
            assert np.array_equal(seed, np.zeros(6))
            return np.array([100.0, 1.1, 0.0, 0.0, 0.0, 0.0])

    monkeypatch.setattr(demo, "FafuArmKinematics", OutOfBoundsKinematics)

    assert demo.main([]) == 1
    output = capsys.readouterr().out
    assert "violates finite URDF limits" in output
    assert "joint(s): 2" in output
