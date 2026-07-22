"""Run the production FAFU ACT preflight or finite hardware rollout."""

from __future__ import annotations

import sys

from lerobot_robot_fafu_arm.inference.cli import act_main

if __name__ == "__main__":
    raise SystemExit(act_main(sys.argv[1:]))
