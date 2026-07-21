"""FK -> IK round-trip without connecting to hardware."""

import numpy as np

from lerobot_robot_fafu_arm import FafuArmKinematics


def main() -> None:
    kinematics = FafuArmKinematics()
    joints = np.array([0.0, 0.5, 1.0, 0.0, 0.0, 0.0])
    pose = kinematics.forward(joints)
    solved = kinematics.inverse(pose.position, pose.rotation, seed=joints)

    print("TCP position (m):", pose.position)
    print("TCP rotation:\n", pose.rotation)
    print("IK solution (rad):", solved)


if __name__ == "__main__":
    main()
