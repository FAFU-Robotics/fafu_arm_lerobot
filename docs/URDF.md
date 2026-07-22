# URDF 与 TCP 推导

本项目的 URDF 只服务于 FK/IK，因此省略 mesh、visual、collision 和 inertia，避免部署时依赖
ROS package URI 或缺失的 STL 文件。运行时使用的文件是
`src/lerobot_robot_fafu_arm/resources/fafu_arm.urdf`。

## 六轴运动学链

关节参数来自 FAFU Arm 的机械尺寸与 SDK submodule 中固定版本的
`third_party/fafu_arm_sdk/fafu_robot_python/fafu_robot_description/fafu_follower.urdf`，由本项目直接维护：

| Joint | Origin xyz (m) | Axis | Limit (rad) |
|---|---|---|---|
| joint1 | `0 0 0.0584` | `0 0 1` | `[-2.4, 2.4]` |
| joint2 | `0.018199 0 0.053` | `0 1 0` | `[0.0, 3.2]` |
| joint3 | `-0.26 0 0` | `0 -1 0` | `[0.0, 4.0]` |
| joint4 | `0.23 0 0.06` | `0 -1 0` | `[-1.6, 1.6]` |
| joint5 | `0.07 0 0.036319` | `0 0 -1` | `[-1.7, 1.7]` |
| joint6 | `0.02345 0 -0.039` | `1 0 0` | `[-2.5, 2.5]` |

FAFU Arm 的第六轴绕腕部前向轴旋转，因此 URDF 中 joint6 使用 `axis="1 0 0"`。所有 FK/IK
均以 `base_link -> tool_link` 为同一条运动学链。

## TCP 更新

SDK submodule 中的 `fafu_follower.urdf` 使用：

```xml
<origin xyz="0.165 0 0" rpy="0 0 0"/>
```

当前 TCP 长度为 `0.005 m coupling + 0.170 m acting center = 0.175 m`。机械尺寸沿夹爪局部
Z 轴测量；夹爪安装后该方向对应 URDF `link6` 的 X 轴，因此更新后的 fixed joint 为：

```xml
<joint name="tool_joint" type="fixed">
  <origin xyz="0.175 0 0" rpy="0 0 0"/>
  <parent link="link6"/>
  <child link="tool_link"/>
</joint>
```

即 TCP 相对旧模型向前移动 10 mm。正式标定后如测量值不同，应复制 URDF、修改该 fixed joint，
并通过 `FafuArmKinematics(custom_urdf)` 或 LeRobot 的 `--robot.urdf_path` 使用。

## 验证

```bash
fafu-arm-check
python -m pytest tests/test_kinematics.py
```

以上检查验证 URDF 可加载、joint6 轴和 0.175 m TCP；真机尺寸、零位和工作空间仍应按
[Data Collection 指南](DATA_COLLECTION.md)进行只读检查、WRS 观察和低速验证。
