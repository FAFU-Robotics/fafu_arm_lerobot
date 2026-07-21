# URDF 与 TCP 推导

本项目的 URDF 只服务于 FK/IK，因此省略 mesh、visual、collision 和 inertia，避免部署时依赖
ROS package URI 或缺失的 STL 文件。

## 六轴运动学链

关节参数来自 FAFU Arm 的机械尺寸与 `fafu_follower.urdf`，由本项目直接维护：

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

提供的 `fafu_follower.urdf` 使用：

```xml
<origin xyz="0.165 0 0" rpy="0 0 0"/>
```

当前 FAFU 夹爪机械尺寸为：

- coupling：沿夹爪局部 Z 轴 0.005 m；
- acting center：沿夹爪局部 Z 轴 0.170 m；
- 总计：0.175 m。

夹爪安装方向沿 URDF link6 的 X 轴，因此更新后的 fixed joint 为：

```xml
<joint name="tool_joint" type="fixed">
  <origin xyz="0.175 0 0" rpy="0 0 0"/>
  <parent link="link6"/>
  <child link="tool_link"/>
</joint>
```

即 TCP 相对旧模型向前移动 10 mm。正式标定后如测量值不同，应复制 URDF、修改该 fixed joint，
并通过 `FafuArmKinematics(custom_urdf)` 或 LeRobot 的 `--robot.urdf_path` 使用。
