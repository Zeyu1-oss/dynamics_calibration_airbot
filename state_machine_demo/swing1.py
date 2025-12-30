#!/usr/bin/env python3
"""
摆动示例程序 - 演示不同控制模式和更新模式下的摆动运动

参考 examples/cpp/swing_example.cpp
"""

import time
import math
import argparse
import os
import tempfile
from pathlib import Path
from airbot_state_machine import robotic_arm


def prepare_urdf(urdf_path: str) -> str:
    """修复URDF中的相对mesh路径，返回修复后的URDF文件路径（可能是临时文件）。
    
    如果URDF中包含相对路径 '../meshes/'，会将其替换为绝对路径。
    """
    urdf = Path(urdf_path)
    if not urdf.exists():
        return urdf_path
    
    try:
        content = urdf.read_text(encoding="utf-8")
    except Exception:
        return urdf_path
    
    # 检查是否包含相对路径引用
    marker = "../meshes/"
    if marker not in content:
        return urdf_path
    
    # 计算meshes目录的绝对路径
    # URDF在 resources/urdf/，meshes在 resources/meshes/
    resources_dir = urdf.parent.parent  # 从 resources/urdf/ 回到 resources/
    meshes_dir = (resources_dir / "meshes").resolve()
    
    # 替换相对路径为绝对路径
    patched = content.replace(marker, f"{meshes_dir.as_posix()}/")
    
    # 创建临时文件
    try:
        fd, tmp_path = tempfile.mkstemp(prefix="play_urdf_", suffix=".urdf")
        os.close(fd)
        tmp = Path(tmp_path)
        tmp.write_text(patched, encoding="utf-8")
        return str(tmp)
    except Exception:
        return urdf_path


def main():
    parser = argparse.ArgumentParser(description="Swing motion example for robotic arm")
    parser.add_argument(
        "--urdf_path",
        type=str,
        default="./resources/urdf/play.urdf",
        help="URDF path for the robotic arm",
    )
    parser.add_argument(
        "--can_port",
        type=str,
        default="can0",
        # default="slcan0",
        help="CAN port name, default can0",
    )
    args = parser.parse_args()

    # 修复URDF中的相对路径问题
    urdf_path = prepare_urdf(args.urdf_path)
    temp_urdf = None
    if urdf_path != args.urdf_path:
        temp_urdf = urdf_path  # 保存临时文件路径，以便后续清理
        print(f"使用修复后的URDF路径: {urdf_path}")

    arm = robotic_arm.RoboticArm()

    if not arm.initialize(args.can_port, urdf_path):
        # 初始化失败时清理临时文件
        if temp_urdf and Path(temp_urdf).exists():
            try:
                Path(temp_urdf).unlink()
            except Exception:
                pass
        print("Failed to initialize robotic arm")
        return 1

    print("Starting swing motion example...")

    # Swing parameters
    duration_seconds = 10.0  # Total swing time
    ratio = 2.0

    # Phase 1: PVT Control Mode Swinging with immediate updates  
    # 直控模式，直接控制关节位置
    print("\n=== Phase 1: PVT Control Mode (10 seconds) ===")
    arm.set_param("arm.control_mode", robotic_arm.ControlMode.PVT)
    arm.set_param("immediate_update", 1)

    start_time = time.time()

    while time.time() - start_time < duration_seconds:
        t = int(time.time() * 1000)  # milliseconds
        # Calculate sinusoidal swing position for first joint
        target_pos = math.sin(t / 1e3 * ratio)

        # Target positions (only first joint swings, others stay at 0)
        # target_positions = [target_pos, 0.0, 0.0, 0.0, 0.0, 0.0]
        # target_positions = [0.0, target_pos, 0.0, 0.0, 0.0, 0.0]
        # target_positions = [0.0, 0.0, target_pos, 0.0, 0.0, 0.0]
        # target_positions = [0.0, 0.0, 0.0, target_pos, 0.0, 0.0]
        target_positions = [0.0, 0.0, 0.0, 0.0, target_pos, 0.0]
        # target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, target_pos]
        max_velocities = [math.pi , math.pi , math.pi, math.pi , math.pi, math.pi ]
        torques = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

        arm.pvt(target_positions, max_velocities, torques)
        time.sleep(0.01)  # 2ms

    # Phase 2: PVT Control Mode Swinging with filtered updates
    # servo模式，通过滤波器平滑控制关节位置
    # print("\n=== Phase 2: PVT Control Mode with Filtering (10 seconds) ===")
    # arm.set_param("immediate_update", 0)

    # start_time = time.time()

    # while time.time() - start_time < duration_seconds:
    #     t = int(time.time() * 1000)  # milliseconds
    #     # Calculate sinusoidal swing position for first joint
    #     target_pos = math.sin(t / 1e3 * ratio)

    #     # Target positions (only first joint swings, others stay at 0)
    #     # target_positions = [target_pos, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, target_pos, 0.0, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, target_pos, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, 0.0, target_pos, 0.0, 0.0]
    #     target_positions = [0.0, 0.0, 0.0, 0.0, target_pos, 0.0]
    #     # target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, target_pos]
    #     max_velocities = [math.pi / 4, math.pi / 10, math.pi / 10, math.pi / 10, math.pi / 10, math.pi / 10]
    #     torques = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

    #     arm.pvt(target_positions, max_velocities, torques)
    #     time.sleep(0.002)  # 2ms

    # # Phase 3: MIT Control Mode Swinging with immediate updates
    # # 阻抗模式，通过阻抗控制关节位置
    # print("\n=== Phase 3: MIT Control Mode (10 seconds) ===")
    # arm.set_param("arm.control_mode", robotic_arm.ControlMode.MIT)
    # arm.set_param("immediate_update", 1)

    # start_time = time.time()

    # while time.time() - start_time < duration_seconds:
    #     t = int(time.time() * 1000)  # milliseconds
    #     # Calculate sinusoidal swing position for first joint
    #     target_pos = math.sin(t / 1e3 * ratio)

    #     # Target positions (only first joint swings, others stay at 0)
    #     # target_positions = [target_pos, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, target_pos, 0.0, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, target_pos, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, 0.0, target_pos, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, 0.0, 0.0, target_pos, 0.0]
    #     target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, target_pos]
    #     max_velocities = [math.pi / 4, 0.0, 0.0, 0.0, 0.0, math.pi / 4]
    #     torques = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     kp = [1.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    #     kd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    #     arm.mit(target_positions, max_velocities, torques, kp, kd)
    #     time.sleep(0.002)  # 2ms

    # # Phase 4: MIT Control Mode Swinging with filtered updates
    # print("\n=== Phase 4: MIT Control Mode with Filtering (10 seconds) ===")
    # arm.set_param("immediate_update", 0)

    # start_time = time.time()

    # while time.time() - start_time < duration_seconds:
    #     t = int(time.time() * 1000)  # milliseconds
    #     # Calculate sinusoidal swing position for first joint
    #     target_pos = math.sin(t / 1e3 * ratio)

    #     # Target positions (only first joint swings, others stay at 0)
    #     target_positions = [target_pos, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, target_pos, 0.0, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, target_pos, 0.0, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, 0.0, target_pos, 0.0, 0.0]
    #     # target_positions = [0.0, 0.0, 0.0, 0.0, target_pos, 0.0]
    #     # target_positions = [0.0, 0.0, 0.0, 0.0, 0.0, target_pos]
    #     max_velocities = [math.pi / 4, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     torques = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     kp = [1.0, 100.0, 100.0, 10.0, 10.0, 10.0]
    #     kd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]

    #     arm.mit(target_positions, max_velocities, torques, kp, kd)
    #     time.sleep(0.002)  # 2ms

    # # Test EEF if available
    # if arm.has_eef():
    #     print("\n=== EEF Control Test ===")
    #     eef_type = arm.get_eef_type()
    #     if eef_type == robotic_arm.EEFType.G2:
    #         arm.set_param("eef.control_mode", robotic_arm.ControlMode.PVT)
    #         print("G2 gripper detected - testing open/close motion")

    #         start_time = time.time()
    #         while time.time() - start_time < duration_seconds:
    #             t = int(time.time() * 1000)  # milliseconds
    #             arm.eef_pvt(0.03 * (1 + math.sin(t / 1e3 * ratio)), 10, 5.0)
    #             time.sleep(0.002)  # 2ms
    #     elif eef_type == robotic_arm.EEFType.E2:
    #         print("E2 end effector detected - testing small motion")
    #         arm.eef_mit(0.0, 0.0, 0.0, 0.0, 0.0)
    #         time.sleep(2)
    # else:
    #     print("No EEF detected")

    # print("\n=== Swing example completed ===")
    arm.shutdown()
    
    # 清理临时URDF文件
    if temp_urdf and Path(temp_urdf).exists():
        try:
            Path(temp_urdf).unlink()
        except Exception:
            pass
    
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
