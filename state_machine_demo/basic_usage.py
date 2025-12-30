import time
import argparse
from airbot_state_machine import robotic_arm
from urdf_helper import prepare_urdf_path


def main():
    parser = argparse.ArgumentParser(description="Airbot RoboticArm Python SDK demo")
    parser.add_argument(
        "--can",
        type=str,
        default="can0",
        help="CAN interface name (default: can0)",
    )
    parser.add_argument(
        "--urdf",
        type=str,
        help="Path to URDF file of the robot",
    )
    args = parser.parse_args()

    # === Prepare URDF path (fix relative mesh paths) ===
    urdf_path = prepare_urdf_path(args.urdf)

    # === Create robotic arm object ===
    arm = robotic_arm.RoboticArm()

    # === Initialize hardware & software ===
    if not arm.initialize(args.can, urdf_path):
        print("❌ Failed to initialize robotic arm")
        return 1
    print("✅ Robotic arm initialized successfully")

    # === Query current joint position ===
    print("Current position:", arm.get_position())

    # === Set PVT control mode ===
    print("Switching to PVT control mode")
    arm.set_param("arm.control_mode", robotic_arm.ControlMode.PVT)
    arm.set_param("immediate_update", 1)

    # === Send PVT command (position, velocity, torque/current limit) ===
    print("Sending PVT command...")
    # 注意：目标位置必须在关节限制范围内
    target_pos = [1.0, -0.5, 0.5, 0.1, 0.1, 1.0]  # 调整关节2和关节3的值以符合限制
    target_vel = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    target_eff = [2.0, 2.0, 2.0, 2.0, 2.0, 2.0]  # 最大力矩限制
    arm.pvt(target_pos, target_vel, target_eff)

    # === EEF (end-effector) control ===
    print("Controlling EEF (e.g., G2 gripper)")
    arm.eef_pvt(0.02, 1, 2)
    time.sleep(2)

    # === Switch to gravity compensation mode ===
    print("Switching to GRAVITY_COMPENSATION mode (free-floating)")
    arm.switch_state(robotic_arm.ArmState.GRAVITY_COMPENSATION)
    time.sleep(10)

    # === Back to DEFAULT state ===
    print("Returning to DEFAULT state")
    arm.switch_state(robotic_arm.ArmState.DEFAULT)
    arm.set_param("immediate_update", 0)

    # === Return to home position with PVT ===
    print("Returning to home position...")
    # 使用当前位置作为"home"，因为 [0,0,0,0,0,0] 可能超出某些关节限制
    current_pos = arm.get_position()
    home = list(current_pos)  # 使用当前位置作为安全的目标位置
    arm.pvt(home, target_vel, target_eff)
    arm.wait_reach(home, 0.02, 10.0)

    # === MIT control mode (impedance/force control) ===
    print("Switching to MIT mode...")
    arm.set_param("arm.control_mode", robotic_arm.ControlMode.MIT)
    target_pos = [0.2, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_vel = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_eff = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    target_kp = [100.0, 100.0, 200.0, 20.0, 20.0, 20.0]
    target_kd = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    arm.mit(target_pos, target_vel, target_eff, target_kp, target_kd)
    arm.wait_reach(target_pos, 0.02, 10.0)

    # === Velocity control (CSV mode) ===
    print("Switching to CSV (velocity control) mode...")
    arm.set_param("arm.control_mode", robotic_arm.ControlMode.CSV)
    target_vel = [0.1, 0.0, 0.0, 0.1, 0.1, 0.1]
    arm.vel(target_vel)
    time.sleep(3)
    arm.vel([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # === Return to home position ===
    print("Back to home position...")
    arm.set_param("arm.control_mode", robotic_arm.ControlMode.PVT)
    # 使用当前位置作为安全的目标位置
    current_pos = arm.get_position()
    target_pos = list(current_pos)
    target_vel = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    target_eff = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]
    arm.pvt(target_pos, target_vel, target_eff)
    arm.wait_reach(target_pos)

    # === Shutdown ===
    print("Shutting down robotic arm...")
    time.sleep(3)
    arm.shutdown()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
