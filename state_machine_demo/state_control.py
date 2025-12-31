import logging
import os
import threading
import time
import tempfile
from pathlib import Path
from typing import Optional, Iterable, List, Tuple, Union, Any

from airbot_state_machine import robotic_arm
logger = logging.getLogger(__name__)

class JointStates:
    __slots__ = ["stamp", "tau", "q", "dq", "motor_names"]

    def __init__(self):
        self.stamp = 0
        self.tau = []
        self.q = []
        self.dq = []
        self.motor_names = []

class StateMachineController:
    """轻量封装 airbot_state_machine.RoboticArm，用于拖动模式与录制重放。"""

    def __init__(self, resources_dir: Path):
        self._resources_dir = Path(resources_dir)
        self._arm: Optional[robotic_arm.RoboticArm] = None
        self._urdf_path: Optional[Path] = None
        self._recorded_traj: Optional[robotic_arm.ArmTrajectory] = None
        self._record_file: Optional[Path] = None
        self._lock = threading.Lock()
        self._initialized = False
        self._eef_type: Optional[str] = None

        self._can_port: Optional[str] = None

    def initialize(self, can_port: str, eef_type: str) -> bool:
        """初始化 state machine 机械臂，返回是否成功。"""
        with self._lock:
            if self._initialized:
                return True

            self._urdf_path = self._prepare_urdf(eef_type)
            if not self._urdf_path or not self._urdf_path.exists():
                logger.error("URDF 文件不存在: %s", self._urdf_path)
                return False

            try:
                self._arm = robotic_arm.RoboticArm()
                self._can_port = can_port
                ok = self._arm.initialize(can_port, str(self._urdf_path))
                if not ok:
                    logger.error("state machine 机械臂初始化失败")
                    self._arm = None
                    return False

                # 设置默认控制模式为 PVT
                try:
                    self._arm.set_param("arm.control_mode", robotic_arm.ControlMode.PVT)
                    self._arm.set_param("immediate_update", 0)
                except Exception as exc:
                    logger.warning("设置控制模式失败: %s", exc)
                self._initialized = True
                logger.info("state machine 初始化成功: can=%s urdf=%s", can_port, self._urdf_path)
                return True
            except Exception as exc:  # pragma: no cover - 仅运行时捕获
                logger.exception("初始化 state machine 失败: %s", exc)
                self._arm = None
                return False

    def shutdown(self) -> None:
        """安全关闭 state machine 机械臂。"""
        with self._lock:
            if not self._arm:
                return
            try:
                # 尝试停止回放/录制状态
                try:
                    self._arm.replay_stop()
                except Exception:
                    pass
                try:
                    self._arm.record_stop(robotic_arm.ArmTrajectory())
                except Exception:
                    pass
                self._arm.shutdown()
            except Exception as exc:  # pragma: no cover
                logger.warning("关闭 state machine 异常: %s", exc)
            finally:
                self._arm = None
                self._initialized = False
                self._recorded_traj = None
                self._record_file = None

    # -------------------- 内部工具 -------------------- #
    def _ensure_arm(self) -> bool:
        if not self._arm or not self._initialized:
            logger.error("state machine 未初始化")
            return False
        return True

    def _resolve_urdf(self, eef_type: str) -> Path:
        """按末端类型选择 URDF，默认使用 play.urdf。"""
        type_key = (eef_type or "").strip()
        urdf_name = "play.urdf"
        if not type_key or type_key == "none":
            urdf_name = "play.urdf"
        elif type_key == "gripper":
            urdf_name = "play_g2.urdf"
        return self._resources_dir / "urdf" / urdf_name

    def _prepare_urdf(self, eef_type: str) -> Path:
        """返回可用 URDF 路径；若包含相对 mesh 路径则生成临时绝对路径副本。"""
        urdf_path = self._resolve_urdf(eef_type)
        if not urdf_path.exists():
            return urdf_path

        try:
            content = urdf_path.read_text(encoding="utf-8")
        except Exception:
            return urdf_path

        # 仅当存在相对 meshes 引用时进行替换
        marker = "../meshes/"
        if marker not in content:
            return urdf_path

        meshes_dir = (self._resources_dir / "meshes").resolve()
        patched = content.replace(marker, f"{meshes_dir.as_posix()}/")
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="play_urdf_", suffix=".urdf")
            os.close(fd)
            tmp = Path(tmp_path)
            tmp.write_text(patched, encoding="utf-8")
            return tmp
        except Exception:
            return urdf_path

    def set_arm_param(self, vel: Iterable[float], eff: Iterable[float]) -> bool:
        """设置关节/末端的速度与力矩限制。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                # 仅透传，具体参数含义由固件决定
                self._arm.set_param("arm.max_velocity", list(vel))
                self._arm.set_param("arm.max_effort", list(eff))
                return True
            except Exception as exc:
                logger.error("设置关节参数失败: %s", exc)
                return False

    # -------------------- 模式切换 -------------------- #
    def switch_state(self, target_state: robotic_arm.ArmState) -> bool:
        """通用状态切换封装。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return self._arm.switch_state(target_state)
            except Exception as exc:  # pragma: no cover
                logger.error("切换状态失败: %s", exc)
                return False

    def get_arm_state(self) -> Optional[robotic_arm.ArmState]:
        """获取当前内部状态机的状态，若失败返回 None。"""
        if not self._ensure_arm():
            return None
        try:
            return self._arm.get_current_state()
        except Exception:
            return None

    # -------------------- 重力补偿模式 -------------------- #
    def enter_gravity_mode(self) -> bool:
        """进入重力补偿/拖动模式。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                ok = self._arm.switch_state(robotic_arm.ArmState.GRAVITY_COMPENSATION)
                if not ok:
                    # 有些固件返回 False 但状态已切换，做一次状态确认
                    state = self.get_state()
                    if state == robotic_arm.ArmState.GRAVITY_COMPENSATION:
                        logger.info("已处于重力补偿状态（返回值为 False）")
                        return True
                if ok:
                    logger.info("进入重力补偿模式成功")
                else:
                    logger.error("进入重力补偿模式失败，switch_state 返回 False")
                return ok
            except Exception as exc:  # pragma: no cover
                logger.error("进入重力补偿失败: %s", exc)
                return False

    def exit_gravity_mode(self) -> bool:
        """退出重力补偿，恢复默认 PVT 控制。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                self._arm.set_param("arm.control_mode", robotic_arm.ControlMode.PVT)
                self._arm.set_param("immediate_update", 1)
                ok = self._arm.switch_state(robotic_arm.ArmState.DEFAULT)
                time.sleep(0.05)
                self._arm.set_param("immediate_update", 0)
                if ok:
                    logger.info("退出重力补偿模式成功")
                else:
                    # 再确认一次状态
                    state = self.get_state()
                    if state == robotic_arm.ArmState.DEFAULT:
                        logger.info("已处于 DEFAULT 状态（返回值为 False）")
                        return True
                    logger.error("退出重力补偿模式失败，switch_state 返回 False")
                return ok
            except Exception as exc:  # pragma: no cover
                logger.error("退出重力补偿失败: %s", exc)
                return False

    def gravity_calibrate(self) -> bool:
        """执行重力补偿标定。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                ok = self._arm.gravity_calibrate()
                logger.info("gravity_calibrate result: %s", ok)
                return bool(ok)
            except Exception as exc:  # pragma: no cover
                logger.error("重力补偿标定失败: %s", exc)
                return False

    # -------------------- 关节 / 笛卡尔 / EEF 控制 -------------------- #
    def joint_pvt(self, joints: Iterable[float], velocities: Iterable[float], efforts: Iterable[float]) -> bool:
        """直接下发关节 PVT 控制。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return bool(self._arm.pvt(list(joints), list(velocities), list(efforts)))
            except Exception as exc:
                logger.error("joint_pvt 失败: %s", exc)
                return False

    def move_to_joint_pos(self, joint_pos: List[float], blocking: bool = True) -> bool:
        """移动到关节位置。
        
        Args:
            joint_pos: 目标关节位置（6个关节角度，单位：弧度）
            blocking: 是否阻塞直到到达目标位置
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        # 不持有锁执行，因为 plan_and_execute 可能耗时较长
        with self._lock:
            if not self._ensure_arm():
                return False
            arm = self._arm  # 保存引用，避免长时间持有锁
            velocity_factor = 0.06  # 默认速度因子
        
        try:
            # 在锁外执行，避免阻塞其他操作
            ok = bool(arm.plan_and_execute(joint_pos, velocity_factor))
            if ok and blocking:
                # 等待到达目标位置
                return bool(arm.wait_reach(joint_pos, tolerance=0.02, timeout_seconds=10.0))
            return ok
        except Exception as exc:
            logger.error("move_to_joint_pos 失败: %s", exc)
            return False

    def move_with_joint_waypoints(self, waypoints: List[List[float]], blocking: bool = True) -> bool:
        """沿关节路径点移动。
        
        Args:
            waypoints: 关节路径点列表，每个路径点是6个关节角度的列表
            blocking: 是否阻塞直到到达最后一个路径点
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        # 不持有锁执行，因为 plan_and_execute 可能耗时较长
        with self._lock:
            if not self._ensure_arm():
                return False
            arm = self._arm  # 保存引用，避免长时间持有锁
            velocity_factor = 0.06  # 默认速度因子
        
        try:
            # 在锁外执行，避免阻塞其他操作
            ok = bool(arm.plan_and_execute(waypoints, velocity_factor))
            if ok and blocking:
                # 等待到达最后一个路径点
                if waypoints:
                    return bool(arm.wait_reach(waypoints[-1], tolerance=0.02, timeout_seconds=15.0))
            return ok
        except Exception as exc:
            logger.error("move_with_joint_waypoints 失败: %s", exc)
            return False

    def move_with_cart_waypoints(self, waypoints: List[List[List[float]]], blocking: bool = True) -> bool:
        """沿笛卡尔路径点移动。
        
        Args:
            waypoints: 笛卡尔路径点列表，每个路径点是 [[x,y,z], [x,y,z,w]] 格式
            blocking: 是否阻塞直到到达最后一个路径点
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        # 不持有锁执行，因为 plan_and_execute 可能耗时较长
        with self._lock:
            if not self._ensure_arm():
                return False
            arm = self._arm  # 保存引用，避免长时间持有锁
            velocity_factor = 0.06  # 默认速度因子
        
        try:
            # 转换为元组列表格式供 plan_and_execute 使用
            target_waypoints = [(wp[0], wp[1]) for wp in waypoints]
            # 在锁外执行，避免阻塞其他操作
            ok = bool(arm.plan_and_execute(target_waypoints, velocity_factor))
            if ok and blocking:
                # 对于笛卡尔路径点，等待完成
                max_wait = 15.0
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    if not arm.is_running():
                        return True
                    time.sleep(0.1)
                return False
            return ok
        except Exception as exc:
            logger.error("move_with_cart_waypoints 失败: %s", exc)
            return False

    def move_to_cart_pose(self, cart_pose: List[List[float]], blocking: bool = True) -> bool:
        """移动到笛卡尔位姿。
        
        Args:
            cart_pose: 目标位姿，格式为 [[x,y,z], [x,y,z,w]]
                - 第一个列表是位置 [x, y, z]（单位：米）
                - 第二个列表是四元数 [x, y, z, w]
            blocking: 是否阻塞直到到达目标位置
            
        Returns:
            bool: 成功返回 True，失败返回 False
        """
        # 不持有锁执行，因为 plan_and_execute 可能耗时较长
        with self._lock:
            if not self._ensure_arm():
                return False
            arm = self._arm  # 保存引用，避免长时间持有锁
            velocity_factor = 0.06  # 默认速度因子
        
        try:
            # 转换为元组格式供 plan_and_execute 使用
            target = (cart_pose[0], cart_pose[1])
            # 在锁外执行，避免阻塞其他操作
            ok = bool(arm.plan_and_execute(target, velocity_factor))
            if ok and blocking:
                # 对于笛卡尔位姿，等待到达需要获取当前关节位置并等待
                # 这里简化处理，等待一段时间
                time.sleep(0.1)  # 给一些时间让运动开始
                # 可以通过检查 is_running 状态来判断是否完成
                max_wait = 10.0
                start_time = time.time()
                while time.time() - start_time < max_wait:
                    if not arm.is_running():
                        return True
                    time.sleep(0.1)
                return False
            return ok
        except Exception as exc:
            logger.error("move_to_cart_pose 失败: %s", exc)
            return False

    def eef_pvt(self, position: Union[float, List[float]], velocity: float = 1.0, effort: float = 2.0) -> bool:
        """末端执行器 PVT 控制。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return bool(self._arm.eef_pvt(position, velocity, effort))
            except Exception as exc:
                logger.error("eef_pvt 失败: %s", exc)
                return False

    def servo_joint_pos(
        self,
        joint_pos: List[float],
        velocity: Optional[List[float]] = None,
        effort: Optional[List[float]] = None,
    ) -> bool:
        """实时关节位置直发控制（丝滑键控用）。

        - 将 immediate_update 设为 0，避免频繁参数下发。
        - 直接调用 PVT 接口直发。
        """
        with self._lock:
            if not self._ensure_arm():
                return False
            arm = self._arm
            # 默认速度/力矩
            if velocity is None:
                velocity = [0.5] * 6
            if effort is None:
                effort = [2.0] * 6
            try:
                arm.set_param("immediate_update", 0)
            except Exception:
                pass

        try:
            return bool(arm.pvt(list(joint_pos), list(velocity), list(effort)))
        except Exception as exc:
            logger.error("servo_joint_pos 失败: %s", exc)
            return False

    def wait_reach(self, target: Iterable[float], tolerance: float = 0.02, timeout: float = 10.0) -> bool:
        """等待到达目标位置。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return bool(self._arm.wait_reach(list(target), tolerance, timeout))
            except Exception as exc:
                logger.error("wait_reach 失败: %s", exc)
                return False

    def get_joint_states(self) -> Optional["JointStates"]:
        """获取当前关节位置 / 速度 / 力矩 数据反馈。

        Returns:
            JointStates: 关节位置 / 速度 / 力矩 数据反馈对象。
        """

        joint_names = [
            "joint1",
            "joint2",
            "joint3",
            "joint4",
            "joint5",
            "joint6",
        ]
        joint_states = JointStates()
        joint_states.stamp = time.time_ns() / 1e9
        joint_states.motor_names = joint_names

        with self._lock:
            if not self._ensure_arm():
                return None
            # 关节位置 rad
            joint_states.q   = list(self._arm.get_position())
            # 关节速度（弧度/秒）
            joint_states.dq  = list(self._arm.get_velocity())
            # 关节力矩（牛·米）
            joint_states.tau = list(self._arm.get_torque())

        return joint_states

    def get_end_pose(self) -> Optional[List[List[float]]]:
        """获取末端位姿，返回 [[x,y,z], [x,y,z,w]] 格式。"""
        with self._lock:
            if not self._ensure_arm():
                return None
            try:
                pose = self._arm.get_end_cart_pose()
                return [list(pose.pos), list(pose.quat)]
            except Exception as exc:
                logger.error("获取末端位姿失败: %s", exc)
                return None

    def get_eef_pos(self) -> Optional[Union[float, List[float]]]:
        """获取末端位置，失败返回 None。"""
        with self._lock:
            if not self._ensure_arm():
                return None
            try:
                return self._arm.eef_get_position()
            except Exception as exc:
                logger.error("获取末端位置失败: %s", exc)
                return None

    def get_eef_eff(self) -> Optional[List[float]]:
        """获取当前末端执行器力矩。"""
        with self._lock:
            if not self._ensure_arm():
                return None
            try:
                torque = self._arm.eef_get_torque()
                # 转换为列表格式
                if isinstance(torque, (int, float)):
                    return [torque]
                return list(torque) if torque is not None else None
            except Exception as exc:
                logger.error("获取末端力矩失败: %s", exc)
                return None

    # -------------------- 录制/回放 -------------------- #
    def start_record(self, sampling_time: float = 0.01, include_eef: bool = True, save_path: Optional[Path] = None) -> bool:
        """开始录制轨迹。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            target = Path(save_path) if save_path else None
            self._record_file = target
            try:
                self._arm.switch_state(robotic_arm.ArmState.RECORDING_IDLE)
                return self._arm.record_start(sampling_time, include_eef, str(target) if target else "", False)
            except Exception as exc:  # pragma: no cover
                logger.error("开始录制失败: %s", exc)
                return False

    def stop_record(self) -> Optional[robotic_arm.ArmTrajectory]:
        """停止录制并返回轨迹。"""
        with self._lock:
            if not self._ensure_arm():
                return None
            try:
                traj = robotic_arm.ArmTrajectory()
                if self._arm.record_stop(traj):
                    self._recorded_traj = traj
                    return traj
                return None
            except Exception as exc:  # pragma: no cover
                logger.error("停止录制失败: %s", exc)
                return None

    def start_replay(self, traj: Optional[robotic_arm.ArmTrajectory] = None) -> bool:
        """开始回放指定或最近录制的轨迹。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            target = traj or self._recorded_traj
            if target is None:
                logger.warning("无可用轨迹进行回放")
                return False
            try:
                return self._arm.replay_start(target)
            except Exception as exc:  # pragma: no cover
                logger.error("开始回放失败: %s", exc)
                return False

    def halt_replay(self) -> bool:
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return self._arm.replay_halt()
            except Exception as exc:  # pragma: no cover
                logger.error("暂停回放失败: %s", exc)
                return False

    def resume_replay(self) -> bool:
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return self._arm.replay_resume()
            except Exception as exc:  # pragma: no cover
                logger.error("继续回放失败: %s", exc)
                return False

    def stop_replay(self) -> bool:
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return self._arm.replay_stop()
            except Exception as exc:  # pragma: no cover
                logger.error("停止回放失败: %s", exc)
                return False

    # -------------------- 参数 / 固件 -------------------- #
    def get_params(self) -> dict:
        """读取机械臂参数/固件信息。"""
        with self._lock:
            if not self._ensure_arm():
                return {}
            try:
                return dict(self._arm.get_params())
            except Exception as exc:
                logger.error("读取参数失败: %s", exc)
                return {}

    def get_eef_params(self) -> dict:
        """读取末端执行器参数/固件信息。"""
        with self._lock:
            if not self._ensure_arm():
                return {}
            try:
                return dict(self._arm.get_eef_params())
            except Exception as exc:
                logger.error("读取末端参数失败: %s", exc)
                return {}

    def clear_error(self) -> bool:
        """清除电机故障。"""
        with self._lock:
            if not self._ensure_arm():
                return False
            try:
                return bool(self._arm.clear_error())
            except Exception as exc:
                logger.error("清除故障失败: %s", exc)
                return False