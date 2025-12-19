#!/usr/bin/env python3
"""
仿真验证脚本：使用校准模型的逆动力学进行闭环控制

核心思想:
1. 使用预先校准的MuJoCo模型（由create_calibrated_model.py生成，包含pi_s参数）
2. 从期望轨迹获取qdd_des
3. 闭环控制：每步基于实际状态计算tau
   - 读取实际状态：q_actual, qd_actual
   - 获取期望加速度：qdd_des（从理论轨迹）
   - 计算控制力矩：tau = mj_inverse(model_calibrated, q_actual, qd_actual, qdd_des)
4. mj_inverse使用的是校准后的模型（包含pi_s参数）
5. 对比实际轨迹与期望轨迹，验证参数准确性

控制策略（与test.py一致）:
- 闭环逆动力学控制
- 基于实际状态 + 期望加速度
- 实时计算tau（不是预计算插值）

执行前提:
- 必须先运行 create_calibrated_model.py 生成校准后的XML模型

流程:
1. 加载校准后的MuJoCo XML（已包含pi_s更新的完整惯性参数）
2. 加载估计参数（pi_b, pi_fr用于后续对比）
3. 从MAT文件加载理论激励轨迹
4. 准备期望加速度插值函数
5. 仿真执行：闭环控制（实时mj_inverse）
6. 分析和可视化：对比理论轨迹与实际轨迹
"""

import numpy as np
import pickle
import sys
import os
import h5py
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import mujoco
import mujoco.viewer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


def mixed_trajectory_calculator(t_vec, T, N, wf, a, b, c_pol, q0):
    """
    通过解析表达式计算轨迹。
    这种方法得到的 qddot是数学上连续的,无noise
    """
    t_vec = np.atleast_1d(t_vec)
    J = a.shape[0]  
    M = len(t_vec)  
    qd, qdot_d, qddot_d = np.zeros((J, M)), np.zeros((J, M)), np.zeros((J, M))
    tau_vec = t_vec % T  
    
    for i in range(J):
        k_vec = np.arange(1, N + 1).reshape(-1, 1)
        wk_t = wf * k_vec * t_vec
        sin_wk_t, cos_wk_t = np.sin(wk_t), np.cos(wk_t)
        
        # 傅里叶部分
        a_norm, b_norm = a[i, :].reshape(-1, 1) / (wf * k_vec), b[i, :].reshape(-1, 1) / (wf * k_vec)
        qd_fourier = (a_norm * sin_wk_t - b_norm * cos_wk_t).sum(axis=0)
        qdot_d_fourier = (a[i, :].reshape(-1, 1) * cos_wk_t + b[i, :].reshape(-1, 1) * sin_wk_t).sum(axis=0)
        qddot_d_fourier = ((-a[i, :].reshape(-1, 1) * wf * k_vec) * sin_wk_t + 
                           (b[i, :].reshape(-1, 1) * wf * k_vec) * cos_wk_t).sum(axis=0)
        
        # 多项式部分
        qd_poly, qdot_d_poly, qddot_d_poly = np.zeros(M), np.zeros(M), np.zeros(M)
        for k_exp in range(6):
            c = c_pol[i, k_exp]
            qd_poly += c * (tau_vec ** k_exp)
            if k_exp >= 1: qdot_d_poly += c * k_exp * (tau_vec ** (k_exp - 1))
            if k_exp >= 2: qddot_d_poly += c * k_exp * (k_exp - 1) * (tau_vec ** (k_exp - 2))

        qd[i, :] = qd_fourier + qd_poly
        qdot_d[i, :] = qdot_d_fourier + qdot_d_poly
        qddot_d[i, :] = qddot_d_fourier + qddot_d_poly
    
    return qd, qdot_d, qddot_d
    
class SimulationValidator:
    """仿真验证器"""
    
    def __init__(self, xml_path, estimation_pkl_path, mat_file_path):
        """
        Args:
            xml_path: MuJoCo XML模型路径
            estimation_pkl_path: 估计结果pkl文件
            mat_file_path: MAT激励轨迹文件
        """
        self.xml_path = xml_path
        self.estimation_pkl_path = estimation_pkl_path
        self.mat_file_path = mat_file_path
        
        # 数据容器
        self.model = None
        self.data = None
        self.estimation_results = None
        self.traj_params = None
        
        # 仿真结果
        self.sim_time = []
        self.sim_q = []
        self.sim_qd = []
        self.sim_tau = []
        self.sim_tau_mj_inverse = []  # 实际使用的控制tau（来自mj_inverse）
        
        # 理论轨迹
        self.theory_q = None
        self.theory_qd = None
        self.theory_qdd = None
    
    def load_model(self):
        """加载校准后的MuJoCo模型"""
        print(f"\n[1/6] 加载校准后的MuJoCo模型...")
        
        if not os.path.exists(self.xml_path):
            raise FileNotFoundError(f"模型文件不存在: {self.xml_path}")
        
        print(f"  加载: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        
        print(f"  ✓ 模型加载成功")
        print(f"    Bodies: {self.model.nbody}")
        print(f"    Joints: {self.model.nv}")
        print(f"    时间步: {self.model.opt.timestep:.6f}s")
    
    def load_estimation_results(self, method='PC-OLS-REG'):
        """加载估计结果"""
        print(f"\n[2/6] 加载估计参数 ({method})...")
        
        with open(self.estimation_pkl_path, 'rb') as f:
            all_results = pickle.load(f)
        
        if method == 'PC-OLS-REG':
            self.estimation_results = all_results['sol_pc_reg']
        elif method == 'PC-OLS':
            self.estimation_results = all_results['sol_pc_ols']
        else:
            self.estimation_results = all_results['sol_ols']
        
        print(f"  ✓ pi_b: {self.estimation_results['pi_b'].shape}")
        print(f"  ✓ pi_fr: {self.estimation_results['pi_fr'].shape}")
        if self.estimation_results.get('pi_s') is not None:
            print(f"  ✓ pi_s: {self.estimation_results['pi_s'].shape}")
    
    def load_trajectory_params(self):
        """加载理论轨迹参数"""
        print(f"\n[3/6] 加载激励轨迹参数...")
        
        mat_data = loadmat(self.mat_file_path)
        
        # MAT文件结构：traj_par是一个结构体
        traj_par = mat_data['traj_par'][0, 0]
        
        self.traj_params = {
            'T': float(traj_par['T'][0, 0]),
            'N': int(traj_par['N'][0, 0]),
            'wf': float(traj_par['wf'][0, 0]),
            'a': mat_data['a'],
            'b': mat_data['b'],
            'c_pol': mat_data['c_pol'],
            'q0': traj_par['q0'].flatten()
        }
        
    
    def update_model_parameters(self, use_calibrated_xml=True):
        """更新模型参数10参数/link + damping + frictionloss"""
        print(f"\n[2/6] 更新机器人模型参数...")
        
        # 如果使用校准后的XML（包含完整fullinertia）
        if use_calibrated_xml:
            calibrated_xml = self.xml_path.replace('.xml', '_calibrated.xml')
            
            if not os.path.exists(calibrated_xml):
                print(f"  ⚠️  校准模型不存在: {calibrated_xml}")
                print(f"  正在生成...")
                
                # 自动生成校准模型
                from scripts.create_calibrated_model import create_calibrated_xml
                create_calibrated_xml(
                    self.estimation_pkl_path, 
                    self.xml_path, 
                    calibrated_xml,
                    method='PC-OLS-REG'
                )
            
            print(f"  ✓ 使用校准模型: {calibrated_xml}")
            self.model = mujoco.MjModel.from_xml_path(calibrated_xml)
        else:
            print(f"  使用原始模型: {self.xml_path}")
            self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        
        self.data = mujoco.MjData(self.model)
        
        if self.estimation_results.get('pi_s') is None:
            return
        
        pi_s = self.estimation_results['pi_s']
        pi_fr = self.estimation_results['pi_fr']
        
        n_bodies = self.model.nbody
        n_joints = self.model.nv
        
        print(f"  模型: {n_bodies} bodies, {n_joints} joints")
        
        # 更新body参数（10参数/link）
        for i in range(6):  # 6个实际连杆
            body_id = i + 2  # 跳过world
            param_idx = i * 10
            
            if param_idx + 9 < len(pi_s):
                # 提取10个参数
                Ixx, Ixy, Ixz, Iyy, Iyz, Izz = pi_s[param_idx:param_idx + 6]
                mx, my, mz = pi_s[param_idx + 6:param_idx + 9]
                mass = pi_s[param_idx + 9]
                
                # 计算重心
                com = np.array([mx, my, mz]) / mass if mass > 1e-6 else np.zeros(3)
                
                # 赋值给模型
                self.model.body_mass[body_id] = mass
                self.model.body_ipos[body_id] = com
                self.model.body_inertia[body_id] = np.array([Ixx, Iyy, Izz])
                
                print(f"    Link{i+1}: m={mass:.4f}kg, com=[{com[0]:.4f}, {com[1]:.4f}, {com[2]:.4f}]")
        
        # 更新joint参数（damping + frictionloss）
        print(f"\n  更新关节摩擦参数:")
        for i in range(min(6, n_joints)):
            # pi_fr结构: 每个关节3个参数 [viscous, coulomb, constant]
            # 我们取前两个: viscous->damping, coulomb->frictionloss
            viscous = pi_fr[i * 3]      # 第1个：damping
            coulomb = pi_fr[i * 3 + 1]  # 第2个：frictionloss
            # constant = pi_fr[i * 3 + 2]  # 第3个：不使用
            
            self.model.dof_damping[i] = max(0, viscous)
            self.model.dof_frictionloss[i] = max(0, coulomb)
            
            print(f"    Joint{i+1}: damping={viscous:.6f}, frictionloss={coulomb:.6f}")
        
        print("  ✓ 模型参数更新完成")
    
    def compute_control_torques(self):
        """
        从期望轨迹和基参数计算理论力矩（用于后续对比）
        
        注意：这个计算的tau仅用于分析对比，不用于实际控制！
        实际控制使用mj_inverse（基于校准模型）
        
        关键公式：
        tau = Y_base(q_des, qd_des, qdd_des) @ pi_b + Y_friction(qd_des) @ pi_fr
        
        其中：
        - Y_base: 动力学回归矩阵（基参数形式）
        - Y_friction: 摩擦回归矩阵
        - pi_b: 估计的基参数（最小参数集）
        - pi_fr: 估计的摩擦参数
        """
        print(f"\n[4/6] 从基参数计算理论力矩（用于对比）...")
        print(f"  注意: 仅用于分析，不用于实际控制")
        print(f"  公式: tau_theory = Wb(q_des, qd_des, qdd_des) @ [pi_b; pi_fr]")
        
        # 导入parameter_estimation的观测矩阵构建函数
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
        from dynamics.parameter_estimation import build_observation_matrices
        
        # 获取轨迹参数
        T = self.traj_params['T']
        N = self.traj_params['N']
        wf = self.traj_params['wf']
        a = self.traj_params['a']
        b = self.traj_params['b']
        c_pol = self.traj_params['c_pol']
        q0 = self.traj_params['q0']
        
        # 步骤1: 生成期望轨迹 (q_des, qd_des, qdd_des)
        dt_traj = 0.002
        time_vec = np.arange(0, T, dt_traj)
        n_samples = len(time_vec)
        
        print(f"\n  [步骤1] 生成期望轨迹: {n_samples}个采样点")
        
        # 计算期望轨迹（解析式，无噪声）
        q_des, qd_des, qdd_des = mixed_trajectory_calculator(
            time_vec, T, N, wf, a, b, c_pol, q0
        )
        
        # 转置为(n_samples, 6)格式
        q_des = q_des.T
        qd_des = qd_des.T
        qdd_des = qdd_des.T
        
        print(f"    ✓ q_des:   {q_des.shape}")
        print(f"    ✓ qd_des:  {qd_des.shape}")
        print(f"    ✓ qdd_des: {qdd_des.shape}")
        
        # 步骤2: 构造观测矩阵 Wb = [Y_base | Y_friction]
        print(f"\n  [步骤2] 构造观测矩阵 Wb(q_des, qd_des, qdd_des)...")
        
        # 构造trajectory_data字典（与parameter_estimation.py格式一致）
        traj_data = {
            't': time_vec,
            'q': q_des,
            'qd_fltrd': qd_des,
            'q2d_est': qdd_des,
            'i_fltrd': np.zeros_like(q_des)  # 占位，不使用
        }
        
        # 加载baseQR（基参数映射矩阵）
        baseQR_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'baseQR_standard.mat')
        with h5py.File(baseQR_path, 'r') as f:
            baseQR_group = f['baseQR']
            n_base = int(np.array(baseQR_group['numberOfBaseParameters']).flatten()[0])
            E_full = np.array(baseQR_group['permutationMatrix']).T
            beta = np.array(baseQR_group['beta']).T
        
        baseQR = {
            'numberOfBaseParameters': n_base,
            'permutationMatrixFull': E_full,
            'beta': beta.reshape(-1, 1) if beta.ndim == 1 else beta
        }
        
        print(f"    ✓ 加载baseQR: n_base={n_base}个基参数")
        
        # 使用build_observation_matrices批量构建
        # Wb = [Y_base(q,qd,qdd) @ E1 | Y_friction(qd)]
        # 其中 E1 = E[:, :n_base] 是标准参数到基参数的映射
        Tau_dummy, Wb = build_observation_matrices(traj_data, baseQR, np.ones(6))
        
        print(f"    ✓ 观测矩阵 Wb: {Wb.shape}")
        print(f"      - 动力学部分: {n_base}列（基参数）")
        print(f"      - 摩擦部分: 18列（6关节×3参数）")
        
        # 步骤3: 计算控制力矩 tau = Wb @ [pi_b; pi_fr]
        print(f"\n  [步骤3] 计算控制力矩 tau = Wb @ [pi_b; pi_fr]...")
        
        pi_b = self.estimation_results['pi_b']
        pi_fr = self.estimation_results['pi_fr']
        
        print(f"    ✓ 使用基参数 pi_b: {len(pi_b)}个")
        print(f"    ✓ 使用摩擦参数 pi_fr: {len(pi_fr)}个")
        
        params = np.concatenate([pi_b, pi_fr])
        Tau_control = Wb @ params
        
        # 重塑为(n_samples, 6)
        tau_control = Tau_control.reshape(-1, 6)
        
        # 保存结果
        self.theory_time = time_vec
        self.theory_q = q_des
        self.theory_qd = qd_des
        self.theory_qdd = qdd_des
        self.tau_control = tau_control
        
        print(f"\n  ✓ 控制力矩计算完成！")
        print(f"    力矩形状: {tau_control.shape}")
        print(f"    力矩范围: [{np.min(tau_control):.3f}, {np.max(tau_control):.3f}] Nm")
        
        # 显示每个关节的力矩统计
        print(f"\n  各关节力矩统计:")
        for i in range(6):
            print(f"    Joint{i+1}: [{np.min(tau_control[:, i]):7.3f}, {np.max(tau_control[:, i]):7.3f}] Nm, "
                  f"mean={np.mean(tau_control[:, i]):7.3f}, std={np.std(tau_control[:, i]):7.3f}")
        
        print(f"\n  ✓ 这些tau仅用于后续对比分析（不用于实际控制）")
    
    def run_simulation(self, visualize=True):
        """
        使用闭环逆动力学控制校准后的机器人
        
        控制策略（与test.py一致）：
        - 每步读取实际状态：q_actual, qd_actual
        - 从期望轨迹获取：qdd_des
        - 使用mj_inverse计算：tau = f(q_actual, qd_actual, qdd_des)
        - mj_inverse使用的是校准后的模型（包含pi_s参数）
        
        核心验证思想：
        1. 模型参数来自pi_s（完整惯性）
        2. 控制基于校准模型的逆动力学
        3. 如果pi_s准确，轨迹跟踪应该很好
        4. 偏差越小，说明参数估计越准确
        """
        print(f"\n[5/6] 闭环控制校准模型 (可视化={visualize})...")
        print(f"  控制: tau = mj_inverse(校准模型, q_actual, qd_actual, qdd_des)")
        
        # 获取参数
        T = self.traj_params['T']
        N = self.traj_params['N']
        wf = self.traj_params['wf']
        a = self.traj_params['a']
        b = self.traj_params['b']
        c_pol = self.traj_params['c_pol']
        q0 = self.traj_params['q0']
        
        # 重置仿真
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始状态（使用期望轨迹的初始值）
        self.data.qpos[:6] = self.theory_q[0] 
        self.data.qvel[:6] = self.theory_qd[0]
        mujoco.mj_forward(self.model, self.data)
        
        # 仿真参数
        dt = self.model.opt.timestep
        n_steps = int(T / dt)
        
        print(f"\n  仿真参数:")
        print(f"    时长: {T}s")
        print(f"    时间步: {dt:.6f}s")
        print(f"    总步数: {n_steps}")
        print(f"    控制模式: 前馈（tau from pi_b + 期望轨迹）")
        
        # 创建期望轨迹插值函数（用于实时获取期望加速度）
        from scipy.interpolate import interp1d
        qdd_interp = [interp1d(self.theory_time, self.theory_qdd[:, j], 
                              kind='linear', fill_value='extrapolate') 
                     for j in range(6)]
        
        # 创建逆动力学数据结构（用于实时计算tau）
        inv_data = mujoco.MjData(self.model)
        
        # 可视化器
        viewer = None
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("  ✓ 可视化已开启")
        
        # 仿真循环（闭环控制：实际状态 + 期望加速度）
        record_every = max(1, int(0.002 / dt))
        
        print(f"\n  控制策略:")
        print(f"    - 使用校准模型的逆动力学（包含pi_s）")
        print(f"    - 闭环：tau = mj_inverse(q_actual, qd_actual, qdd_des)")
        print(f"    - 每步实时计算tau（基于实际状态）")
        
        for step in range(n_steps):
            t = self.data.time
            
            # 获取期望加速度（插值）
            qdd_des = np.array([qdd_interp[j](t) for j in range(6)])
            
            # 闭环控制：使用实际状态 + 期望加速度
            inv_data.qpos[:6] = self.data.qpos[:6]  # 实际位置
            inv_data.qvel[:6] = self.data.qvel[:6]  # 实际速度
            inv_data.qacc[:6] = qdd_des              # 期望加速度
            
            # 计算tau（使用校准模型，包含pi_s）
            mujoco.mj_inverse(self.model, inv_data)
            tau_control = inv_data.qfrc_inverse[:6].copy()
            
            # 应用控制
            self.data.ctrl[:6] = tau_control
            
            # 仿真一步
            mujoco.mj_step(self.model, self.data)
            
            # 更新可视化
            if visualize and viewer is not None:
                viewer.sync()
            
            # 记录数据
            if step % record_every == 0:
                self.sim_time.append(self.data.time)
                self.sim_q.append(self.data.qpos[:6].copy())
                self.sim_qd.append(self.data.qvel[:6].copy())
                self.sim_tau.append(tau_control.copy())
                self.sim_tau_mj_inverse.append(tau_control.copy())  # 记录实际控制tau
            
            # 进度
            if (step + 1) % 10000 == 0:
                print(f"    进度: {step+1}/{n_steps} ({100*(step+1)/n_steps:.1f}%)")
        
        if viewer is not None:
            viewer.close()
            print("  ✓ 可视化已关闭")
        
        # 转换为数组
        self.sim_time = np.array(self.sim_time)
        self.sim_q = np.array(self.sim_q)
        self.sim_qd = np.array(self.sim_qd)
        self.sim_tau = np.array(self.sim_tau)
        self.sim_tau_mj_inverse = np.array(self.sim_tau_mj_inverse)
        
        print(f"  ✓ 仿真完成，记录了 {len(self.sim_time)} 个数据点")
    
    def compare_with_theory(self, save_plots=True):
        """对比理论轨迹和力矩"""
        print(f"\n[6/6] 对比理论轨迹和力矩...")
        
        # 计算理论轨迹
        T = self.traj_params['T']
        N = self.traj_params['N']
        wf = self.traj_params['wf']
        a = self.traj_params['a']
        b = self.traj_params['b']
        c_pol = self.traj_params['c_pol']
        q0 = self.traj_params['q0']
        
        self.theory_q, self.theory_qd, self.theory_qdd = mixed_trajectory_calculator(
            self.sim_time, T, N, wf, a, b, c_pol, q0
        )
        
        # 转置以匹配(n_samples, n_joints)格式
        self.theory_q = self.theory_q.T
        self.theory_qd = self.theory_qd.T
        self.theory_qdd = self.theory_qdd.T
        
        # 准备两条tau曲线用于对比
        print("  准备tau对比数据...")
        
        # 1. tau_theory: 从pi_b计算的理论tau（基于期望轨迹）
        tau_theory_interp = np.zeros((len(self.sim_time), 6))
        from scipy.interpolate import interp1d
        for j in range(6):
            f = interp1d(self.theory_time, self.tau_control[:, j], 
                        kind='linear', fill_value='extrapolate')
            tau_theory_interp[:, j] = f(self.sim_time)
        
        # 2. tau_mj_inverse: 实际控制使用的tau（来自校准模型的mj_inverse）
        tau_mj_actual = self.sim_tau_mj_inverse
        
        # 保存用于绘图
        self.tau_theory = tau_theory_interp
        self.tau_mj_actual = tau_mj_actual
        
        # tau误差统计（关键验证！）
        tau_error = tau_theory_interp - tau_mj_actual
        
        print("\n  === 关键验证：Tau对比 ===")
        print("  对比两条tau曲线:")
        print("    1. tau_theory: 从pi_b计算（基于期望轨迹）")
        print("    2. tau_mj_inverse: 实际控制使用（校准模型逆动力学）")
        print("\n  如果两者接近 → pi_s和pi_b一致 → 参数估计成功！")
        print("\n  误差统计 (tau_theory vs tau_mj_inverse):")
        print("  关节 |  RMSE(Nm) |  最大(Nm) |  平均(Nm) |  相对误差%")
        print("  " + "-"*65)
        for j in range(6):
            rmse = np.sqrt(np.mean(tau_error[:, j]**2))
            max_err = np.max(np.abs(tau_error[:, j]))
            mean_err = np.mean(np.abs(tau_error[:, j]))
            tau_mean = np.mean(np.abs(tau_mj_actual[:, j]))
            rel_err = (rmse / tau_mean * 100) if tau_mean > 1e-6 else 0
            print(f"  Joint{j+1} | {rmse:9.6f} | {max_err:9.6f} | {mean_err:9.6f} | {rel_err:7.2f}%")
        
        avg_rmse = np.mean([np.sqrt(np.mean(tau_error[:, j]**2)) for j in range(6)])
        print(f"\n  平均RMSE: {avg_rmse:.6f} Nm")
        
        if avg_rmse < 0.1:
            print("  ✅ 误差很小！参数估计非常准确，pi_s和pi_b高度一致！")
        elif avg_rmse < 0.5:
            print("  ✓ 误差较小，参数估计良好")
        else:
            print("  ⚠️ 误差较大，可能pi_s和pi_b不完全匹配")
        
        self.theory_qdd = self.theory_qdd.T  # 转回去，保持一致
        
        print(f"\n  注意：图表中的tau对比是核心验证！")
        print(f"  - 绿色背景：RMSE < 0.1 Nm，参数估计非常准确")
        print(f"  - 黄色背景：RMSE 0.1-0.5 Nm，参数估计良好")  
        print(f"  - 红色背景：RMSE > 0.5 Nm，参数可能不匹配")
        
        # 计算误差
        q_error = self.sim_q - self.theory_q
        qd_error = self.sim_qd - self.theory_qd
        
        # 统计
        print("\n  位置误差 (rad):")
        print("  关节 |   RMSE   |   最大   |  平均")
        print("  " + "-"*50)
        for j in range(6):
            rmse = np.sqrt(np.mean(q_error[:, j]**2))
            max_err = np.max(np.abs(q_error[:, j]))
            mean_err = np.mean(np.abs(q_error[:, j]))
            print(f"  Joint{j+1} | {rmse:8.6f} | {max_err:8.6f} | {mean_err:8.6f}")
        
        print("\n  速度误差 (rad/s):")
        print("  关节 |   RMSE   |   最大   |  平均")
        print("  " + "-"*50)
        for j in range(6):
            rmse = np.sqrt(np.mean(qd_error[:, j]**2))
            max_err = np.max(np.abs(qd_error[:, j]))
            mean_err = np.mean(np.abs(qd_error[:, j]))
            print(f"  Joint{j+1} | {rmse:8.6f} | {max_err:8.6f} | {mean_err:8.6f}")
        
        if save_plots:
            self._plot_comparison()
    
    def _plot_comparison(self):
        print("\n  绘制对比图...")
        
        fig, axes = plt.subplots(6, 3, figsize=(18, 15))
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
        
        for j in range(6):
            # 位置
            ax_q = axes[j, 0]
            ax_q.plot(self.sim_time, self.theory_q[:, j], '-', 
                     linewidth=2, color=colors[j], alpha=0.7, label='therory')
            ax_q.plot(self.sim_time, self.sim_q[:, j], '--', 
                     linewidth=1.5, color='red', alpha=0.8, label='real')
            ax_q.set_ylabel(f'J{j+1} position (rad)')
            ax_q.grid(True, alpha=0.3)
            ax_q.legend(fontsize=8)
            if j == 0:
                ax_q.set_title('position', fontweight='bold')
            if j == 5:
                ax_q.set_xlabel('t (s)')
            
            # 误差统计
            q_err = self.sim_q[:, j] - self.theory_q[:, j]
            rmse = np.sqrt(np.mean(q_err**2))
            max_err = np.max(np.abs(q_err))
            ax_q.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMax: {max_err:.4f}',
                     transform=ax_q.transAxes, fontsize=8, 
                     verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 速度
            ax_qd = axes[j, 1]
            ax_qd.plot(self.sim_time, self.theory_qd[:, j], '-', 
                      linewidth=2, color=colors[j], alpha=0.7, label='therory')
            ax_qd.plot(self.sim_time, self.sim_qd[:, j], '--', 
                      linewidth=1.5, color='red', alpha=0.8, label='real')
            ax_qd.set_ylabel(f'J{j+1} speed (rad/s)')
            ax_qd.grid(True, alpha=0.3)
            ax_qd.legend(fontsize=8)
            if j == 0:
                ax_qd.set_title('speed', fontweight='bold')
            if j == 5:
                ax_qd.set_xlabel('t (s)')
            
            # 误差统计
            qd_err = self.sim_qd[:, j] - self.theory_qd[:, j]
            rmse = np.sqrt(np.mean(qd_err**2))
            max_err = np.max(np.abs(qd_err))
            ax_qd.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMax: {max_err:.4f}',
                      transform=ax_qd.transAxes, fontsize=8,
                      verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            # 力矩对比：关键验证！
            ax_tau = axes[j, 2]
            
            # tau_theory: 从pi_b计算的理论tau
            tau_theory = self.tau_theory[:, j]
            
            # tau_mj_inverse: 实际控制使用的tau（来自校准模型）
            tau_mj = self.tau_mj_actual[:, j]
            
            ax_tau.plot(self.sim_time, tau_theory, '-', 
                       linewidth=2, color=colors[j], alpha=0.7, label='Theory (pi_b)')
            ax_tau.plot(self.sim_time, tau_mj, '--', 
                       linewidth=1.5, color='green', alpha=0.8, label='MJ Inverse (pi_s)')
            ax_tau.set_ylabel(f'J{j+1} Torque (Nm)')
            ax_tau.grid(True, alpha=0.3)
            ax_tau.legend(fontsize=8)
            if j == 0:
                ax_tau.set_title('Tau: pi_b vs pi_s', fontweight='bold')
            if j == 5:
                ax_tau.set_xlabel('Time (s)')
            
            # tau误差统计（关键！）
            tau_err = tau_theory - tau_mj
            rmse = np.sqrt(np.mean(tau_err**2))
            max_err = np.max(np.abs(tau_err))
            rel_err = (rmse / np.mean(np.abs(tau_mj)) * 100) if np.mean(np.abs(tau_mj)) > 1e-6 else 0
            
            # 根据误差大小设置颜色
            if rmse < 0.1:
                color_box = 'lightgreen'
            elif rmse < 0.5:
                color_box = 'lightyellow'
            else:
                color_box = 'lightcoral'
            
            ax_tau.text(0.02, 0.98, f'RMSE: {rmse:.4f} Nm\nRel: {rel_err:.2f}%',
                       transform=ax_tau.transAxes, fontsize=7,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.5))
        
        # 添加总标题
        fig.suptitle('Validation: Theory (pi_b) vs Calibrated Model (pi_s)', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.99])
        
        # 确保diagram文件夹存在
        os.makedirs('diagram', exist_ok=True)
        output_path = 'diagram/sim_validation_theory_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存图表: {output_path}")
        plt.close()


def main():
    """
    
    验证目标：
    1. 模型用pi_s
    2. 控制用pi_b计算（基参数 + 期望轨迹）
    3. 如果两者一致，实际轨迹应接近期望轨迹
    """
    print("仿真验证：闭环控制校准模型")
    print("  模型: 使用pi_s更新的MuJoCo XML（完整惯性）")
    print("  控制: tau = mj_inverse(q_actual, qd_actual, qdd_des)")
    print("  验证: 实际轨迹 vs 期望轨迹 → 偏差越小，参数越准确")
    print("  策略: 闭环（与test.py一致）")
    
    # 配置路径
    # 注意：这里使用校准后的XML（已用pi_s更新）
    xml_path = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml"
    estimation_pkl = "results/estimation_results.pkl"
    mat_file = "models/ptrnSrch_N7T25QR-5.mat"
    
    print(f"\n配置:")
    print(f"  模型: {xml_path}")
    print(f"  参数: {estimation_pkl}")
    print(f"  轨迹: {mat_file}")
    
    # 创建验证器
    validator = SimulationValidator(xml_path, estimation_pkl, mat_file)
    
    try:
        # 执行验证流程（XML已由create_calibrated_model.py预先校准）
        validator.load_model()                                   # 1. 加载校准模型
        validator.load_estimation_results(method='PC-OLS-REG')  # 2. 加载pi_b, pi_fr
        validator.load_trajectory_params()                       # 3. 加载轨迹参数
        validator.compute_control_torques()                      # 4. 从pi_b计算tau
        validator.run_simulation(visualize=True)                 # 5. 用tau控制校准模型
        validator.compare_with_theory(save_plots=True)           # 6. 对比分析
        
        print("\n✅ 仿真验证完成！")
        print("  查看结果: diagram/sim_validation_theory_comparison.png")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
