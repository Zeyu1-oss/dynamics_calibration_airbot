#!/usr/bin/env python3
"""
仿真验证脚本：使用估计参数创建新机器人，对比理论轨迹

流程:
1. 加载估计参数pi_s: 10参数/link, pi_fr: 摩擦参数）
2. 更新机器人模型 参数质量、惯性、重心、damping、frictionloss
3. 从MAT文件加载理论激励轨迹参数
4. 使用最小参数和期待轨迹构建tau
5. 仿真执行力矩控制（带可视化）
6. 对比理论轨迹与实际轨迹
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
    Returns:
        qd: 期望位置 (n_joints, n_times)
        qdot_d: 期望速度 (n_joints, n_times)
        qddot_d: 期望加速度 (n_joints, n_times)
    """
    t_vec = np.atleast_1d(t_vec)
    J = a.shape[0]  # 关节数
    M = len(t_vec)  # 时间点数
    qd, qdot_d, qddot_d = np.zeros((J, M)), np.zeros((J, M)), np.zeros((J, M))
    tau_vec = t_vec % T  
    
    for i in range(J):
        k_vec = np.arange(1, N + 1).reshape(-1, 1)
        wk_t = wf * k_vec * t_vec
        sin_wk_t, cos_wk_t = np.sin(wk_t), np.cos(wk_t)
        
        # 傅里叶部分
        a_norm = a[i, :].reshape(-1, 1) / (wf * k_vec)
        b_norm = b[i, :].reshape(-1, 1) / (wf * k_vec)
        qd_fourier = (a_norm * sin_wk_t - b_norm * cos_wk_t).sum(axis=0)
        qdot_d_fourier = (a[i, :].reshape(-1, 1) * cos_wk_t + 
                         b[i, :].reshape(-1, 1) * sin_wk_t).sum(axis=0)
        qddot_d_fourier = ((-a[i, :].reshape(-1, 1) * wf * k_vec) * sin_wk_t + 
                          (b[i, :].reshape(-1, 1) * wf * k_vec) * cos_wk_t).sum(axis=0)
        
        # 多项式部分
        qd_poly = np.zeros(M)
        qdot_d_poly = np.zeros(M)
        qddot_d_poly = np.zeros(M)
        for k_exp in range(6):
            c = c_pol[i, k_exp]
            qd_poly += c * (tau_vec ** k_exp)
            if k_exp >= 1: 
                qdot_d_poly += c * k_exp * (tau_vec ** (k_exp - 1))
            if k_exp >= 2: 
                qddot_d_poly += c * k_exp * (k_exp - 1) * (tau_vec ** (k_exp - 2))
        
        # 总和
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
        
        # 理论轨迹
        self.theory_q = None
        self.theory_qd = None
        self.theory_qdd = None
    
    def load_estimation_results(self, method='PC-OLS-REG'):
        """加载估计结果"""
        print(f"\n[1/5] 加载估计参数 ({method})...")
        
        with open(self.estimation_pkl_path, 'rb') as f:
            all_results = pickle.load(f)
        
        if method == 'PC-OLS-REG':
            self.estimation_results = all_results['sol_pc']
        else:
            self.estimation_results = all_results['sol_ols']
        
        print(f"  ✓ pi_b: {self.estimation_results['pi_b'].shape}")
        print(f"  ✓ pi_fr: {self.estimation_results['pi_fr'].shape}")
        if self.estimation_results.get('pi_s') is not None:
            print(f"  ✓ pi_s: {self.estimation_results['pi_s'].shape}")
    
    def load_trajectory_params(self):
        """加载理论轨迹参数"""
        print(f"\n[3/6] 加载理论轨迹参数...")
        
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
        
        print(f"  ✓ 周期 T: {self.traj_params['T']}s")
        print(f"  ✓ 傅里叶项数 N: {self.traj_params['N']}")
        print(f"  ✓ 初始位置 q0: {self.traj_params['q0']}")
    
    def update_model_parameters(self):
        """更新模型参数10参数/link + damping + frictionloss"""
        print(f"\n[2/6] 更新机器人模型参数...")
        
        # 加载模型
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
        """使用理论轨迹和识别的最小参数计算控制力矩"""
        print(f"\n[4/6] 计算控制力矩（用识别的参数）...")
        
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
        
        # 生成时间序列（从0到T，步长0.002s）
        dt_traj = 0.002
        time_vec = np.arange(0, T, dt_traj)
        n_samples = len(time_vec)
        
        print(f"  生成理论轨迹: {n_samples}个采样点")
        
        # 计算完整理论轨迹
        q_theory, qd_theory, qdd_theory = mixed_trajectory_calculator(
            time_vec, T, N, wf, a, b, c_pol, q0
        )
        
        # 转置为(n_samples, 6)格式
        q_theory = q_theory.T
        qd_theory = qd_theory.T
        qdd_theory = qdd_theory.T
        
        # 构造trajectory_data字典（与parameter_estimation.py格式一致）
        traj_data = {
            't': time_vec,
            'q': q_theory,
            'qd_fltrd': qd_theory,
            'q2d_est': qdd_theory,
            'i_fltrd': np.zeros_like(q_theory)  # 占位，不使用
        }
        
        # 加载baseQR
        print(f"  加载baseQR...")
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
        
        print(f"  ✓ baseQR加载完成: n_base={n_base}")
        
        # 使用build_observation_matrices批量构建（与parameter_estimation.py一致）
        print(f"  构建观测矩阵（批量）...")
        Tau_dummy, Wb = build_observation_matrices(traj_data, baseQR, np.ones(6))
        
        print(f"  ✓ 观测矩阵: {Wb.shape}")
        
        # 计算控制力矩：tau = Wb @ [pi_b; pi_fr]
        print(f"  计算控制力矩...")
        params = np.concatenate([
            self.estimation_results['pi_b'],
            self.estimation_results['pi_fr']
        ])
        
        Tau_control = Wb @ params
        
        # 重塑为(n_samples, 6)
        tau_control = Tau_control.reshape(-1, 6)
        
        # 保存结果
        self.theory_time = time_vec
        self.theory_q = q_theory
        self.theory_qd = qd_theory
        self.theory_qdd = qdd_theory
        self.tau_control = tau_control
        
        print(f"  ✓ 控制力矩计算完成")
        print(f"  力矩形状: {tau_control.shape}")
        print(f"  力矩范围: [{np.min(tau_control):.3f}, {np.max(tau_control):.3f}] Nm")
    
    def run_simulation(self, visualize=True):
        print(f"\n[5/6] 运行仿真 (可视化={visualize})...")
        
        # 重置仿真
        mujoco.mj_resetData(self.model, self.data)
        
        # 设置初始状态
        self.data.qpos[:6] = self.traj_params['q0']
        self.data.qvel[:6] = 0.0
        
        # 仿真参数
        dt = self.model.opt.timestep
        sim_time = self.traj_params['T']
        n_steps = int(sim_time / dt)
        
        print(f"  仿真时间: {sim_time}s")
        print(f"  时间步: {dt:.6f}s")
        print(f"  总步数: {n_steps}")
        
        # 创建tau插值函数（从预计算的tau_control）
        from scipy.interpolate import interp1d
        tau_interp = []
        for j in range(6):
            tau_interp.append(
                interp1d(self.theory_time, self.tau_control[:, j], 
                        kind='linear', fill_value='extrapolate')
            )
        
        # 可视化器
        viewer = None
        if visualize:
            viewer = mujoco.viewer.launch_passive(self.model, self.data)
            print("  ✓ 可视化已开启")
        
        # 仿真循环
        record_every = max(1, int(0.002 / dt))
        
        for step in range(n_steps):
            t = self.data.time
            
            # 获取当前时刻的控制力矩（从预计算的值插值）
            tau_current = np.array([tau_interp[j](t) for j in range(6)])
            
            # 应用力矩控制
            self.data.ctrl[:6] = tau_current
            
            # 仿真一步
            mujoco.mj_step(self.model, self.data)
            
            # 更新可视化（在仿真步后立即更新）
            if visualize and viewer is not None:
                viewer.sync()
            
            # 记录数据
            if step % record_every == 0:
                self.sim_time.append(self.data.time)
                self.sim_q.append(self.data.qpos[:6].copy())
                self.sim_qd.append(self.data.qvel[:6].copy())
                self.sim_tau.append(tau_current.copy())
            
            # 进度显示
            if (step + 1) % 10000 == 0:
                print(f"    进度: {step+1}/{n_steps} ({100*(step+1)/n_steps:.1f}%)")
        
        if viewer is not None:
            viewer.close()
            print("  ✓ 可视化已关闭")
        
        # 转换为numpy数组
        self.sim_time = np.array(self.sim_time)
        self.sim_q = np.array(self.sim_q)
        self.sim_qd = np.array(self.sim_qd)
        self.sim_tau = np.array(self.sim_tau)
        
        print(f"  ✓ 仿真完成，记录了 {len(self.sim_time)} 个数据点")
    
    def compare_with_theory(self, save_plots=True):
        """对比理论轨迹"""
        print(f"\n[6/6] 对比理论轨迹...")
        
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
            
            # 力矩
            ax_tau = axes[j, 2]
            ax_tau.plot(self.sim_time, self.sim_tau[:, j], '-', 
                       linewidth=1.5, color=colors[j], alpha=0.8)
            ax_tau.set_ylabel(f'J{j+1} tau (Nm)')
            ax_tau.grid(True, alpha=0.3)
            if j == 0:
                ax_tau.set_title('control torque', fontweight='bold')
            if j == 5:
                ax_tau.set_xlabel('t (s)')
        
        plt.tight_layout()
        plt.savefig('sim_validation_theory_comparison.png', dpi=300, bbox_inches='tight')
        print("  ✓ 保存图表: sim_validation_theory_comparison.png")
        plt.close()


def main():
    print("仿真验证：使用估计参数控制新机器人，对比理论轨迹")
    
    # 配置路径
    xml_path = "models/mjcf/manipulator/airbot_play_force/_play_force.xml"
    estimation_pkl = "results/estimation_results.pkl"
    mat_file = "models/ptrnSrch_N7T25QR-5.mat"
    
    # 创建验证器
    validator = SimulationValidator(xml_path, estimation_pkl, mat_file)
    
    try:
        # 执行验证流程（按正确顺序）
        validator.load_estimation_results(method='PC-OLS-REG')  # 1. 加载估计参数
        validator.update_model_parameters()                      # 2. 先更新模型参数
        validator.load_trajectory_params()                       # 3. 加载轨迹参数
        validator.compute_control_torques()                      # 4. 计算控制力矩
        validator.run_simulation(visualize=True)                 # 5. 运行仿真（可视化）
        validator.compare_with_theory(save_plots=True)           # 6. 对比分析
        
        print("✅ 仿真验证完成！")
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
