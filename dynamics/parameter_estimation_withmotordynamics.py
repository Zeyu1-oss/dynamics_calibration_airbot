#!/usr/bin/env python3
"""
参数估计脚本 - 包含电机反射惯量
修改自 parameter_estimation.py，添加电机动力学支持
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import inv
import pickle
import sys
import os
from scipy.io import loadmat
from oct2py import Oct2Py
HAS_OCT2PY = True
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)
from utils.parse_urdf import parse_urdf
import h5py


def friction_motor_regressor_single(qd, qdd):
    """
    每个关节 3 个参数: [粘性Fv, 库伦Fc, 电机反射惯量I_motor]
    模型: τ_motor = τ_dyn + Fv*qd + Fc*sign(qd) + I_motor*qdd
    """
    Y_frctn_motor = np.zeros((6, 18))  # 6关节 × 3参数 = 18
    
    for i in range(6):
        Y_frctn_motor[i, 3*i:3*i+3] = [
            qd[i],           # 粘性摩擦 Fv
            np.sign(qd[i]),  # 库伦摩擦 Fc
            qdd[i]           # 电机反射惯量 I_motor * N^2
        ]
    
    return Y_frctn_motor


def friction_motor_regressor_batched(qd_matrix, qdd_matrix):
    """
    批量计算摩擦+电机回归矩阵
    """
    N = qd_matrix.shape[0]
    Y_frctn_motor_total = np.zeros((N * 6, 18))  # 6关节 × 3参数 = 18
    
    for i in range(N):
        Y_frctn_motor_i = friction_motor_regressor_single(qd_matrix[i, :], qdd_matrix[i, :])
        Y_frctn_motor_total[i*6:(i+1)*6, :] = Y_frctn_motor_i
    
    return Y_frctn_motor_total


def build_observation_matrices_oct2py(idntfcn_traj, baseQR, drv_gains):
    """
    使用Oct2Py调用MATLAB函数构建观测矩阵（包含电机动力学）
    
    Returns:
        Tau: 力矩向量 (6N,)
        Wb: 观测矩阵 (6N, n_base + 18)
    """
    
    oc = Oct2Py()
    oc.addpath('matlab')  
    oc.addpath('.')       
    
    q_matrix = idntfcn_traj['q']
    qd_fltrd_matrix = idntfcn_traj['qd_fltrd']
    q2d_est_matrix = idntfcn_traj['q2d_est']
    Tau_matrix = idntfcn_traj['i_fltrd']
    n_samples = q_matrix.shape[0]
    
    print("      使用 MATLAB 符号推导的回归器...")
    
    try:
        Y_std_total = oc.standard_regressor_airbot_batched(
            q_matrix, qd_fltrd_matrix, q2d_est_matrix
        )
    except:
        Y_std_list = []
        
        for i in range(n_samples):
            if (i + 1) % 200 == 0:
                print(f"        进度: {i+1}/{n_samples}")
            
            q_col = q_matrix[i, :].reshape(-1, 1)
            qd_col = qd_fltrd_matrix[i, :].reshape(-1, 1)
            qdd_col = q2d_est_matrix[i, :].reshape(-1, 1)
            
            Yi = oc.standard_regressor_airbot(q_col, qd_col, qdd_col)
            Y_std_list.append(Yi)
        
        Y_std_total = np.vstack(Y_std_list)
    
    # 计算摩擦+电机回归矩阵
    Y_frctn_motor_total = friction_motor_regressor_batched(qd_fltrd_matrix, q2d_est_matrix)
    
    # 构建基础观测矩阵
    n_base = baseQR['numberOfBaseParameters']
    E_full = baseQR['permutationMatrixFull']
    E1 = E_full[:, :n_base]
    
    W_dyn = Y_std_total @ E1
    Wb = np.hstack([W_dyn, Y_frctn_motor_total])
    Tau = Tau_matrix.flatten()
    
    oc.exit()
    
    print(f"      ✓ 观测矩阵: {Wb.shape} (包含 {n_base} 基础参数 + 18 摩擦/电机参数)")
    
    return Tau, Wb


def build_observation_matrices_python(idntfcn_traj, baseQR, drv_gains):
    
    try:
        sys.path.insert(0, 'matlab')
        sys.path.insert(0, 'autogen')
        from standard_regressor_airbot import standard_regressor_airbot
        print("    ✓ 成功导入Python版本的standard_regressor_airbot")
    except ImportError:
        raise ImportError("无法导入standard_regressor_airbot函数")
    
    q_matrix = idntfcn_traj['q']
    qd_fltrd_matrix = idntfcn_traj['qd_fltrd']
    q2d_est_matrix = idntfcn_traj['q2d_est']
    Tau_matrix = idntfcn_traj['i_fltrd']
    n_samples = q_matrix.shape[0]
    
    # 逐个计算标准回归矩阵
    Y_std_list = []
    for i in range(n_samples):
        if (i + 1) % 200 == 0:
            print(f"      进度: {i+1}/{n_samples}")
        
        Yi = standard_regressor_airbot(
            *q_matrix[i, :],
            *qd_fltrd_matrix[i, :],
            *q2d_est_matrix[i, :]
        )
        Y_std_list.append(Yi)
    
    Y_std_total = np.vstack(Y_std_list)
    
    # 摩擦+电机回归矩阵
    Y_frctn_motor_total = friction_motor_regressor_batched(qd_fltrd_matrix, q2d_est_matrix)
    
    # 构建基础观测矩阵
    n_base = baseQR['numberOfBaseParameters']
    E_full = baseQR['permutationMatrixFull']
    E1 = E_full[:, :n_base]
    
    W_dyn = Y_std_total @ E1
    Wb = np.hstack([W_dyn, Y_frctn_motor_total])
    Tau = Tau_matrix.flatten()
    
    return Tau, Wb


def build_observation_matrices(idntfcn_traj, baseQR, drv_gains):
    return build_observation_matrices_oct2py(idntfcn_traj, baseQR, drv_gains)


def ordinary_least_square_estimation(Tau, Wb, baseQR):
    """OLS 估计"""
    pi_OLS = np.linalg.lstsq(Wb, Tau, rcond=1e-15)[0]
    
    n_b = baseQR['numberOfBaseParameters']
    pi_b_OLS = pi_OLS[:n_b]
    pi_frctn_motor_OLS = pi_OLS[n_b:]  # 18个参数（6×3）
    
    return pi_b_OLS, pi_frctn_motor_OLS


def physically_consistent_estimation(Tau, Wb, baseQR, pi_urdf=None, lambda_reg=0):
    """
    物理一致性约束的参数估计（包含电机动力学）
    """
    
    n_b = baseQR['numberOfBaseParameters']
    n_d = 60 - n_b
    
    print(f"    基础参数: {n_b}, 依赖参数: {n_d}")
    print(f"    摩擦+电机参数: 18 (6关节 × 3参数: Fv, Fc, I_motor)")
    
    # 优化变量
    pi_frctn_motor = cp.Variable(18)  # [Fv1, Fc1, Im1, Fv2, Fc2, Im2, ...]
    pi_b = cp.Variable(n_b)
    pi_d = cp.Variable(n_d)
    
    # 参数映射
    mapping_matrix = np.block([
        [np.eye(n_b), -baseQR['beta']],
        [np.zeros((n_d, n_b)), np.eye(n_d)]
    ])
    E = baseQR['permutationMatrixFull']
    pii = E @ mapping_matrix @ cp.hstack([pi_b, pi_d])
    
    constraints = []
    
    # 1. 质量约束
    mass_indices = list(range(9, 60, 10))
    mass_urdf = np.array([0.607, 0.918, 0.7, 0.359, 0.403, 0.11])
    error_range = 0.10
    mass_upper = mass_urdf * (1 + error_range)
    mass_lower = mass_urdf * (1 - error_range)
    
    for i, idx in enumerate(mass_indices):
        constraints.append(pii[idx] >= max(1e-3, mass_lower[i]))
        constraints.append(pii[idx] <= mass_upper[i])
    
    print(f"    质量约束: 6个link, 误差范围±{error_range*100:.0f}%")
    
    # 2. 惯性约束（正定性 + 三角不等式）
    for link_idx in range(6):
        i = link_idx * 10
        
        I_link = cp.vstack([
            cp.hstack([pii[i],     pii[i+1], pii[i+2]]),
            cp.hstack([pii[i+1],   pii[i+3], pii[i+4]]),
            cp.hstack([pii[i+2],   pii[i+4], pii[i+5]])
        ])
        
        h_link = pii[i+6:i+9]
        m_link = pii[i+9]
        
        epsilon_triangle = 5e-4
        epsilon_inertia = 1e-5
        
        L_link = cp.vstack([
            cp.hstack([0.5*(pii[i+3]+pii[i+5]-pii[i]), -pii[i+1], -pii[i+2]]),
            cp.hstack([-pii[i+1], 0.5*(pii[i]+pii[i+5]-pii[i+3]), -pii[i+4]]),
            cp.hstack([-pii[i+2], -pii[i+4], 0.5*(pii[i]+pii[i+3]-pii[i+5])])
        ]) - 0.5 * epsilon_triangle * np.eye(3)
        
        h_link_col = cp.reshape(h_link, (3, 1), order='C')
        h_link_row = cp.reshape(h_link, (1, 3), order='C')
        m_link_reshaped = cp.reshape(m_link, (1, 1), order='C')
        
        D_link = cp.bmat([
            [L_link, h_link_col],
            [h_link_row, m_link_reshaped]
        ])
        
        constraints.append(D_link >> 0)
        
        # 对角元素正定
        constraints.append(pii[i] >= epsilon_inertia)
        constraints.append(pii[i+3] >= epsilon_inertia)
        constraints.append(pii[i+5] >= epsilon_inertia)
        
        # 质心约束
        h_max_component = 0.10
        constraints.append(h_link[0] >= -h_max_component)
        constraints.append(h_link[0] <= h_max_component)
        constraints.append(h_link[1] >= -h_max_component)
        constraints.append(h_link[1] <= h_max_component)
        constraints.append(h_link[2] >= -h_max_component)
        constraints.append(h_link[2] <= h_max_component)
    
    # 3. 摩擦+电机参数约束
    print("    摩擦+电机约束: Fv > 0, Fc > 0, I_motor > 0")
    
    epsilon_friction = 1e-12
    epsilon_motor = 1e-6  # 电机惯量下界（kg·m²）
    
    # 典型电机反射惯量范围（根据关节大小）
    I_motor_max = np.array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1])  # kg·m²
    
    for i in range(6):
        # Fv >= 0
        constraints.append(pi_frctn_motor[3*i] >= epsilon_friction)
        
        # Fc >= 0
        constraints.append(pi_frctn_motor[3*i + 1] >= epsilon_friction)
        
        constraints.append(pi_frctn_motor[3*i + 2] >= epsilon_motor)
        constraints.append(pi_frctn_motor[3*i + 2] <= I_motor_max[i])
    
    # 4. 目标函数
    tau_error = cp.norm(Tau - Wb @ cp.hstack([pi_b, pi_frctn_motor]))
    
    use_regularization = (pi_urdf is not None) and (lambda_reg > 0)
    
    if use_regularization:
        if len(pi_urdf) != 60:
            print(f"  ⚠️  pi_urdf长度为{len(pi_urdf)}，期望60")
            pi_urdf_padded = np.zeros(60)
            pi_urdf_padded[:min(60, len(pi_urdf))] = pi_urdf[:min(60, len(pi_urdf))]
            pi_urdf = pi_urdf_padded
        
        param_regularization = lambda_reg * cp.norm(pii[:60] - pi_urdf)
        objective = tau_error + param_regularization
        print(f"    目标 min ||Wb * pi_b - tau||² + λ * ||pi_s - pi_urdf||²")
    else:
        objective = tau_error
        print("    目标 = tau误差 (无正则化)")
    
    # 5. 求解SDP
    print("  求解SDP优化问题...")
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        try:
            result = problem.solve(
                solver=cp.MOSEK,
                verbose=False,
                mosek_params={
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-12,
                    'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-12,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-12,
                    'MSK_DPAR_INTPNT_CO_TOL_MU_RED': 1e-12,
                    'MSK_IPAR_INTPNT_MAX_ITERATIONS': 1000,
                }
            )
        except Exception as mosek_err:
            # print(f"  MOSEK不可用: {mosek_err}")
            result = problem.solve(
                solver=cp.SCS,
                verbose=False,
                max_iters=50000,
                eps_abs=1e-9,
                eps_rel=1e-9,
                eps_infeas=1e-12,
                alpha=1.5,
                rho_x=1e-6,
                scale=5.0,
                normalize=True,
                adaptive_scale=True,
                acceleration_lookback=20,
                acceleration_interval=1,
                warm_start=True,
            )
            print(f"  SCS求解完成 (状态: {problem.status})")
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            result = problem.solve(
                solver=cp.CVXOPT, 
                verbose=False,
                abstol=1e-12,
                reltol=1e-12,
                feastol=1e-12,
                max_iters=500
            )
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            print(f"  ✓ SDP求解成功! (状态: {problem.status})")
            if problem.status == 'optimal_inaccurate':
                print("    注意: 求解精度可能不够")
        else:
            print(f"    ⚠️  求解状态: {problem.status}")
    
    except Exception as e:
        print(f"  ❌ SDP求解失败: {e}")
        raise
    
    # 6. 提取结果
    pi_b_SDP = pi_b.value
    pi_frctn_motor_SDP = pi_frctn_motor.value
    pi_d_value = pi_d.value
    
    if pi_b_SDP is None or pi_frctn_motor_SDP is None:
        raise RuntimeError("SDP求解失败")
    
    # 计算完整标准参数
    pi_full = E @ mapping_matrix @ np.concatenate([pi_b_SDP, pi_d_value])
    mass_estimated = pi_full[mass_indices]
    
    # 验证结果
    tau_pred = Wb @ np.concatenate([pi_b_SDP, pi_frctn_motor_SDP])
    tau_pred_error = np.linalg.norm(Tau - tau_pred)
    rel_tau_error = 100 * tau_pred_error / np.linalg.norm(Tau)
    print(f"  Tau预测误差: {tau_pred_error:.4e} ({rel_tau_error:.2f}%)")
    
    if use_regularization:
        param_deviation = np.linalg.norm(pi_full[:60] - pi_urdf)
        print(f"  参数偏离URDF: {param_deviation:.4e}")
    
    # 显示结果
    print("\n  连杆 | URDF质量 | 估计质量 | 相对误差")
    print("  " + "-"*50)
    for i in range(6):
        rel_error = 100 * (mass_estimated[i] - mass_urdf[i]) / mass_urdf[i]
        print(f"  Link{i+1} | {mass_urdf[i]:7.4f}kg | {mass_estimated[i]:7.4f}kg | {rel_error:+7.2f}%")
    
    print("\n  关节 |    Fv (粘性)    |    Fc (库伦)    | I_motor (反射惯量)")
    print("  " + "-"*70)
    for i in range(6):
        Fv = pi_frctn_motor_SDP[3*i]
        Fc = pi_frctn_motor_SDP[3*i + 1]
        Im = pi_frctn_motor_SDP[3*i + 2]
        print(f"  J{i+1}    | {Fv:14.6e} | {Fc:14.6e} | {Im:14.6e} kg·m²")
    
    # 检查惯性矩阵
    print("\n  检查惯性矩阵物理约束:")
    for link_idx in range(6):
        i = link_idx * 10
        Ixx, Ixy, Ixz = pi_full[i], pi_full[i+1], pi_full[i+2]
        Iyy, Iyz = pi_full[i+3], pi_full[i+4]
        Izz = pi_full[i+5]
        
        I_val = np.array([
            [Ixx, Ixy, Ixz],
            [Ixy, Iyy, Iyz],
            [Ixz, Iyz, Izz]
        ])
        eig_vals = np.linalg.eigvalsh(I_val)
        
        positive_definite = np.all(eig_vals > -1e-10)
        
        margin1 = (Ixx + Iyy) - Izz
        margin2 = (Ixx + Izz) - Iyy
        margin3 = (Iyy + Izz) - Ixx
        min_margin = min(margin1, margin2, margin3)
        triangle_satisfied = min_margin >= -1e-10
        
        status = "✓" if (positive_definite and triangle_satisfied) else "✗"
        
        print(f"    Link{link_idx+1}: {status} (λ_min={np.min(eig_vals):.4e}, 三角余量={min_margin:.4e})")
    
    return pi_b_SDP, pi_frctn_motor_SDP, pi_full, mass_estimated



def parse_ur_data(path_to_data, idx_start, idx_end):
    import pandas as pd
    
    if not os.path.exists(path_to_data):
        raise FileNotFoundError(f"数据文件不存在: {path_to_data}")
    
    print(f"  从 {path_to_data} 加载数据...")
    df = pd.read_csv(path_to_data)
    
    if idx_end >= len(df):
        print(f"  ⚠️  索引超出范围，调整为: [{idx_start}, {len(df)-1}]")
        idx_end = len(df) - 1
    
    data = df.iloc[idx_start:idx_end+1]
    
    data_dict = {
        't': data.iloc[:, 0].values,
        'q': data.iloc[:, 1:7].values,
        'qd': data.iloc[:, 7:13].values,
        'i': data.iloc[:, 13:19].values,
    }
    
    data_dict['qd_fltrd'] = data_dict['qd'].copy()
    data_dict['q2d_est'] = np.zeros_like(data_dict['qd'])
    data_dict['i_fltrd'] = data_dict['i'].copy()
    
    print(f"  ✓ 加载了 {len(data_dict['t'])} 个数据点")
    return data_dict


def filter_data(data_dict, fs=500.0): 
    from scipy.signal import butter, filtfilt
    
    print(f"  过滤数据 (采样率 fs={fs} Hz)...")
    
    nyq = 0.5 * fs
    normal_cutoff_vel = 0.15 
    N_order = 5
    
    b_vel, a_vel = butter(N_order, normal_cutoff_vel, btype='low', analog=False)
    
    qd_fltrd = np.zeros_like(data_dict['qd'])
    for i in range(6):
        qd_fltrd[:, i] = filtfilt(b_vel, a_vel, data_dict['qd'][:, i])
    data_dict['qd_fltrd'] = qd_fltrd
    
    dt = np.mean(np.diff(data_dict['t']))
    if dt <= 0:
        dt = 1.0 / fs
    
    q2d_est = np.zeros_like(data_dict['qd_fltrd'])
    N_samples = data_dict['qd_fltrd'].shape[0]
    
    for i in range(1, N_samples - 1):
       dlta_qd_fltrd =  data_dict['qd_fltrd'][i+1,:] - data_dict['qd_fltrd'][i-1,:]
       dlta_t_msrd = data_dict['t'][i+1] - data_dict['t'][i-1]
       q2d_est[i,:] = dlta_qd_fltrd / dlta_t_msrd

    data_dict['q2d_est'] = q2d_est
    
    for i in range(6):
        data_dict['q2d_est'][:, i] = filtfilt(b_vel, a_vel, data_dict['q2d_est'][:, i])
    
    normal_cutoff_curr = 0.20 
    b_curr, a_curr = butter(N_order, normal_cutoff_curr, btype='low', analog=False)
    
    i_fltrd = np.zeros_like(data_dict['i'])
    for i in range(6):
        i_fltrd[:, i] = filtfilt(b_curr, a_curr, data_dict['i'][:, i])
    data_dict['i_fltrd'] = i_fltrd
    
    print("  ✓ 数据过滤完成")
    return data_dict


def load_urdf_parameters(urdf_path):
    if not os.path.exists(urdf_path):
        print(f"  ⚠️  URDF文件不存在: {urdf_path}")
        return np.zeros(60)
    
    try:
        sys.path.insert(0, 'utils')
        robot = parse_urdf(urdf_path)
        pi_urdf_6links = robot['pi'].flatten('F')
        return pi_urdf_6links
    except Exception as e:
        print(f"  ⚠️  加载URDF失败: {e}")
        return np.zeros(60)


def print_estimation_summary(sol, Tau, tau_pred):
    print(f"  基础参数数量: {len(sol['pi_b'])}")
    print(f"  摩擦+电机参数数量: {len(sol['pi_fr_motor'])}")
    
    residual = Tau - tau_pred
    rmse = np.sqrt(np.mean(residual**2))
    r_squared = 1 - np.sum(residual**2) / np.sum((Tau - np.mean(Tau))**2)
    
    print(f"\n  拟合质量:")
    print(f"    RMSE: {rmse:.4e}")
    print(f"    R²: {r_squared:.6f}")
    
    if sol['masses'] is not None:
        print(f"\n  估计的总质量: {np.sum(sol['masses']):.4f} kg")


def estimate_dynamic_params(path_to_data, idx, drv_gains, baseQR, method='PC-OLS-REG',
                           lambda_reg=1e-3, urdf_path=None):
    """
    主估计函数（包含电机动力学）
    """
    
    if isinstance(path_to_data, str):
        path_to_data = [path_to_data]
        idx = [idx]
    
    n_trajectories = len(path_to_data)
    print(f"  使用 {n_trajectories} 条轨迹进行参数估计")
    print(f"  模型: 刚体动力学 + 摩擦 + 电机反射惯量")
    
    # 1. 加载和处理数据
    print("\n步骤 1/4: 加载和处理数据...")
    all_Tau = []
    all_Wb = []
    
    for traj_idx, (data_path, data_range) in enumerate(zip(path_to_data, idx)):
        print(f"\n  轨迹 {traj_idx+1}/{n_trajectories}: {data_path}")
        print(f"    数据范围: [{data_range[0]}, {data_range[1]}]")
        
        idntfcn_traj = parse_ur_data(data_path, data_range[0], data_range[1])
        
        dt = np.mean(np.diff(idntfcn_traj['t']))
        fs_actual = 1.0 / dt
        print(f"    检测到采样率: {fs_actual:.2f} Hz")
        
        idntfcn_traj = filter_data(idntfcn_traj, fs=fs_actual)
        
        print(f"    构建观测矩阵...")
        Tau_i, Wb_i = build_observation_matrices(idntfcn_traj, baseQR, drv_gains)
        print(f"    ✓ 观测矩阵: {Wb_i.shape}, 力矩向量: {Tau_i.shape}")
        
        all_Tau.append(Tau_i)
        all_Wb.append(Wb_i)
    
    # 2. 合并数据
    print("\n步骤 2/4: 合并多轨迹数据...")
    Tau = np.concatenate(all_Tau)
    Wb = np.concatenate(all_Wb)
    n_base = baseQR['numberOfBaseParameters']
    
    print(f"  合并后: Tau={Tau.shape}, Wb={Wb.shape}")
    print(f"  参数结构: {n_base} 基础 + 18 摩擦/电机 = {Wb.shape[1]} 总参数")
    
    # 3. 估计参数
    print(f"\n步骤 3/4: 参数估计 ({method})...")
    
    sol = {}
    
    if method == 'OLS':
        pi_b, pi_fr_motor = ordinary_least_square_estimation(Tau, Wb, baseQR)
        sol['pi_b'] = pi_b
        sol['pi_fr_motor'] = pi_fr_motor
        sol['pi_s'] = None
        sol['masses'] = None
        
    elif method in ['PC-OLS', 'PC-OLS-REG']:
        pi_urdf = None
        if method == 'PC-OLS-REG':
            if urdf_path is None:
                raise ValueError("PC-OLS-REG需要urdf_path")
            print(f"  加载URDF参考参数: {urdf_path}")
            pi_urdf = load_urdf_parameters(urdf_path)
        
        pi_b, pi_fr_motor, pi_s, masses = physically_consistent_estimation(
            Tau, Wb, baseQR, pi_urdf, lambda_reg
        )
        sol['pi_b'] = pi_b
        sol['pi_fr_motor'] = pi_fr_motor
        sol['pi_s'] = pi_s
        sol['masses'] = masses
    else:
        raise ValueError(f"未知方法: {method}")
    
    # 4. 统计分析
    print("\n步骤 4/4: 统计分析...")
    
    tau_pred = Wb @ np.concatenate([sol['pi_b'], sol['pi_fr_motor']])
    residual = Tau - tau_pred
    
    n_samples = Wb.shape[0]
    n_params = Wb.shape[1]
    sqrd_sigma_e = np.linalg.norm(residual)**2 / (n_samples - n_params)
    
    try:
        Wb_inv = inv(Wb.T @ Wb)
        Cpi = sqrd_sigma_e * Wb_inv
        sol['std'] = np.sqrt(np.diag(Cpi))
    except np.linalg.LinAlgError:
        sol['std'] = np.ones(len(sol['pi_b']) + len(sol['pi_fr_motor']))
    
    params_all = np.concatenate([sol['pi_b'], sol['pi_fr_motor']])
    params_all_safe = np.where(np.abs(params_all) < 1e-10, 1e-10, params_all)
    sol['rel_std'] = 100 * sol['std'] / np.abs(params_all_safe)
    
    print_estimation_summary(sol, Tau, tau_pred)
    
    return sol


def main():
    print("参数估计脚本 - 包含电机反射惯量")
    
    mat_filename_standard = 'models/baseQR_standard.mat'
    
    print(f"\n加载base_params_qr.mat: {mat_filename_standard}")
    
    try:
        if not os.path.exists(mat_filename_standard):
            raise FileNotFoundError(f"文件不存在: {mat_filename_standard}")
            
        with h5py.File(mat_filename_standard, 'r') as f:
            print("    ✓ 使用 h5py 读取 HDF5 格式...")
            
            baseQR_group = f['baseQR']
            
            bb = np.array(baseQR_group['numberOfBaseParameters']).flatten()[0]
            E_full = np.array(baseQR_group['permutationMatrix']).T
            beta = np.array(baseQR_group['beta']).T
            # motorDynamicsIncluded = bool(np.array(baseQR_group['motorDynamicsIncluded']).flatten()[0])
            motorDynamicsIncluded = bool(np.array(baseQR_group['motorDynamicsIncluded']).flatten()[0])

            baseQR = {
                'numberOfBaseParameters': int(bb),
                'permutationMatrixFull': E_full,
                'beta': beta,
                'motorDynamicsIncluded': motorDynamicsIncluded
            }

        if baseQR['beta'].ndim == 1:
            baseQR['beta'] = baseQR['beta'].reshape(-1, 1)
        
        print(f"    ✓ 基础参数数量: {baseQR['numberOfBaseParameters']}")
        print(f"    ✓ 包含电机动力学: {baseQR['motorDynamicsIncluded']}")

    except FileNotFoundError:
        sys.exit(1)
    except Exception as e:
        print(f"    ❌ 加载失败: {e}")
        sys.exit(1)
    
    # 配置
    METHOD = 'PC-OLS-REG'
    LAMBDA_REG = 0.0001
    USE_MULTIPLE_TRAJECTORIES = True
    
    if USE_MULTIPLE_TRAJECTORIES:
        data_paths = [
            'state_machine_demo/real_data/vali_ptrnSrch_N7T25QR-6(1).csv',
            'state_machine_demo/real_data/vali_ptrnSrch_N7T25QR-5(1).csv',
        ]
        data_ranges = [
            [0, 4800],
            [0, 4800],
        ]
    else:
        data_paths = 'state_machine_demo/real_data/vali_ptrnSrch_N7T25QR-6.csv'
        data_ranges = [0, 4800]
    
    drv_gains = np.ones(6)
    urdf_path = 'state_machine_demo/resources/urdf/play.urdf'
    
    # 检查数据
    if USE_MULTIPLE_TRAJECTORIES:
        for i, path in enumerate(data_paths):
            if not os.path.exists(path):
                print(f"❌ 数据文件不存在: {path}")
                sys.exit(1)
    else:
        if not os.path.exists(data_paths):
            print(f"❌ 数据文件不存在: {data_paths}")
            sys.exit(1)
    
    # 运行估计
    sol = estimate_dynamic_params(
        path_to_data=data_paths,
        idx=data_ranges,
        drv_gains=drv_gains,
        baseQR=baseQR,
        method=METHOD,
        lambda_reg=LAMBDA_REG,
        urdf_path=urdf_path
    )
    
    # 保存结果
    os.makedirs('results', exist_ok=True)
    result_path = 'results/estimation_results_with_motor.pkl'
    
    save_data = {
        'sol': sol,
        'method': METHOD,
        'include_motor_dynamics': True,
        'use_multiple_trajectories': USE_MULTIPLE_TRAJECTORIES
    }
    if METHOD == 'PC-OLS-REG':
        save_data['lambda_reg'] = LAMBDA_REG
    
    with open(result_path, 'wb') as f:
        pickle.dump(save_data, f)
    
    print(f"\n✓ 保存到 {result_path}")


if __name__ == "__main__":
    main()