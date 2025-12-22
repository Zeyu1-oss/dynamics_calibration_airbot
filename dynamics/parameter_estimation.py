
import numpy as np
import cvxpy as cp
from scipy.linalg import inv
import pickle
import sys
import os
from scipy.io import loadmat
from oct2py import Oct2Py
HAS_OCT2PY = True

import h5py

def friction_regressor_single(qd):
    """
    每个关节2个摩擦参数: [粘性, 库伦]
    """
    Y_frctn = np.zeros((6, 12))
    
    for i in range(6):
        Y_frctn[i, 2*i:2*i+2] = [
            qd[i],           # 粘性摩擦
            np.sign(qd[i])   # 库伦摩擦
        ]
    
    return Y_frctn


def friction_regressor_batched(qd_matrix):
    """
    
    Args:
        qd_matrix: 关节速度矩阵 (N, 6)
    
    Returns:
        Y_frctn_total: 摩擦回归矩阵 (6N, 12)
    """
    N = qd_matrix.shape[0]
    Y_frctn_total = np.zeros((N * 6, 12))
    
    for i in range(N):
        Y_frctn_i = friction_regressor_single(qd_matrix[i, :])
        Y_frctn_total[i*6:(i+1)*6, :] = Y_frctn_i
    
    return Y_frctn_total



def build_observation_matrices_oct2py(idntfcn_traj, baseQR, drv_gains):
    """
    使用Oct2Py调用MATLAB函数构建观测矩阵
    
    Args:
        idntfcn_traj: 轨迹数据字典
        baseQR: QR分解结果
        drv_gains: 驱动增益
    
    Returns:
        Tau: 力矩向量 (6N,)
        Wb: 观测矩阵 (6N, n_base + 12)
    """
    print("  使用Oct2Py方法构建观测矩阵...")
    
    oc = Oct2Py()
    oc.addpath('autogen')  # 添加MATLAB函数路径
    oc.addpath('matlab')   # 如果有其他路径
    oc.addpath('.')        # 当前目录
    
    # 准备数据
    q_matrix = idntfcn_traj['q']
    qd_fltrd_matrix = idntfcn_traj['qd_fltrd']
    q2d_est_matrix = idntfcn_traj['q2d_est']
    Tau_matrix = idntfcn_traj['i_fltrd']
    n_samples = q_matrix.shape[0]
    
    try:
        # 尝试调用批量函数（如果存在）
        Y_std_total = oc.standard_regressor_airbot_batched(
            q_matrix, qd_fltrd_matrix, q2d_est_matrix
        )
    except:
        Y_std_list = []
        
        for i in range(n_samples):
            if (i + 1) % 200 == 0:
                print(f"      进度: {i+1}/{n_samples}")
            
            q_col = q_matrix[i, :].reshape(-1, 1)
            qd_col = qd_fltrd_matrix[i, :].reshape(-1, 1)
            qdd_col = q2d_est_matrix[i, :].reshape(-1, 1)
            
            Yi = oc.standard_regressor_airbot(q_col, qd_col, qdd_col)
            Y_std_list.append(Yi)
        
        Y_std_total = np.vstack(Y_std_list)
    
    # 计算摩擦回归矩阵
    print("    计算摩擦回归矩阵...")
    Y_frctn_total = friction_regressor_batched(qd_fltrd_matrix)
    
    # 构建基础观测矩阵
    n_base = baseQR['numberOfBaseParameters']
    E_full = baseQR['permutationMatrixFull']
    E1 = E_full[:, :n_base]
    
    W_dyn = Y_std_total @ E1
    Wb = np.hstack([W_dyn, Y_frctn_total])
    
    # 力矩向量
    Tau = Tau_matrix.flatten()
    
    # 清理
    oc.exit()
    
    print(f"  ✓ 观测矩阵构建完成: {Wb.shape}")
    
    return Tau, Wb



def build_observation_matrices_python(idntfcn_traj, baseQR, drv_gains):
    """
    
    Args:
        idntfcn_traj: 轨迹数据字典
        baseQR: QR分解结果
        drv_gains: 驱动增益
    
    Returns:
        Tau: 力矩向量 (6N,)
        Wb: 观测矩阵 (6N, n_base + 12)
    """
    print("  使用纯Python方法构建观测矩阵...")
    
    # 尝试导入Python版本的regressor
    try:
        sys.path.insert(0, 'matlab')
        sys.path.insert(0, 'autogen')
        from standard_regressor_airbot import standard_regressor_airbot
        print("    ✓ 成功导入Python版本的standard_regressor_airbot")
    except ImportError:
        raise ImportError(
            "无法导入standard_regressor_airbot函数。\n"
            "请先运行 generate_rb_regressor_complete.py 生成Python函数，\n"
            "或安装 oct2py 使用MATLAB函数。"
        )
    
    # 准备数据
    q_matrix = idntfcn_traj['q']
    qd_fltrd_matrix = idntfcn_traj['qd_fltrd']
    q2d_est_matrix = idntfcn_traj['q2d_est']
    Tau_matrix = idntfcn_traj['i_fltrd']
    n_samples = q_matrix.shape[0]
    
    print(f"    处理 {n_samples} 个样本...")
    
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
    
    # 计算摩擦回归矩阵
    print("    计算摩擦回归矩阵...")
    Y_frctn_total = friction_regressor_batched(qd_fltrd_matrix)
    
    # 构建基础观测矩阵
    n_base = baseQR['numberOfBaseParameters']
    E_full = baseQR['permutationMatrixFull']
    E1 = E_full[:, :n_base]
    
    W_dyn = Y_std_total @ E1
    Wb = np.hstack([W_dyn, Y_frctn_total])
    
    # 力矩向量
    Tau = Tau_matrix.flatten()
    
    print(f"  ✓ 观测矩阵构建完成: {Wb.shape}")
    
    return Tau, Wb



def build_observation_matrices(idntfcn_traj, baseQR, drv_gains):
    """
    Returns:
        Tau: 力矩向量 (6N,)
        Wb: 观测矩阵 (6N, n_base + 12)
    """
    if HAS_OCT2PY:
        try:
            return build_observation_matrices_oct2py(idntfcn_traj, baseQR, drv_gains)
        except Exception as e:
            return build_observation_matrices_python(idntfcn_traj, baseQR, drv_gains)
    else:
        return build_observation_matrices_python(idntfcn_traj, baseQR, drv_gains)

def ordinary_least_square_estimation(Tau, Wb, baseQR):
    print("  使用普通最小二乘法...")
    
    pi_OLS = np.linalg.lstsq(Wb, Tau, rcond=None)[0]
    
    n_b = baseQR['numberOfBaseParameters']
    pi_b_OLS = pi_OLS[:n_b]
    pi_frctn_OLS = pi_OLS[n_b:]
    
    print(f"  ✓ 估计了 {n_b} 个基础参数和 {len(pi_frctn_OLS)} 个摩擦参数")
    return pi_b_OLS, pi_frctn_OLS


def physically_consistent_estimation(Tau, Wb, baseQR, pi_urdf=None, lambda_reg=0, 
                                    physical_consistency=0):
    """
    物理一致性参数估计
    
    Args:
        Tau: 力矩向量
        Wb: 观测矩阵
        baseQR: QR分解结果
        pi_urdf: URDF参考参数（前5个link，50维）
        lambda_reg: 正则化系数
        physical_consistency: 0=半物理一致性(MATLAB默认), 1=完全物理一致性
    """
    if physical_consistency == 0:
        print("  使用半物理一致性约束优化（与MATLAB一致）...")
    else:
        print("  使用完全物理一致性约束优化...")
    
    # 参数设置
    n_b = baseQR['numberOfBaseParameters']
    n_d = 60 - n_b
    
    print(f"    基础参数: {n_b}, 依赖参数: {n_d}")
    
    # 定义优化变量
    pi_frctn = cp.Variable(12)  # 6个关节 × 2个摩擦参数 = 12
    pi_b = cp.Variable(n_b)
    pi_d = cp.Variable(n_d)
    
    # 映射到标准参数
    mapping_matrix = np.block([
        [np.eye(n_b), -baseQR['beta']],
        [np.zeros((n_d, n_b)), np.eye(n_d)]
    ])
    E = baseQR['permutationMatrixFull']
    pii = E @ mapping_matrix @ cp.hstack([pi_b, pi_d])
    
    # 约束条件
    constraints = []
    
    # 1. 质量约束（只针对前5个link）
    mass_indices = list(range(9, 50, 10))  # link1-link5: [9, 19, 29, 39, 49]
    mass_urdf = np.array([0.607, 0.918, 0.7, 0.359, 0.403])  # 前5个link的质量
    error_range = 0.15  # 放宽到15%以提高求解成功率
    mass_upper = mass_urdf * (1 + error_range)
    mass_lower = mass_urdf * (1 - error_range)
    
    print(f"    添加质量约束（前5个link，±{error_range*100}%）...")
    for i, idx in enumerate(mass_indices):
        constraints.append(pii[idx] >= max(0, mass_lower[i]))
        constraints.append(pii[idx] <= mass_upper[i])
    
    # 2. 物理一致性约束
    print(f"    添加物理一致性约束（类型: {physical_consistency}）...")
    for link_idx in range(6):
        i = link_idx * 10
        
        # 惯性张量 (3x3)
        I_link = cp.vstack([
            cp.hstack([pii[i],     pii[i+1], pii[i+2]]),
            cp.hstack([pii[i+1],   pii[i+3], pii[i+4]]),
            cp.hstack([pii[i+2],   pii[i+4], pii[i+5]])
        ])
        
        # 一阶矩 h = m*r_com
        h_link = pii[i+6:i+9]
        
        # 质量
        m_link = pii[i+9]
        
        if physical_consistency == 1:
            # 完全物理一致性：D = [0.5*tr(I)*I_3 - I,  h; h^T,  m]
            h_link_col = cp.reshape(h_link, (3, 1), order='C')
            h_link_row = cp.reshape(h_link, (1, 3), order='C')
            m_link_reshaped = cp.reshape(m_link, (1, 1), order='C')
            
            trace_I = cp.trace(I_link)
            upper_left = 0.5 * trace_I * np.eye(3) - I_link
            upper_right = h_link_col
            upper_part = cp.hstack([upper_left, upper_right])
            
            lower_left = h_link_row
            lower_right = m_link_reshaped
            lower_part = cp.hstack([lower_left, lower_right])
            
            D_link = cp.vstack([upper_part, lower_part])
        else:
            # 半物理一致性（与MATLAB一致）：D = [I, h^T; h, m*I_3]
            # 这里 h 是斜对称矩阵形式
            def vec2skew(v):
                """将3D向量转换为斜对称矩阵"""
                return cp.vstack([
                    cp.hstack([0, -v[2], v[1]]),
                    cp.hstack([v[2], 0, -v[0]]),
                    cp.hstack([-v[1], v[0], 0])
                ])
            
            h_link_skew = vec2skew(h_link)
            h_link_skew_T = cp.reshape(cp.vec(h_link_skew.T), (3, 3), order='C')
            
            # D = [I, h_skew^T; h_skew, m*I_3]
            upper_part = cp.hstack([I_link, h_link_skew_T])  # (3, 6)
            lower_part = cp.hstack([h_link_skew, m_link * np.eye(3)])  # (3, 6)
            D_link = cp.vstack([upper_part, lower_part])  # (6, 6)
        
        # 半正定约束
        constraints.append(D_link >> 0)
        
        # 三角不等式约束（MuJoCo要求：A+B>=C）
        # 对于任何物理刚体惯性张量，必须满足：
        # Ixx+Iyy≥Izz, Ixx+Izz≥Iyy, Iyy+Izz≥Ixx
        # 使用较大的安全余量，确保MuJoCo编译器接受
        epsilon_triangle = 1e-4  # 增大到1e-4（之前5e-5还不够）
        Ixx = pii[i]
        Iyy = pii[i+3]
        Izz = pii[i+5]
        
        # 添加三角不等式约束（每个都加上安全余量）
        constraints.append(Ixx + Iyy >= Izz + epsilon_triangle)
        constraints.append(Ixx + Izz >= Iyy + epsilon_triangle)
        constraints.append(Iyy + Izz >= Ixx + epsilon_triangle)
        
        # 额外约束：确保对角元素都是正的且有下界
        epsilon_inertia = 1e-6
        constraints.append(Ixx >= epsilon_inertia)
        constraints.append(Iyy >= epsilon_inertia)
        constraints.append(Izz >= epsilon_inertia)
        
        # ✅ COM位置约束：确保转换后的I_COM也正定
        # h = m * r_com，限制COM偏移在合理范围内
        h_max_component = 0.10  # kg·m per axis（更严格：~0.1-0.2m对于0.5-1kg的link）
        constraints.append(h_link[0] >= -h_max_component)
        constraints.append(h_link[0] <= h_max_component)
        constraints.append(h_link[1] >= -h_max_component)
        constraints.append(h_link[1] <= h_max_component)
        constraints.append(h_link[2] >= -h_max_component)
        constraints.append(h_link[2] <= h_max_component)
    
    
    # 3. 摩擦参数约束（所有参数都必须 > 0）
    print("    粘性和库伦摩擦 > 0")
    epsilon_friction = 1e-10  # 小的正值下界，确保严格大于零
    for i in range(6):
        constraints.append(pi_frctn[2*i] >= epsilon_friction)      # 粘性摩擦 > 0
        constraints.append(pi_frctn[2*i + 1] >= epsilon_friction)  # 库伦摩擦 > 0
    
    # 目标函数
    tau_error = cp.norm(Tau - Wb @ cp.hstack([pi_b, pi_frctn]))
    
    use_regularization = (pi_urdf is not None) and (lambda_reg > 0)
    
    if use_regularization:
        # 只对前5个link（前50个参数）进行正则化
        # pii[0:50] 对应 link1-link5 的参数
        if len(pi_urdf) != 50:
            print(f"  警告: pi_urdf长度为{len(pi_urdf)}，期望50。将截断或填充。")
            pi_urdf_padded = np.zeros(50)
            pi_urdf_padded[:min(50, len(pi_urdf))] = pi_urdf[:min(50, len(pi_urdf))]
            pi_urdf = pi_urdf_padded
        
        param_regularization = lambda_reg * cp.norm(pii[:50] - pi_urdf)
        objective = tau_error + param_regularization
        print(f"    目标 = tau误差 + {lambda_reg:.1e} * 参数正则化（仅前5个link）")
    else:
        objective = tau_error
        print("    目标 = tau误差 (无正则化)")
    
    # 求解SDP
    print("  求解SDP优化问题...")
    
    problem = cp.Problem(cp.Minimize(objective), constraints)
    
    try:
        # 尝试使用MOSEK求解器
        try:
            result = problem.solve(
                solver=cp.MOSEK,
                verbose=False
            )
            print(f"  ✓ 使用MOSEK求解器 (状态: {problem.status})")
        except:
            # 回退到SCS求解器，但提高精度
            print("  MOSEK不可用，使用SCS求解器（超高精度模式）...")
            result = problem.solve(
                solver=cp.SCS,
                verbose=False,
                max_iters=100000,     # 大幅增加迭代次数，确保约束严格满足
                eps=1e-9,             # 超高精度，确保三角不等式满足
                alpha=1.8,            # 改善收敛
                scale=10.0,           # 改善数值稳定性
                normalize=True,       # 归一化以提高数值稳定性
                acceleration_lookback=20  # 加速收敛
            )
        
        if problem.status not in ['optimal', 'optimal_inaccurate']:
            print(f" 求解状态: {problem.status}")
            print("  尝试CVXOPT求解器...")
            result = problem.solve(solver=cp.CVXOPT, verbose=False)
        
        if problem.status in ['optimal', 'optimal_inaccurate']:
            print(f"  ✓ SDP求解成功! (状态: {problem.status})")
            if problem.status == 'optimal_inaccurate':
                print("   注意: 求解精度可能不够，结果可能轻微违反约束")
        else:
            print(f"    求解状态: {problem.status}")
            print("  结果可能不准确")
    
    except Exception as e:
        print(f"  SDP求解失败: {e}")
        raise
    
    # 提取结果
    pi_b_SDP = pi_b.value
    pi_frctn_SDP = pi_frctn.value
    pi_d_value = pi_d.value
    
    if pi_b_SDP is None or pi_frctn_SDP is None:
        raise RuntimeError("SDP求解失败，无法提取参数值")
    
    # 计算完整标准参数
    pi_full = E @ mapping_matrix @ np.concatenate([pi_b_SDP, pi_d_value])
    mass_estimated = pi_full[mass_indices]
    
    # 验证结果
    
    tau_pred = Wb @ np.concatenate([pi_b_SDP, pi_frctn_SDP])
    tau_pred_error = np.linalg.norm(Tau - tau_pred)
    rel_tau_error = 100 * tau_pred_error / np.linalg.norm(Tau)
    print(f"  Tau预测误差: {tau_pred_error:.4e} ({rel_tau_error:.2f}%)")
    
    if use_regularization:
        param_deviation = np.linalg.norm(pi_full[:50] - pi_urdf)
        print(f"  前5个link参数偏离URDF: {param_deviation:.4e}")
    
    # 质量对比（前5个link）
    print("  连杆 | URDF参考 | 估计值 | 相对误差")
    for i in range(5):  # 只显示前5个link
        rel_error = 100 * (mass_estimated[i] - mass_urdf[i]) / mass_urdf[i]
        print(f"  Link{i+1} | {mass_urdf[i]:7.4f}kg | {mass_estimated[i]:7.4f}kg | {rel_error:+7.2f}%")
    
    # 检查惯性矩阵（正定性 + 三角不等式）
    print("\n  检查惯性矩阵物理约束:")
    triangle_violations = []
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
        
        # 正定性检查
        positive_definite = np.all(eig_vals > -1e-8)
        
        # 三角不等式检查（MuJoCo要求）
        margin1 = (Ixx + Iyy) - Izz
        margin2 = (Ixx + Izz) - Iyy
        margin3 = (Iyy + Izz) - Ixx
        min_margin = min(margin1, margin2, margin3)
        triangle_satisfied = min_margin >= -1e-8  # 允许极小的数值误差
        
        # 综合状态
        if positive_definite and triangle_satisfied:
            status = "✓"
        else:
            status = "✗"
            if not triangle_satisfied:
                triangle_violations.append((link_idx + 1, min_margin))
        
        regularized = "【正则化】" if link_idx < 5 else "【无约束】"
        print(f"    Link{link_idx+1} {regularized}: {status} (λ_min={np.min(eig_vals):.4e}, 三角余量={min_margin:.4e})")
    
    if triangle_violations:
        print("\n  以下link违反三角不等式约束:")
        for link_num, margin in triangle_violations:
            print(f"      Link{link_num}: 最小余量 = {margin:.4e}")
    
    return pi_b_SDP, pi_frctn_SDP, pi_full, mass_estimated


# 数据处理函数

def parse_ur_data(path_to_data, idx_start, idx_end):
    import pandas as pd
    
    if not os.path.exists(path_to_data):
        raise FileNotFoundError(f"数据文件不存在: {path_to_data}")
    
    print(f"  从 {path_to_data} 加载数据...")
    df = pd.read_csv(path_to_data)
    
    # 检查数据范围
    if idx_end >= len(df):
        print(f"  ⚠️  索引超出范围，调整为: [{ idx_start}, {len(df)-1}]")
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
    
    print(f"  过滤数据 (基于 MATLAB 逻辑，采样率 fs={fs} Hz)...")
    
    nyq = 0.5 * fs # 奈奎斯特频率
    
    # --- 1. 速度滤波设计和应用 ---
    
    # MATLAB: FilterOrder=5, HalfPowerFrequency=0.15
    normal_cutoff_vel = 0.15 
    N_order = 5 # 5阶滤波器
    
    # 计算滤波器系数
    b_vel, a_vel = butter(N_order, normal_cutoff_vel, btype='low', analog=False)
    
    # 过滤速度
    qd_fltrd = np.zeros_like(data_dict['qd'])
    for i in range(6):
        qd_fltrd[:, i] = filtfilt(b_vel, a_vel, data_dict['qd'][:, i])
    data_dict['qd_fltrd'] = qd_fltrd
    
    # --- 2. 估计加速度：三点中心差分 ---
    
    dt = np.mean(np.diff(data_dict['t']))
    if dt <= 0:
        dt = 1.0 / fs
        print(f"  ⚠️  无效时间步长，使用默认值: {dt}s")
    
    q2d_est = np.zeros_like(data_dict['qd_fltrd'])
    N_samples = data_dict['qd_fltrd'].shape[0]
    
    # 显式实现 MATLAB 中的三点中心差分 (i=2 到 N-1，边界点保持为零)
    for i in range(1, N_samples - 1): # Python 索引 1 到 N-2 对应 MATLAB 索引 2 到 N-1
       dlta_qd_fltrd =  data_dict['qd_fltrd'][i+1,:] - data_dict['qd_fltrd'][i-1,:]
       dlta_t_msrd = data_dict['t'][i+1] - data_dict['t'][i-1]
       q2d_est[i,:] = dlta_qd_fltrd / dlta_t_msrd

    # 替换 np.gradient 的结果
    data_dict['q2d_est'] = q2d_est
    
    # --- 3. 加速度滤波设计和应用 ---
    
    # MATLAB: 加速度滤波与速度滤波参数相同
    b_accel = b_vel
    a_accel = a_vel
    
    # 过滤加速度
    for i in range(6):
        data_dict['q2d_est'][:, i] = filtfilt(b_accel, a_accel, data_dict['q2d_est'][:, i])
    
    # --- 4. 电流/力矩滤波设计和应用 ---
    
    # MATLAB: FilterOrder=5, HalfPowerFrequency=0.20
    normal_cutoff_curr = 0.20 
    b_curr, a_curr = butter(N_order, normal_cutoff_curr, btype='low', analog=False)
    
    # 过滤电流/力矩
    i_fltrd = np.zeros_like(data_dict['i'])
    for i in range(6):
        i_fltrd[:, i] = filtfilt(b_curr, a_curr, data_dict['i'][:, i])
    data_dict['i_fltrd'] = i_fltrd
    
    print("  ✓ 数据过滤完成")
    return data_dict


def load_urdf_parameters(urdf_path):
    """
    从URDF加载前5个link的参考参数
    Returns:
        pi_urdf: (50,) 数组，包含link1-link5的参数（每个link 10个参数）
    """
    if not os.path.exists(urdf_path):
        print(f"  ⚠️  URDF文件不存在: {urdf_path}")
        return np.zeros(50)  # 5个link × 10个参数/link = 50
    
    try:
        sys.path.insert(0, 'utils')
        from utils.parse_urdf import parse_urdf
        robot = parse_urdf(urdf_path)
        # parse_urdf 返回 (10, 5) 的数组（5个link）
        # 需要展平为 (50,) 按列优先（Fortran顺序）以匹配MATLAB格式
        pi_urdf_5links = robot['pi'].flatten('F')  # 'F' = Fortran/列优先顺序
        print(f"  ✓ 从URDF加载了前5个link的 {len(pi_urdf_5links)} 个参考参数")
        return pi_urdf_5links
    except Exception as e:
        print(f"  ⚠️  无法从URDF加载参数: {e}")
        return np.zeros(50)


def print_estimation_summary(sol, Tau, tau_pred):
    
    
    print(f"  基础参数数量: {len(sol['pi_b'])}")
    print(f"  摩擦参数数量: {len(sol['pi_fr'])}")
    
    residual = Tau - tau_pred
    rmse = np.sqrt(np.mean(residual**2))
    r_squared = 1 - np.sum(residual**2) / np.sum((Tau - np.mean(Tau))**2)
    
    print(f"\n  拟合质量:")
    print(f"    RMSE: {rmse:.4e}")
    print(f"    R²: {r_squared:.6f}")
    
    # print(f"\n  参数统计:")
    # print(f"    平均相对标准差: {np.mean(sol['rel_std']):.2f}%")
    # print(f"    最大相对标准差: {np.max(sol['rel_std']):.2f}%")
    
    if sol['masses'] is not None:
        print(f"\n  估计的总质量: {np.sum(sol['masses']):.4f} kg")
    


# 主函数

def estimate_dynamic_params(path_to_data, idx, drv_gains, baseQR, method='OLS',
                           lambda_reg=1e-3, urdf_path=None):
    """
    
    Args:
        path_to_data: 数据文件路径
        idx: 数据索引范围 [start, end]
        drv_gains: 驱动增益
        baseQR: QR分解结果
        method: 'OLS', 'PC-OLS', 或 'PC-OLS-REG'
        lambda_reg: 正则化系数
        urdf_path: URDF文件路径
    
    Returns:
        sol: 估计结果字典
    """
    
    print(f"动力学参数估计 - 方法: {method}")
    
    # 1. 加载和处理数据
    print("\n步骤 1/4: 加载和处理数据...")
    idntfcn_traj = parse_ur_data(path_to_data, idx[0], idx[1])
    idntfcn_traj = filter_data(idntfcn_traj)
    
    # 2. 构建观测矩阵
    print("\n步骤 2/4: 构建观测矩阵...")
    Tau, Wb = build_observation_matrices(idntfcn_traj, baseQR, drv_gains)
    print(f"  观测矩阵 Wb 形状: {Wb.shape}")
    print(f"  力矩向量 Tau 形状: {Tau.shape}")
    n_base = baseQR['numberOfBaseParameters']
    n_params = n_base + 18
    
    # 3. 估计参数
    print(f"\n步骤 3/4: 参数估计 ({method})...")
    
    sol = {}
    
    if method == 'OLS':
        pi_b, pi_fr = ordinary_least_square_estimation(Tau, Wb, baseQR)
        sol['pi_b'] = pi_b
        sol['pi_fr'] = pi_fr
        sol['pi_s'] = None
        sol['masses'] = None
        
    elif method == 'PC-OLS':
        pi_b, pi_fr, pi_s, masses = physically_consistent_estimation(
            Tau, Wb, baseQR, pi_urdf=None, lambda_reg=0, physical_consistency=0
        )
        sol['pi_b'] = pi_b
        sol['pi_fr'] = pi_fr
        sol['pi_s'] = pi_s
        sol['masses'] = masses
        
    elif method == 'PC-OLS-REG':
        if urdf_path is None:
            raise ValueError("PC-OLS-REG方法需要提供urdf_path参数")
        
        print(f"  加载URDF参考参数: {urdf_path}")
        pi_urdf = load_urdf_parameters(urdf_path)
        
        pi_b, pi_fr, pi_s, masses = physically_consistent_estimation(
            Tau, Wb, baseQR, pi_urdf, lambda_reg, physical_consistency=0
        )
        sol['pi_b'] = pi_b
        sol['pi_fr'] = pi_fr
        sol['pi_s'] = pi_s
        sol['masses'] = masses
        
    else:
        raise ValueError(f" 可选: 'OLS', 'PC-OLS', 'PC-OLS-REG'")
    
    # 4. 统计分析
    print("\n步骤 4/4: 统计分析...")
    
    tau_pred = Wb @ np.concatenate([sol['pi_b'], sol['pi_fr']])
    residual = Tau - tau_pred
    
    n_samples = Wb.shape[0]
    n_params = Wb.shape[1]
    sqrd_sigma_e = np.linalg.norm(residual)**2 / (n_samples - n_params)
    
    try:
        Wb_inv = inv(Wb.T @ Wb)
        Cpi = sqrd_sigma_e * Wb_inv
        sol['std'] = np.sqrt(np.diag(Cpi))
    except np.linalg.LinAlgError:
        sol['std'] = np.ones(len(sol['pi_b']) + len(sol['pi_fr']))
    
    params_all = np.concatenate([sol['pi_b'], sol['pi_fr']])
    # 避免除以零
    params_all_safe = np.where(np.abs(params_all) < 1e-10, 1e-10, params_all)
    sol['rel_std'] = 100 * sol['std'] / np.abs(params_all_safe)
    
    print_estimation_summary(sol, Tau, tau_pred)
    
    
    return sol


def main():
    mat_filename_standard = 'models/baseQR_standard.mat'
    
    print(f"加载base_params_qr.mat: {mat_filename_standard}")
    
    try:
        if not os.path.exists(mat_filename_standard):
            raise FileNotFoundError(f"文件不存在: {mat_filename_standard}")
            
        # 1. 使用 h5py.File 读取 HDF5 文件
        with h5py.File(mat_filename_standard, 'r') as f:
            print("    ✓ 正在使用 h5py 读取 HDF5 格式...")
            
            # 访问 baseQR 结构体对应的 HDF5 Group
            baseQR_group = f['baseQR']
            
            # 2. 从 HDF5 Group 中提取数据，并处理维度和转置 (MATLAB 列优先 vs Python 行优先)
            
            # numberOfBaseParameters: 标量
            bb = np.array(baseQR_group['numberOfBaseParameters']).flatten()[0]
            
            # permutationMatrix: 需要转置 (.T)
            E_full = np.array(baseQR_group['permutationMatrix']).T
            
            # beta: 需要转置 (.T)
            beta = np.array(baseQR_group['beta']).T
            
            # motorDynamicsIncluded: 标量
            motorDynamicsIncluded = bool(np.array(baseQR_group['motorDynamicsIncluded']).flatten()[0])

            # 3. 封装到 Python 字典
            baseQR = {
                'numberOfBaseParameters': int(bb),
                'permutationMatrixFull': E_full,
                'beta': beta,
                'motorDynamicsIncluded': motorDynamicsIncluded
            }

        # --- 原始的维度检查仍然非常重要 ---
        if baseQR['beta'].ndim == 1:
            # 如果 beta 是 (N,) 向量，确保它是 (N, 1) 的列向量
            baseQR['beta'] = baseQR['beta'].reshape(-1, 1)

        print(f"成功从 {mat_filename_standard} 加载基础参数。")
        print(f"   基础参数数量 N_b: {baseQR['numberOfBaseParameters']}")
        
    except FileNotFoundError:
        print(f"找不到文件 {mat_filename_standard}")
        print("请检查路径是否正确，并确保已运行 MATLAB 脚本 base_params_qr.m 生成文件。")
        sys.exit(1)
    except Exception as e:
        # 如果是权限问题、h5py导入问题或解析错误，都可以在这里捕获
        print(f"加载或解析 MATLAB HDF5 文件失败: {e}")
        # 提示用户可能需要安装 h5py 或检查文件格式
        if 'h5py' not in sys.modules:
            print("提示: 您的错误可能是缺少 'h5py' 库。请运行 'pip install h5py'。")
        print("请确保 MATLAB 文件已使用 '-v7.3' 选项保存。")
        sys.exit(1)
    
    drv_gains = np.ones(6)
    idx = [0, 2500]
    
    data_path = 'results/data_csv/vali.csv'
    urdf_path = 'models/mjcf/manipulator/airbot_play_force/_play_force.urdf'

    # 检查数据文件是否存在
    if not os.path.exists(data_path):
        print(f"❌ 错误: 数据文件不存在: {data_path}")
        sys.exit(1)

    # 方法1: OLS
    print(" OLS")
    sol_ols = estimate_dynamic_params(
        path_to_data=data_path,
        idx=idx,
        drv_gains=drv_gains,
        baseQR=baseQR,
        method='OLS'
    )
    
    # 方法2: PC-OLS (物理一致性，无正则化)
    print(" PC-OLS")
    sol_pc_ols = estimate_dynamic_params(
        path_to_data=data_path,
        idx=idx,
        drv_gains=drv_gains,
        baseQR=baseQR,
        method='PC-OLS'
    )
    
    # 方法3: PC-OLS-REG (物理一致性 + 正则化，推荐)
    if os.path.exists(urdf_path):
        print(" PC-OLS-REG ")
        sol_pc_reg = estimate_dynamic_params(
            path_to_data=data_path,
            idx=idx,
            drv_gains=drv_gains,
            baseQR=baseQR,
            method='PC-OLS-REG',
            lambda_reg=5e-4,  # 降低正则化强度，避免违反物理约束
            urdf_path=urdf_path
        )
    else:
        sol_pc_reg = None
    
    # 保存所有结果
    os.makedirs('results', exist_ok=True)
    result_path = 'results/estimation_results.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump({
            'sol_ols': sol_ols,
            'sol_pc_ols': sol_pc_ols,
            'sol_pc_reg': sol_pc_reg
        }, f)
    
    print(f"所有结果已保存到 {result_path}")


if __name__ == "__main__":
    # 确保pandas已安装，因为parse_ur_data依赖它
    try:
        import pandas as pd
    except ImportError:
        print(" 错误: 需要安装 Pandas 库: pip install pandas")
        sys.exit(1)
        
    main()