#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import mujoco
import os
from scipy import interpolate

# 配置参数
MODEL_XML_PATH = "models/mjcf/manipulator/airbot_play_force/_play_force.xml"
MAT_FILE_PATH = "models/ptrnSrch_N8T25QR_maxq1.mat"
OUTPUT_CSV_PATH = "mujoco_theoretical_vali.csv"  # 输出理论扭矩数据
SAMPLE_RATE = 100  # Hz
SIM_TIME = 40  # 仿真时间（秒）


def mixed_trajectory_calculator(t_vec, T, N, wf, a, b, c_pol, q0):
    """
    计算混合轨迹（傅里叶 + 多项式）
    返回: qd, qdot_d, qddot_d
    """
    t_vec = np.atleast_1d(t_vec)
    
    J = a.shape[0]  # 关节数 (6)
    M = len(t_vec)  # 时间点数
    
    qd = np.zeros((J, M))
    qdot_d = np.zeros((J, M))
    qddot_d = np.zeros((J, M))
    
    tau_vec = t_vec % T  # 周期时间（对时间取模，支持循环）
    
    for i in range(J):  # 遍历每个关节 i
        k_vec = np.arange(1, N + 1).reshape(-1, 1)  # k = 1, 2, ..., N
        wk_t = wf * k_vec * t_vec  # (N x M)
        sin_wk_t = np.sin(wk_t)
        cos_wk_t = np.cos(wk_t)
        
        # --- (A) 傅里叶部分计算 ---
        a_norm = a[i, :].reshape(-1, 1) / (wf * k_vec) 
        b_norm = b[i, :].reshape(-1, 1) / (wf * k_vec) 
        
        # 1. 位置 qd_fourier
        qd_fourier = (a_norm * sin_wk_t - b_norm * cos_wk_t).sum(axis=0)
        
        # 2. 速度 qdot_d_fourier
        qdot_d_fourier = (a[i, :].reshape(-1, 1) * cos_wk_t + 
                         b[i, :].reshape(-1, 1) * sin_wk_t).sum(axis=0)
        
        # 3. 加速度 qddot_d_fourier
        ak_wf_k = a[i, :].reshape(-1, 1) * wf * k_vec
        bk_wf_k = b[i, :].reshape(-1, 1) * wf * k_vec
        qddot_d_fourier = (-ak_wf_k * sin_wk_t + bk_wf_k * cos_wk_t).sum(axis=0)
        
        # --- (B) 多项式部分计算 ---
        qd_poly = np.zeros(M)
        qdot_d_poly = np.zeros(M)
        qddot_d_poly = np.zeros(M)
        
        for k_idx in range(6):  # k=0 到 k=5
            k_exp = k_idx 
            c = c_pol[i, k_idx]
            
            # 位置
            qd_poly += c * (tau_vec ** k_exp)
            
            # 速度
            if k_exp >= 1:
                qdot_d_poly += c * k_exp * (tau_vec ** (k_exp - 1))
            
            # 加速度
            if k_exp >= 2:
                qddot_d_poly += c * k_exp * (k_exp - 1) * (tau_vec ** (k_exp - 2))

        # --- (C) 叠加 ---
        qd[i, :] = qd_fourier + qd_poly
        qdot_d[i, :] = qdot_d_fourier + qdot_d_poly
        qddot_d[i, :] = qddot_d_fourier + qddot_d_poly
    
    return qd, qdot_d, qddot_d


def compute_theoretical_torques(model, data, time_points, T, N, wf, a, b, c_pol, q0, n_joints):
    """
    使用MuJoCo逆动力学计算理论扭矩
    """
    n_samples = len(time_points)
    
    # 存储结果
    q_array = np.zeros((n_samples, n_joints))
    qdot_array = np.zeros((n_samples, n_joints))
    qddot_array = np.zeros((n_samples, n_joints))
    tau_array = np.zeros((n_samples, n_joints))
    
    print(f"\n开始计算理论扭矩...")
    print(f"  - 总样本数: {n_samples}")
    print(f"  - 时间范围: {time_points[0]:.3f}s - {time_points[-1]:.3f}s")
    
    for idx, t in enumerate(time_points):
        # 计算期望轨迹
        qd, qdot_d, qddot_d = mixed_trajectory_calculator(t, T, N, wf, a, b, c_pol, q0)
        
        # 保存轨迹数据
        q_array[idx, :] = qd[:, 0]
        qdot_array[idx, :] = qdot_d[:, 0]
        qddot_array[idx, :] = qddot_d[:, 0]
        
        # 设置期望状态到MuJoCo数据结构
        data.qpos[:n_joints] = qd[:, 0]
        data.qvel[:n_joints] = qdot_d[:, 0]
        data.qacc[:n_joints] = qddot_d[:, 0]
        
        # 调用MuJoCo逆动力学
        mujoco.mj_inverse(model, data)
        
        # 提取计算的扭矩
        tau_array[idx, :] = data.qfrc_inverse[:n_joints].copy()
        
        # 进度显示
        if (idx + 1) % 500 == 0 or idx == 0 or idx == n_samples - 1:
            print(f"  进度: {idx + 1}/{n_samples} ({100*(idx+1)/n_samples:.1f}%)")
    
    print("✓ 理论扭矩计算完成")
    
    return q_array, qdot_array, qddot_array, tau_array


def main():
    
    
    if not os.path.exists(MODEL_XML_PATH):
        print(f"错误：找不到模型文件 {MODEL_XML_PATH}")
        return
    
    try:
        model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
        data = mujoco.MjData(model)
        print(f"  - 关节数量: {model.njnt}")
        print(f"  - 执行器数量: {model.nu}")
        print(f"  - 时间步长: {model.opt.timestep} 秒")
    except Exception as e:
        print(f"错误：加载模型失败 - {e}")
        return
    
    
    if not os.path.exists(MAT_FILE_PATH):
        print(f"错误：找不到 .mat 文件 {MAT_FILE_PATH}")
        return
    
    try:
        mat_contents = sio.loadmat(MAT_FILE_PATH)
        a = mat_contents['a']
        b = mat_contents['b']
        c_pol = mat_contents['c_pol']
        traj_par_struct = mat_contents['traj_par'][0, 0]
        
        T = traj_par_struct['T'][0, 0]
        N = int(traj_par_struct['N'][0, 0])
        wf = traj_par_struct['wf'][0, 0]
        q0 = traj_par_struct['q0']
        
        
    except Exception as e:
        print(f"错误：加载轨迹参数失败 - {e}")
        return
    
    # ========== 检查关节数匹配 ==========
    n_joints = a.shape[0]
    if model.nu != n_joints:
        print(f"\n警告：执行器数量 ({model.nu}) 与轨迹关节数 ({n_joints}) 不匹配")
        n_joints = min(model.nu, n_joints)
        print(f"将使用前 {n_joints} 个关节")
    
    print("\n正在初始化模型...")
    mujoco.mj_resetData(model, data)
    
    q_initial, _, _ = mixed_trajectory_calculator(0.0, T, N, wf, a, b, c_pol, q0)
    data.qpos[:n_joints] = q_initial[:, 0]
    
    mujoco.mj_forward(model, data)
    
    sample_interval = 1.0 / SAMPLE_RATE
    time_points = np.arange(0, SIM_TIME, sample_interval)

    
    q_array, qdot_array, qddot_array, tau_array = compute_theoretical_torques(
        model, data, time_points, T, N, wf, a, b, c_pol, q0, n_joints
    )
    
    print(f"\n正在保存数据到 CSV...")
    
    output_data = np.column_stack([
        time_points,           # Col 1: time
        q_array,               # Col 2-7: q (期望位置)
        qdot_array,            # Col 8-13: qd (期望速度)
        tau_array,             # Col 14-19: tau (理论扭矩)
        tau_array,             # Col 20-25: tau (理论扭矩，重复)
        tau_array              # Col 26-31: tau (理论扭矩，重复)
    ])
    
    np.savetxt(OUTPUT_CSV_PATH, output_data, delimiter=',', fmt='%.6f')
    


if __name__ == "__main__":
    main()