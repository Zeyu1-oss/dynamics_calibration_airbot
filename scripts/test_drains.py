#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import mujoco
import mujoco.viewer
import os
import matplotlib.pyplot as plt
from scipy import interpolate

# ================= 配置區 =================
COMPARE_CSV_PATH = "results/data/vali_ptrnSrch_N7T25QR-6_converted.csv"
MODEL_XML_PATH = "models/mjcf/manipulator/airbot_play_force/_play_force.xml" 
MAT_FILE_PATH = "models/ptrnSrch_N7T25QR-6.mat"
SIM_TIME = 25
CONTROL_HZ = 1000  
CONTROL_DT = 1.0 / CONTROL_HZ

# 比例系數 K (你可以手動調整這個值，或者看腳本最後生成的建議值)
# 最終力矩 = MuJoCo力矩 * K
K_SCALES = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0] 

# ================= 核心邏輯 =================

def get_real_data(csv_path):
    """讀取真實機器人數據"""
    data = np.loadtxt(csv_path, delimiter=',')
    t = data[:, 0] - data[0, 0]
    # 假設列 13-18 是真實力矩 (tau1-6)
    tau_real = data[:, 13:19]
    return t, tau_real

def mixed_trajectory_calculator(t_vec, T, N, wf, a, b, c_pol, q0):
    t_vec = np.atleast_1d(t_vec)
    J, M = a.shape[0], len(t_vec)
    qd, qdot_d, qddot_d = np.zeros((J, M)), np.zeros((J, M)), np.zeros((J, M))
    tau_vec = t_vec % T  
    for i in range(J):
        k_vec = np.arange(1, N + 1).reshape(-1, 1)
        wk_t = wf * k_vec * t_vec
        sin_wk_t, cos_wk_t = np.sin(wk_t), np.cos(wk_t)
        a_norm, b_norm = a[i, :].reshape(-1, 1) / (wf * k_vec), b[i, :].reshape(-1, 1) / (wf * k_vec)
        qd_fourier = (a_norm * sin_wk_t - b_norm * cos_wk_t).sum(axis=0)
        qdot_d_fourier = (a[i, :].reshape(-1, 1) * cos_wk_t + b[i, :].reshape(-1, 1) * sin_wk_t).sum(axis=0)
        qddot_d_fourier = ((-a[i, :].reshape(-1, 1) * wf * k_vec) * sin_wk_t + (b[i, :].reshape(-1, 1) * wf * k_vec) * cos_wk_t).sum(axis=0)
        qd_poly = sum(c_pol[i, k] * (tau_vec**k) for k in range(6))
        qdot_poly = sum(c_pol[i, k] * k * (tau_vec**(k-1)) for k in range(1, 6))
        qddot_poly = sum(c_pol[i, k] * k * (k-1) * (tau_vec**(k-2)) for k in range(2, 6))
        qd[i, :] = qd_fourier + qd_poly
        qdot_d[i, :] = qdot_d_fourier + qdot_poly
        qddot_d[i, :] = qddot_d_fourier + qddot_poly
    return qd, qdot_d, qddot_d

def main():
    # 1. 加載模型與軌跡
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    mat_contents = sio.loadmat(MAT_FILE_PATH)
    a, b, c_pol = mat_contents['a'], mat_contents['b'], mat_contents['c_pol']
    tp = mat_contents['traj_par'][0, 0]
    T, N, wf, q0 = tp['T'][0,0], int(tp['N'][0,0]), tp['wf'][0,0], tp['q0']

    # 2. 獲取真實數據用於對比
    t_real, tau_real_all = get_real_data(COMPARE_CSV_PATH)

    recorded_sim = {'time': [], 'tau_sim': []}
    inv_data = mujoco.MjData(model)

    print("🚀 正在運行 MuJoCo 仿真...")
    while data.time < SIM_TIME:
        t = data.time
        # 計算期望狀態
        qd, qv, qa = mixed_trajectory_calculator(t, T, N, wf, a, b, c_pol, q0)
        
        # 逆動力學計算理論力矩
        inv_data.qpos[:6] = qd[:, 0]
        inv_data.qvel[:6] = qv[:, 0]
        inv_data.qacc[:6] = qa[:, 0]
        mujoco.mj_inverse(model, inv_data)
        
        tau_theoretical = inv_data.qfrc_inverse[:6].copy()
        
        recorded_sim['time'].append(t)
        recorded_sim['tau_sim'].append(tau_theoretical)
        
        # 物理步進
        data.ctrl[:6] = tau_theoretical
        mujoco.mj_step(model, data)

    # 3. 分析與擬合 K
    time_sim = np.array(recorded_sim['time'])
    tau_sim = np.array(recorded_sim['tau_sim'])
    
    plt.figure(figsize=(15, 10))
    print("\n📊 比例系數 K 分析結果 (Real = K * Sim):")
    
    for i in range(6):
        # 將仿真數據插值到真實數據的時間戳
        f_interp = interpolate.interp1d(time_sim, tau_sim[:, i], fill_value="extrapolate")
        tau_sim_aligned = f_interp(t_real)
        
        # 計算最優 K (最小二乘法: K = sum(sim*real) / sum(sim^2))
        k_opt = np.sum(tau_sim_aligned * tau_real_all[:, i]) / np.sum(tau_sim_aligned**2)
        
        # 繪圖
        plt.subplot(3, 2, i+1)
        plt.plot(t_real, tau_real_all[:, i], 'r', alpha=0.5, label='Real Data')
        plt.plot(time_sim, tau_sim[:, i] * k_opt, 'b--', label=f'Sim * {k_opt:.3f}')
        plt.title(f"Joint {i+1} | Suggested K: {k_opt:.4f}")
        plt.legend()
        print(f"  關節 {i+1}: 建議 K = {k_opt:.4f}")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()