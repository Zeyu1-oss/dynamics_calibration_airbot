#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import mujoco
import mujoco.viewer
import os
import matplotlib.pyplot as plt
from scipy import interpolate

# ================= 配置区 =================
COMPARE_CSV_PATH = "results/data/vali_ptrnSrch_N7T25QR-6_converted.csv"
MODEL_XML_PATH = "models/mjcf/manipulator/airbot_play_force/_play_force.xml" 
MAT_FILE_PATH = "models/ptrnSrch_N7T25QR-6.mat"

USE_VIEWER = True  
RECORD_DATA = True  
SIM_TIME = 25
CONTROL_HZ = 1000  
CONTROL_DT = 1.0 / CONTROL_HZ

# ================= 核心力矩对比绘图函数 =================
def plot_torque_comparison(recorded_data, n_joints, external_csv_path):
    """
    专门对比 MuJoCo 仿真生成的力矩与外部 CSV 中的力矩
    """
    time_sim = np.array(recorded_data['time'])
    tau_sim = np.array(recorded_data['tau'])
    
    if not os.path.exists(external_csv_path):
        print(f"错误: 找不到外部对比文件 {external_csv_path}")
        return

    try:
        # 加载外部数据 
        ext_data = np.loadtxt(external_csv_path, delimiter=',')
        ext_time = ext_data[:, 0]
        
        # --- 请根据你的 CSV 列顺序修改这里 ---
        # 假设前 13 列是 [time, q1-6, qdot1-6]，则力矩从索引 13 开始
        TAU_START_IDX = 13 

        fig, axes = plt.subplots(n_joints, 2, figsize=(16, 3 * n_joints))
        fig.suptitle('Torque Validation: MuJoCo Simulation vs External CSV Data', fontsize=18, fontweight='bold', y=0.98)

        for i in range(n_joints):
            # 对齐外部力矩数据到仿真时间轴
            f_interp = interpolate.interp1d(ext_time, ext_data[:, TAU_START_IDX + i], kind='linear', fill_value='extrapolate')
            tau_ext_aligned = f_interp(time_sim)

            # 左侧：力矩追踪对比 (Simulation vs External)
            ax_t = axes[i, 0]
            ax_t.plot(time_sim, tau_ext_aligned, 'b-', label='External Ref Torque', linewidth=2, alpha=0.7)
            ax_t.plot(time_sim, tau_sim[:, i], 'r--', label='MuJoCo Cmd Torque', linewidth=1.5)
            ax_t.set_ylabel(f'Joint {i+1} Torque (Nm)', fontsize=10, fontweight='bold')
            ax_t.grid(True, linestyle=':', alpha=0.6)
            if i == 0: ax_t.set_title("Torque Tracking Comparison", fontsize=12)
            ax_t.legend(loc='upper right', fontsize=8)

            # 右侧：力矩误差 (Residuals)
            ax_err = axes[i, 1]
            torque_error = tau_sim[:, i] - tau_ext_aligned
            ax_err.plot(time_sim, torque_error, 'g-', linewidth=1)
            ax_err.fill_between(time_sim, torque_error, color='green', alpha=0.1)
            ax_err.set_ylabel('Error (Nm)', fontsize=10)
            ax_err.grid(True, linestyle=':', alpha=0.6)
            
            rmse = np.sqrt(np.mean(torque_error**2))
            max_err = np.max(np.abs(torque_error))
            ax_err.set_title(f'RMSE: {rmse:.4f} Nm | Max: {max_err:.4f} Nm', fontsize=10)
            if i == 0: ax_err.set_title("Tracking Error (Sim - Ext)", fontsize=12)

        plt.xlabel("Time (s)", fontsize=12)
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        os.makedirs('diagram', exist_ok=True)
        plt.savefig('diagram/torque_comparison.png', dpi=300)
        print("✓ 力矩对比图已保存至: diagram/torque_comparison.png")
        plt.show()

    except Exception as e:
        print(f"绘图出错: {e}。请检查 CSV 索引 TAU_START_IDX 是否正确。")

# ================= 轨迹计算 (与辨识代码一致) =================
def mixed_trajectory_calculator(t_vec, T, N, wf, a, b, c_pol, q0):
    t_vec = np.atleast_1d(t_vec)
    J = a.shape[0]  
    M = len(t_vec)  
    qd, qdot_d, qddot_d = np.zeros((J, M)), np.zeros((J, M)), np.zeros((J, M))
    tau_vec = t_vec % T  
    
    for i in range(J):
        k_vec = np.arange(1, N + 1).reshape(-1, 1)
        wk_t = wf * k_vec * t_vec
        sin_wk_t, cos_wk_t = np.sin(wk_t), np.cos(wk_t)
        
        a_norm, b_norm = a[i, :].reshape(-1, 1) / (wf * k_vec), b[i, :].reshape(-1, 1) / (wf * k_vec)
        qd_fourier = (a_norm * sin_wk_t - b_norm * cos_wk_t).sum(axis=0)
        qdot_d_fourier = (a[i, :].reshape(-1, 1) * cos_wk_t + b[i, :].reshape(-1, 1) * sin_wk_t).sum(axis=0)
        qddot_d_fourier = ((-a[i, :].reshape(-1, 1) * wf * k_vec) * sin_wk_t + 
                           (b[i, :].reshape(-1, 1) * wf * k_vec) * cos_wk_t).sum(axis=0)
        
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

# ================= 主程序 =================
def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = CONTROL_DT
    n_joints = model.nu
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE

    mat_contents = sio.loadmat(MAT_FILE_PATH)
    a, b, c_pol = mat_contents['a'], mat_contents['b'], mat_contents['c_pol']
    tp = mat_contents['traj_par'][0, 0]
    T, N, wf, q0 = tp['T'][0,0], int(tp['N'][0,0]), tp['wf'][0,0], tp['q0']

    q_init, qv_init, _ = mixed_trajectory_calculator(0.0, T, N, wf, a, b, c_pol, q0)
    data.qpos[:n_joints] = q_init[:, 0]
    data.qvel[:n_joints] = qv_init[:, 0]
    mujoco.mj_forward(model, data)

    # 重点：增加 'tau' 键来记录力矩
    recorded_data = {'time': [], 'q': [], 'tau': []}
    inv_data = mujoco.MjData(model) 

    print(f"开始仿真 (时长: {SIM_TIME}s)...")
    
    def run_step():
        t = data.time
        q_des_m, qv_des_m, qa_des_m = mixed_trajectory_calculator(t, T, N, wf, a, b, c_pol, q0)
        q_des, qv_des = q_des_m[:, 0], qv_des_m[:, 0]

        # 离散时间加速度修正
        _, qv_next_m, _ = mixed_trajectory_calculator(t + CONTROL_DT, T, N, wf, a, b, c_pol, q0)
        target_qacc = (qv_next_m[:, 0] - qv_des) / CONTROL_DT

        # 逆动力学计算 (Feedforward Torque)
        inv_data.qpos[:n_joints] = data.qpos[:n_joints]
        inv_data.qvel[:n_joints] = data.qvel[:n_joints]
        inv_data.qacc[:n_joints] = target_qacc
        mujoco.mj_inverse(model, inv_data)
        
        torque_cmd = inv_data.qfrc_inverse[:n_joints].copy()
        data.ctrl[:n_joints] = torque_cmd

        if RECORD_DATA:
            recorded_data['time'].append(t)
            recorded_data['q'].append(data.qpos[:n_joints].copy())
            recorded_data['tau'].append(torque_cmd)

    if USE_VIEWER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < SIM_TIME:
                run_step()
                mujoco.mj_step(model, data)
                if len(recorded_data['time']) % 10 == 0:
                    viewer.sync()
    else:
        while data.time < SIM_TIME:
            run_step()
            mujoco.mj_step(model, data)

    print("仿真完成，生成力矩对比分析...")
    plot_torque_comparison(recorded_data, n_joints, COMPARE_CSV_PATH)

if __name__ == "__main__":
    main()