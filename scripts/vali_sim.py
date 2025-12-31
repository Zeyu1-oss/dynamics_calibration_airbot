#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import mujoco
import mujoco.viewer
import os
import sys
import re
import glob
import matplotlib.pyplot as plt
from scipy import interpolate

# COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_opt(1).csv"
# COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_ga_N12T25(1).csv"
# COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_ptrnSrch_N7T25QR-6(1).csv"
# COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_ptrnSrch_N7T25QR-7(1).csv"
COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_ptrnSrch_N8T25QR_maxq1.csv"
# COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_ptrnSrch_N8T25QR-8.csv"
# COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_ptrnSrch_N8T25QR.csv"
# COMPARE_CSV_PATH = "state_machine_demo/real_data/vali_ptrnSrch_N7T25QR-5(1).csv"
# COMPARE_CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ga_N12T25(1).csv"
MODEL_XML_PATH = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml" 

def get_matched_mat_path(csv_path, models_dir="models"):
    filename = os.path.basename(csv_path)
    
    pattern = r"vali_([^.\(]+)"
    match = re.search(pattern, filename)
    
    if match:
        core_name = match.group(1).strip()
        mat_filename = f"{core_name}.mat"
        
        print(f"🔍 正在為 {filename} 尋找標識符為 '{core_name}' 的軌跡參數...")

        for root, dirs, files in os.walk(models_dir):
            if mat_filename in files:
                return os.path.join(root, mat_filename)
            
            for f in files:
                if f.lower() == mat_filename.lower():
                    return os.path.join(root, f)
    
    raise FileNotFoundError(f"❌ 找不到與 {filename} 對應的 .mat 文件（預期文件名: {core_name}.mat）。")

try:
    MAT_FILE_PATH = get_matched_mat_path(COMPARE_CSV_PATH)
    print(f"✅ 成功匹配軌跡定義文件: {MAT_FILE_PATH}")
except Exception as e:
    print(e)
    sys.exit(1)

# --- 仿真參數 ---
USE_VIEWER = True  
RECORD_DATA = True  
SIM_TIME = 25
CONTROL_HZ = 1000  
CONTROL_DT = 1.0 / CONTROL_HZ

TIME_SHIFT_COMPENSATION = -0.25 


def plot_torque_comparison(recorded_data, n_joints, external_csv_path):
    time_sim = np.array(recorded_data['time'])
    tau_sim = np.array(recorded_data['tau'])
    
    if not os.path.exists(external_csv_path):
        print(f"錯誤: 找不到外部對比文件 {external_csv_path}")
        return

    try:
        ext_data = np.loadtxt(external_csv_path, delimiter=',')
        ext_time_raw = ext_data[:, 0]
        ext_time = ext_time_raw - ext_time_raw[0] # 時間歸零
        
        # 假設 CSV 列順序: [time, q1-6, qdot1-6, tau1-6...]
        TAU_START_IDX = 13 

        fig, axes = plt.subplots(n_joints, 2, figsize=(16, 3 * n_joints))
        fig.suptitle(f'Torque Validation: MuJoCo vs Real Data\n(Source: {os.path.basename(MAT_FILE_PATH)})', 
                     fontsize=16, fontweight='bold', y=0.98)

        # 應用時間偏移補償
        adjusted_time_sim = time_sim - TIME_SHIFT_COMPENSATION

        for i in range(n_joints):
            # 對齊真實數據到仿真時間軸
            f_interp = interpolate.interp1d(ext_time, ext_data[:, TAU_START_IDX + i], 
                                           kind='linear', fill_value='extrapolate')
            tau_ext_aligned = f_interp(adjusted_time_sim)

            # 左側：曲線對比
            ax_t = axes[i, 0]
            ax_t.plot(time_sim, tau_ext_aligned, 'b-', label='Real Data (CSV)', linewidth=1.5, alpha=0.6)
            ax_t.plot(time_sim, tau_sim[:, i], 'r--', label='MuJoCo Sim', linewidth=1.2)
            ax_t.set_ylabel(f'Joint {i+1} (Nm)', fontweight='bold')
            ax_t.grid(True, linestyle=':', alpha=0.5)
            if i == 0: ax_t.legend(loc='upper right')

            # 右側：殘差
            ax_err = axes[i, 1]
            torque_error = tau_sim[:, i] - tau_ext_aligned
            ax_err.plot(time_sim, torque_error, 'g-', linewidth=0.8)
            ax_err.fill_between(time_sim, torque_error, color='green', alpha=0.1)
            rmse = np.sqrt(np.mean(torque_error**2))
            ax_err.set_title(f'RMSE: {rmse:.4f} Nm', fontsize=10)
            ax_err.grid(True, linestyle=':', alpha=0.5)

        plt.tight_layout(rect=[0, 0.03, 1, 0.94])
        os.makedirs('diagram', exist_ok=True)
        plt.savefig('diagram/torque_comparison_aligned.png', dpi=300)
        print(f"✓ 對齊圖表已保存至: diagram/torque_comparison_aligned.png")
        plt.show()

    except Exception as e:
        print(f"繪圖出錯: {e}")

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

    recorded_data = {'time': [], 'tau': []}
    inv_data = mujoco.MjData(model)

    print(f"🚀 開始仿真...")
    
    def run_step():
        t = data.time
        q_des_m, qv_des_m, _ = mixed_trajectory_calculator(t, T, N, wf, a, b, c_pol, q0)
        _, qv_next_m, _ = mixed_trajectory_calculator(t + CONTROL_DT, T, N, wf, a, b, c_pol, q0)
        target_qacc = (qv_next_m[:, 0] - qv_des_m[:, 0]) / CONTROL_DT

        inv_data.qpos[:n_joints] = data.qpos[:n_joints]
        inv_data.qvel[:n_joints] = data.qvel[:n_joints]
        inv_data.qacc[:n_joints] = target_qacc
        mujoco.mj_inverse(model, inv_data)
        
        torque_cmd = inv_data.qfrc_inverse[:n_joints].copy()
        data.ctrl[:n_joints] = torque_cmd

        if RECORD_DATA:
            recorded_data['time'].append(t)
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

    plot_torque_comparison(recorded_data, n_joints, COMPARE_CSV_PATH)

if __name__ == "__main__":
    main()