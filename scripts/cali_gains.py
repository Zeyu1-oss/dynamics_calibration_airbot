#!/usr/bin/env python3

import numpy as np
import scipy.io as sio
import mujoco
import os
import glob
import re
from scipy import interpolate

# --- 路徑配置 ---
INPUT_DATA_DIR = "state_machine_demo/real_data"
MODEL_XML_PATH = "models/mjcf/manipulator/airbot_play_force/_play_force.xml" 
OUTPUT_FOLDER = "results/unified_corrected_j2j3"

# --- 參數 ---
SIM_TIME = 25
CONTROL_DT = 0.001
TIME_SHIFT_COMPENSATION = -0.25  

def get_matched_mat_path(csv_path, models_dir="models"):
    filename = os.path.basename(csv_path)
    pattern = r"vali_([^.\(]+)"
    match = re.search(pattern, filename)
    if match:
        core_name = match.group(1).strip()
        mat_filename = f"{core_name}.mat"
        for root, dirs, files in os.walk(models_dir):
            if mat_filename in files: return os.path.join(root, mat_filename)
    return None

def calc_traj(t, T, N, wf, a, b, c_pol):
    tau_v = t % T
    qd, qv, qa = np.zeros(6), np.zeros(6), np.zeros(6)
    for i in range(6):
        k_idx = np.arange(1, N + 1)
        wk_t = wf * k_idx * t
        a_n, b_n = a[i,:]/(wf*k_idx), b[i,:]/(wf*k_idx)
        qd[i] = (a_n * np.sin(wk_t) - b_n * np.cos(wk_t)).sum() + sum(c_pol[i,k]*(tau_v**k) for k in range(6))
        qv[i] = (a[i,:]*np.cos(wk_t) + b[i,:]*np.sin(wk_t)).sum() + sum(c_pol[i,k]*k*(tau_v**(k-1)) for k in range(1,6))
        qa[i] = ((-a[i,:]*wf*k_idx)*np.sin(wk_t) + (b[i,:]*wf*k_idx)*np.cos(wk_t)).sum() + sum(c_pol[i,k]*k*(k-1)*(tau_v**(k-2)) for k in range(2,6))
    return qd, qv, qa

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    all_files = sorted(glob.glob(os.path.join(INPUT_DATA_DIR, "vali_*.csv")))
    
    if not all_files:
        print("❌ 未找到數據文件")
        return

    # --- 第一階段：分析所有數據，計算全局平均 K ---
    print(f"🔍 階段 1: 正在分析 {len(all_files)} 個文件以獲取全局最優 K...")
    all_k2 = []
    all_k3 = []

    for f in all_files:
        mat_path = get_matched_mat_path(f)
        if not mat_path: continue
        
        mat_contents = sio.loadmat(mat_path)
        a, b, c_pol = mat_contents['a'], mat_contents['b'], mat_contents['c_pol']
        tp = mat_contents['traj_par'][0, 0]
        T, N, wf = tp['T'][0,0], int(tp['N'][0,0]), tp['wf'][0,0]

        inv_data = mujoco.MjData(model)
        sim_tau = []
        sim_time = []
        
        # 快速仿真獲取理論力矩
        t_sim = 0
        while t_sim < SIM_TIME:
            qd, qv, _ = calc_traj(t_sim, T, N, wf, a, b, c_pol)
            _, qv_next, _ = calc_traj(t_sim + CONTROL_DT, T, N, wf, a, b, c_pol)
            inv_data.qpos[:6], inv_data.qvel[:6] = qd, qv
            inv_data.qacc[:6] = (qv_next - qv) / CONTROL_DT
            mujoco.mj_inverse(model, inv_data)
            sim_time.append(t_sim)
            sim_tau.append(inv_data.qfrc_inverse[:6].copy())
            t_sim += CONTROL_DT
        
        # 對齊並計算 K
        real_raw = np.loadtxt(f, delimiter=',')
        real_time = real_raw[:, 0] - real_raw[0, 0]
        sim_tau = np.array(sim_tau)
        sim_time = np.array(sim_time)
        
        for i, target_list in zip([1, 2], [all_k2, all_k3]):
            f_interp = interpolate.interp1d(sim_time - TIME_SHIFT_COMPENSATION, sim_tau[:, i], fill_value="extrapolate")
            sim_aligned = f_interp(real_time)
            k = np.sum(sim_aligned * real_raw[:, 13 + i]) / np.sum(sim_aligned**2)
            target_list.append(k)

    # 計算平均最優 K
    final_k2 = np.mean(all_k2)
    final_k3 = np.mean(all_k3)
    
    print(f"\n✅ 分析完成:")
    print(f"📊 J2 全局平均 K = {final_k2:.6f} (標準差: {np.std(all_k2):.4f})")
    print(f"📊 J3 全局平均 K = {final_k3:.6f} (標準差: {np.std(all_k3):.4f})")

    # --- 第二階段：使用統一 K 處理所有原始數據 ---
    print(f"\n🚀 階段 2: 正在使用統一 K 值修正並保存數據...")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    for f in all_files:
        real_raw = np.loadtxt(f, delimiter=',')
        real_corrected = real_raw.copy()
        
        # 僅修正 J2 和 J3
        real_corrected[:, 14] = real_raw[:, 14] / final_k2
        real_corrected[:, 15] = real_raw[:, 15] / final_k3
        
        new_filename = os.path.join(OUTPUT_FOLDER, f"unified_corrected_{os.path.basename(f)}")
        np.savetxt(new_filename, real_corrected, delimiter=',', fmt='%.6f')
        print(f"   已處理: {os.path.basename(f)}")

    print(f"\n✨ 任務成功完成！修正後的數據保存在: {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()