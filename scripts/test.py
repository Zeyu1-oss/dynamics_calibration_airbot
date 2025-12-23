#!/usr/bin/env python3

from re import T
import numpy as np
import scipy.io as sio
import mujoco
import mujoco.viewer
import os
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
import time

MODEL_XML_PATH = "models/mjcf/manipulator/airbot_play_force/_play_force.xml" 
MAT_FILE_PATH = "models/ptrnSrch_N7T25QR-6.mat"
OUTPUT_CSV_PATH = "results/data_csv/vali——0fre.csv" 
USE_VIEWER = True  
RECORD_DATA = True  
SIM_TIME = 25
CONTROL_HZ = 1000  
CONTROL_DT = 1.0 / CONTROL_HZ

USE_FEEDBACK = True


def plot_comparison(recorded_data, T, N, wf, a, b, c_pol, q0, n_joints):
    """
    绘制理论轨迹与实际轨迹的对比图，保持与 torque_control.py 相同的输出格式
    """

    time_actual = np.array(recorded_data['time'])
    q_actual = np.array(recorded_data['q'])
    qdot_actual = np.array(recorded_data['qdot'])
    tau_actual = np.array(recorded_data['tau'])
    tau_ff = np.array(recorded_data['tau_ff'])
    tau_fb = np.array(recorded_data['tau_fb'])

    # 计算参考轨迹（使用相同时间戳）
    q_theory = np.zeros((len(time_actual), n_joints))
    qdot_theory = np.zeros((len(time_actual), n_joints))
    for idx, t in enumerate(time_actual):
        qd, qdot_d, _ = mixed_trajectory_calculator(t, T, N, wf, a, b, c_pol, q0)
        q_theory[idx, :] = qd[:, 0]
        qdot_theory[idx, :] = qdot_d[:, 0]

    fig = plt.figure(figsize=(18, 12))
    fig.suptitle(f'Theoretical Trajectory vs Torque Control Results (Feedback: {USE_FEEDBACK})',
                 fontsize=16, fontweight='bold')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    for joint_idx in range(n_joints):
        # 位置
        ax_q = plt.subplot(n_joints, 4, joint_idx * 4 + 1)
        ax_q.plot(time_actual, q_theory[:, joint_idx], label='Theoretical',
                  linestyle='-', linewidth=2, color=colors[joint_idx], alpha=0.7)
        ax_q.plot(time_actual, q_actual[:, joint_idx], label='Actual',
                  linestyle='--', linewidth=1.5, color='red', alpha=0.8)
        ax_q.set_ylabel(f'J{joint_idx+1}\nPos (rad)', fontsize=9)
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(fontsize=7, loc='upper right')
        if joint_idx == 0:
            ax_q.set_title('Position', fontweight='bold')
        if joint_idx == n_joints - 1:
            ax_q.set_xlabel('Time (s)', fontsize=9)
        q_error = q_actual[:, joint_idx] - q_theory[:, joint_idx]
        ax_q.text(0.02, 0.98,
                  f'RMSE: {np.sqrt(np.mean(q_error**2)):.4f}\nMax: {np.max(np.abs(q_error)):.4f}',
                  transform=ax_q.transAxes, fontsize=7, verticalalignment='top',
                  bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 速度
        ax_qdot = plt.subplot(n_joints, 4, joint_idx * 4 + 2)
        ax_qdot.plot(time_actual, qdot_theory[:, joint_idx], label='Theoretical',
                     linestyle='-', linewidth=2, color=colors[joint_idx], alpha=0.7)
        ax_qdot.plot(time_actual, qdot_actual[:, joint_idx], label='Actual',
                     linestyle='--', linewidth=1.5, color='red', alpha=0.8)
        ax_qdot.set_ylabel(f'J{joint_idx+1}\nVel (rad/s)', fontsize=9)
        ax_qdot.grid(True, alpha=0.3)
        ax_qdot.legend(fontsize=7, loc='upper right')
        if joint_idx == 0:
            ax_qdot.set_title('Velocity', fontweight='bold')
        if joint_idx == n_joints - 1:
            ax_qdot.set_xlabel('Time (s)', fontsize=9)
        qdot_error = qdot_actual[:, joint_idx] - qdot_theory[:, joint_idx]
        ax_qdot.text(0.02, 0.98,
                     f'RMSE: {np.sqrt(np.mean(qdot_error**2)):.4f}\nMax: {np.max(np.abs(qdot_error)):.4f}',
                     transform=ax_qdot.transAxes, fontsize=7, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 扭矩
        ax_tau = plt.subplot(n_joints, 4, joint_idx * 4 + 3)
        ax_tau.plot(time_actual, tau_ff[:, joint_idx], label='FF (Inverse Dyn)',
                    linestyle='-', linewidth=2, color=colors[joint_idx], alpha=0.7)
        ax_tau.plot(time_actual, tau_actual[:, joint_idx], label='Total',
                    linestyle='--', linewidth=1.5, color='red', alpha=0.8)
        ax_tau.set_ylabel(f'J{joint_idx+1}\nTorque (Nm)', fontsize=9)
        ax_tau.grid(True, alpha=0.3)
        ax_tau.legend(fontsize=7, loc='upper right')
        if joint_idx == 0:
            ax_tau.set_title('Torque', fontweight='bold')
        if joint_idx == n_joints - 1:
            ax_tau.set_xlabel('Time (s)', fontsize=9)
        tau_error = tau_actual[:, joint_idx] - tau_ff[:, joint_idx]
        ax_tau.text(0.02, 0.98,
                    f'RMSE: {np.sqrt(np.mean(tau_error**2)):.4f}\nMax: {np.max(np.abs(tau_error)):.4f}',
                    transform=ax_tau.transAxes, fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # 反馈扭矩分量
        ax_fb = plt.subplot(n_joints, 4, joint_idx * 4 + 4)
        ax_fb.plot(time_actual, tau_fb[:, joint_idx], label='Feedback',
                   linestyle='-', linewidth=1.5, color='green', alpha=0.8)
        ax_fb.axhline(y=0, color='gray', linestyle=':', linewidth=1)
        ax_fb.set_ylabel(f'J{joint_idx+1}\nFB Torque (Nm)', fontsize=9)
        ax_fb.grid(True, alpha=0.3)
        ax_fb.legend(fontsize=7, loc='upper right')
        if joint_idx == 0:
            ax_fb.set_title('Feedback Component', fontweight='bold')
        if joint_idx == n_joints - 1:
            ax_fb.set_xlabel('Time (s)', fontsize=9)
        fb_mean = np.mean(np.abs(tau_fb[:, joint_idx]))
        fb_max = np.max(np.abs(tau_fb[:, joint_idx]))
        ax_fb.text(0.02, 0.98, f'Mean: {fb_mean:.4f}\nMax: {fb_max:.4f}',
                   transform=ax_fb.transAxes, fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    plt.tight_layout()
    os.makedirs('diagram', exist_ok=True)
    fig_path = "diagram/trajectory_comparison.png"
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"✓ 对比图已保存: {fig_path}")
    plt.show()

    print("\n" + "="*60)
    print("总体误差统计:")
    print("="*60)
    for joint_idx in range(n_joints):
        q_error = q_actual[:, joint_idx] - q_theory[:, joint_idx]
        qdot_error = qdot_actual[:, joint_idx] - qdot_theory[:, joint_idx]
        fb_mean = np.mean(np.abs(tau_fb[:, joint_idx]))
        fb_max = np.max(np.abs(tau_fb[:, joint_idx]))
        print(f"\n关节 {joint_idx+1}:")
        print(f"  位置 RMSE: {np.sqrt(np.mean(q_error**2)):.6f} rad ({np.sqrt(np.mean(q_error**2))*180/np.pi:.4f}°)")
        print(f"  位置最大误差: {np.max(np.abs(q_error)):.6f} rad ({np.max(np.abs(q_error))*180/np.pi:.4f}°)")
        print(f"  速度 RMSE: {np.sqrt(np.mean(qdot_error**2)):.6f} rad/s")
        print(f"  速度最大误差: {np.max(np.abs(qdot_error)):.6f} rad/s")
        print(f"  反馈扭矩平均: {fb_mean:.4f} Nm")
        print(f"  反馈扭矩最大: {fb_max:.4f} Nm")
    print("="*60)

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

def main():


    model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
    data = mujoco.MjData(model)
    model.opt.timestep = CONTROL_DT
    n_joints = model.nu


    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE

    # 3. 加载轨迹
    mat_contents = sio.loadmat(MAT_FILE_PATH)
    a, b, c_pol = mat_contents['a'], mat_contents['b'], mat_contents['c_pol']
    tp = mat_contents['traj_par'][0, 0]
    T, N, wf, q0 = tp['T'][0,0], int(tp['N'][0,0]), tp['wf'][0,0], tp['q0']

    # 初始化状态
    q_init, qv_init, _ = mixed_trajectory_calculator(0.0, T, N, wf, a, b, c_pol, q0)
    data.qpos[:n_joints] = q_init[:, 0]
    data.qvel[:n_joints] = qv_init[:, 0]
    mujoco.mj_forward(model, data)

    recorded_data = {
        'time': [],
        'q': [],
        'qdot': [],
        'qddot': [],
        'tau': [],
        'tau_ff': [],
        'tau_fb': []
    }
    inv_data = mujoco.MjData(model) 
    
    step_count = 0
    last_print_time = 0.0

    def controller(model, data):
        nonlocal step_count, last_print_time
        t = data.time
        
        q_des_m, qv_des_m, qa_des_m = mixed_trajectory_calculator(t, T, N, wf, a, b, c_pol, q0)
        q_des, qv_des, qa_des = q_des_m[:, 0], qv_des_m[:, 0], qa_des_m[:, 0]

        target_qacc = qa_des.copy()
        
        if (model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_INVDISCRETE) and \
           (model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4):
            # 获取下一个步长的理论速度期望
            _, qv_next_m, _ = mixed_trajectory_calculator(t + model.opt.timestep, T, N, wf, a, b, c_pol, q0)
            target_qacc = (qv_next_m[:, 0] - qv_des) / model.opt.timestep

        inv_data.qpos[:n_joints] = data.qpos[:n_joints]  # 实际位置
        inv_data.qvel[:n_joints] = data.qvel[:n_joints]  # 实际速度
        inv_data.qacc[:n_joints] = target_qacc           # 期望加速度

        mujoco.mj_inverse(model, inv_data)
        
        tau_ff = inv_data.qfrc_inverse[:n_joints].copy()
        tau_fb = np.zeros(n_joints)  #
        tau_total = tau_ff.copy()

        data.ctrl[:n_joints] = tau_total

        if RECORD_DATA:
            recorded_data['time'].append(t)
            recorded_data['q'].append(data.qpos[:n_joints].copy())
            recorded_data['qdot'].append(data.qvel[:n_joints].copy())
            recorded_data['qddot'].append(data.qacc[:n_joints].copy())
            recorded_data['tau'].append(tau_total)
            recorded_data['tau_ff'].append(tau_ff)
            recorded_data['tau_fb'].append(tau_fb)

    # 4. 运行
    if USE_VIEWER:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            while viewer.is_running() and data.time < SIM_TIME:
                # step_start = time.time()
                controller(model, data)
                mujoco.mj_step(model, data)
                # viewer.sync()
                # # 控制仿真频率
                # elapsed = time.time() - step_start
                # if elapsed < model.opt.timestep:
                #     time.sleep(model.opt.timestep - elapsed)

                step_count += 1
                if step_count % 10 == 0:
                    viewer.sync()
    else:
        while data.time < SIM_TIME:
            controller(model, data)
            mujoco.mj_step(model, data)

    print(f"仿真完成。数据点数量: {len(recorded_data['time'])}")

    if RECORD_DATA and len(recorded_data['time']) > 0:
        print("\n正在保存仿真结果...")

        time_array = np.array(recorded_data['time'])
        q_array = np.array(recorded_data['q'])
        qdot_array = np.array(recorded_data['qdot'])
        qddot_array = np.array(recorded_data['qddot'])
        tau_array = np.array(recorded_data['tau'])
        tau_ff_array = np.array(recorded_data['tau_ff'])
        tau_fb_array = np.array(recorded_data['tau_fb'])

        # 计算参考轨迹
        q_des_array = np.zeros_like(q_array)
        qdot_des_array = np.zeros_like(qdot_array)
        for idx, t in enumerate(time_array):
            qd, qdot_d, _ = mixed_trajectory_calculator(t, T, N, wf, a, b, c_pol, q0)
            q_des_array[idx, :] = qd[:, 0]
            qdot_des_array[idx, :] = qdot_d[:, 0]

        # 重采样到100Hz（与理论数据保持一致）
        sample_interval = 0.01  # 100 Hz
        resampled_times = np.arange(time_array[0], time_array[-1], sample_interval)
        q_resampled = np.zeros((len(resampled_times), n_joints))
        qdot_resampled = np.zeros((len(resampled_times), n_joints))
        qddot_resampled = np.zeros((len(resampled_times), n_joints))
        tau_resampled = np.zeros((len(resampled_times), n_joints))
        tau_ff_resampled = np.zeros((len(resampled_times), n_joints))
        tau_fb_resampled = np.zeros((len(resampled_times), n_joints))

        for i in range(n_joints):
            f_q = interpolate.interp1d(time_array, q_array[:, i], kind='linear', fill_value='extrapolate')
            f_qdot = interpolate.interp1d(time_array, qdot_array[:, i], kind='linear', fill_value='extrapolate')
            f_qddot = interpolate.interp1d(time_array, qddot_array[:, i], kind='linear', fill_value='extrapolate')
            f_tau = interpolate.interp1d(time_array, tau_array[:, i], kind='linear', fill_value='extrapolate')
            f_tau_ff = interpolate.interp1d(time_array, tau_ff_array[:, i], kind='linear', fill_value='extrapolate')
            f_tau_fb = interpolate.interp1d(time_array, tau_fb_array[:, i], kind='linear', fill_value='extrapolate')

            q_resampled[:, i] = f_q(resampled_times)
            qdot_resampled[:, i] = f_qdot(resampled_times)
            qddot_resampled[:, i] = f_qddot(resampled_times)
            tau_resampled[:, i] = f_tau(resampled_times)
            tau_ff_resampled[:, i] = f_tau_ff(resampled_times)
            tau_fb_resampled[:, i] = f_tau_fb(resampled_times)

        detailed_data = np.column_stack([
            resampled_times,
            q_resampled,
            qdot_resampled,
            qddot_resampled,
            tau_resampled,
            tau_resampled,
            tau_resampled
        ])

        detailed_path = OUTPUT_CSV_PATH.replace('.csv', '_detailed.csv')
        np.savetxt(detailed_path, detailed_data, delimiter=',', fmt='%.6f',
                   header='time, q1-6, qdot1-6, qddot1-6, tau_total1-6, tau_ff1-6, tau_fb1-6')

        parseur_data = np.column_stack([
            resampled_times,
            q_resampled,
            qdot_resampled,
            tau_resampled,
            tau_resampled,
            tau_resampled
        ])
        np.savetxt(OUTPUT_CSV_PATH, parseur_data, delimiter=',', fmt='%.6f')

        print(f"✓ 详细数据已保存到: {detailed_path}")
        print(f"✓ 兼容数据已保存到: {OUTPUT_CSV_PATH}")
        print(f"  - 原始数据点: {len(time_array)}")
        print(f"  - 重采样后数据点: {len(resampled_times)} (100Hz)")
        print(f"  - 时间范围: {resampled_times[0]:.3f}s - {resampled_times[-1]:.3f}s")

        # 生成对比图（与 torque_control.py 一致）
        try:
            plot_comparison(recorded_data, T, N, wf, a, b, c_pol, q0, n_joints)
        except Exception as e:
            print(f"警告：生成对比图时出错 - {e}")

        print("\n" + "="*60)
        print("仿真结果统计:")
        print("="*60)
        q_errors = []
        qdot_errors = []
        fb_magnitudes = []
        for i in range(n_joints):
            q_error = q_array[:, i] - q_des_array[:, i]
            qdot_error = qdot_array[:, i] - qdot_des_array[:, i]
            q_errors.append(np.sqrt(np.mean(q_error**2)))
            qdot_errors.append(np.sqrt(np.mean(qdot_error**2)))
            fb_magnitudes.append(np.mean(np.abs(tau_fb_array[:, i])))
        print(f"位置:")
        print(f"  - 平均RMSE: {np.mean(q_errors):.6f} rad ({np.mean(q_errors)*180/np.pi:.4f}°)")
        print(f"  - 最佳关节: {np.argmin(q_errors)+1} (RMSE: {np.min(q_errors):.6f} rad)")
        print(f"  - 最差关节: {np.argmax(q_errors)+1} (RMSE: {np.max(q_errors):.6f} rad)")
        print(f"\n速度:")
        print(f"  - 平均RMSE: {np.mean(qdot_errors):.6f} rad/s")

        print(f"\n数据质量评估:")
        if np.mean(q_errors) < 0.01:
            print("  ✓ 数据质量优秀，适合参数识别")
        elif np.mean(q_errors) < 0.05:
            print("  ⚠ 数据质量良好，可以考虑用于参数识别")
        else:
            print("  ✗ 数据质量较差，建议调整控制参数或检查模型")
    else:
        print("\n警告：没有记录到有效数据")

if __name__ == "__main__":
    main()