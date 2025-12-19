#!/usr/bin/env python3
"""

验证思想:
1. 使用test.py的控制逻辑分别控制两个模型
2. 记录两次仿真的tau曲线
3. 对比tau曲线
"""

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.io import loadmat
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def mixed_trajectory_calculator(t_vec, T, N, wf, a, b, c_pol, q0):
    """计算期望轨迹（与test.py一致）"""
    t_vec = np.atleast_1d(t_vec)
    J = a.shape[0]  
    M = len(t_vec)  
    qd, qdot_d, qddot_d = np.zeros((J, M)), np.zeros((J, M)), np.zeros((J, M))
    tau_vec = t_vec % T  
    
    for i in range(J):
        k_vec = np.arange(1, N + 1).reshape(-1, 1)
        wk_t = wf * k_vec * t_vec
        sin_wk_t, cos_wk_t = np.sin(wk_t), np.cos(wk_t)
        
        a_norm = a[i, :].reshape(-1, 1) / (wf * k_vec)
        b_norm = b[i, :].reshape(-1, 1) / (wf * k_vec)
        qd_fourier = (a_norm * sin_wk_t - b_norm * cos_wk_t).sum(axis=0)
        qdot_d_fourier = (a[i, :].reshape(-1, 1) * cos_wk_t + 
                         b[i, :].reshape(-1, 1) * sin_wk_t).sum(axis=0)
        qddot_d_fourier = ((-a[i, :].reshape(-1, 1) * wf * k_vec) * sin_wk_t + 
                          (b[i, :].reshape(-1, 1) * wf * k_vec) * cos_wk_t).sum(axis=0)
        
        qd_poly, qdot_d_poly, qddot_d_poly = np.zeros(M), np.zeros(M), np.zeros(M)
        for k_exp in range(6):
            c = c_pol[i, k_exp]
            qd_poly += c * (tau_vec ** k_exp)
            if k_exp >= 1: 
                qdot_d_poly += c * k_exp * (tau_vec ** (k_exp - 1))
            if k_exp >= 2: 
                qddot_d_poly += c * k_exp * (k_exp - 1) * (tau_vec ** (k_exp - 2))

        qd[i, :] = qd_fourier + qd_poly
        qdot_d[i, :] = qdot_d_fourier + qdot_d_poly
        qddot_d[i, :] = qddot_d_fourier + qddot_d_poly
    
    return qd, qdot_d, qddot_d


def run_simulation(model_path, T, N, wf, a, b, c_pol, q0, sim_time, control_dt, visualize=False):
    """
    
    Args:
        model_path: 模型XML路径
        T, N, wf, a, b, c_pol, q0: 轨迹参数
        sim_time: 仿真时长
        control_dt: 控制时间步
        visualize: 是否可视化
    
    Returns:
        recorded_data: 记录的数据
    """
    
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    model.opt.timestep = control_dt
    n_joints = model.nu
    
    model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_INVDISCRETE
    
    q_init, qv_init, _ = mixed_trajectory_calculator(0.0, T, N, wf, a, b, c_pol, q0)
    data.qpos[:n_joints] = q_init[:, 0]
    data.qvel[:n_joints] = qv_init[:, 0]
    mujoco.mj_forward(model, data)
    
    recorded_data = {
        'time': [],
        'q': [],
        'qdot': [],
        'tau': []
    }
    
    inv_data = mujoco.MjData(model)
    
    def controller(model, data):
        t = data.time
        
        q_des_m, qv_des_m, qa_des_m = mixed_trajectory_calculator(
            t, T, N, wf, a, b, c_pol, q0
        )
        q_des, qv_des, qa_des = q_des_m[:, 0], qv_des_m[:, 0], qa_des_m[:, 0]
        
        target_qacc = qa_des.copy()
        
        if (model.opt.enableflags & mujoco.mjtEnableBit.mjENBL_INVDISCRETE) and \
           (model.opt.integrator != mujoco.mjtIntegrator.mjINT_RK4):
            _, qv_next_m, _ = mixed_trajectory_calculator(
                t + model.opt.timestep, T, N, wf, a, b, c_pol, q0
            )
            target_qacc = (qv_next_m[:, 0] - qv_des) / model.opt.timestep
        
        inv_data.qpos[:n_joints] = data.qpos[:n_joints]  # 实际位置
        inv_data.qvel[:n_joints] = data.qvel[:n_joints]  # 实际速度
        inv_data.qacc[:n_joints] = target_qacc            # 调整后的期望加速度
        
        mujoco.mj_inverse(model, inv_data)
        
        tau_ff = inv_data.qfrc_inverse[:n_joints].copy()
        data.ctrl[:n_joints] = tau_ff
        
        # 记录数据
        recorded_data['time'].append(t)
        recorded_data['q'].append(data.qpos[:n_joints].copy())
        recorded_data['qdot'].append(data.qvel[:n_joints].copy())
        recorded_data['tau'].append(tau_ff.copy())
    
    # 运行仿真
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)
        print(f"    ✓ 可视化已开启 - 正在查看模型: {os.path.basename(model_path)}")
    
    n_steps = int(sim_time / control_dt)
    
    for step in range(n_steps):
        controller(model, data)
        mujoco.mj_step(model, data)
        
        if visualize and viewer is not None:
            viewer.sync()
        
        if (step + 1) % 5000 == 0:
            print(f"      进度: {step+1}/{n_steps} ({100*(step+1)/n_steps:.1f}%)")
    
    if viewer is not None:
        viewer.close()
    
    for key in recorded_data:
        recorded_data[key] = np.array(recorded_data[key])
    
    return recorded_data


def main():
    
    print("对比两个模型的tau曲线（使用test.py的控制逻辑）")
    print("\n验证思想:")
    print("  - 分别控制原始模型和校准模型")
    print("  - 记录两次仿真的tau曲线")
    print("  - 对比tau曲线")
    
    # 配置
    original_xml = "models/mjcf/manipulator/airbot_play_force/_play_force.xml"
    calibrated_xml = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml"
    mat_file = "models/ptrnSrch_N7T25QR-6.mat"
    
    sim_time = 25.0
    control_dt = 0.001
    visualize = True  # 可视化两次仿真
    
    print(f"\n[1/3] 加载轨迹参数...")
    
    mat_data = loadmat(mat_file)
    traj_par = mat_data['traj_par'][0, 0]
    
    T = float(traj_par['T'][0, 0])
    N = int(traj_par['N'][0, 0])
    wf = float(traj_par['wf'][0, 0])
    a = mat_data['a']
    b = mat_data['b']
    c_pol = mat_data['c_pol']
    q0 = traj_par['q0'].flatten()
    
    print(f"\n[2/3] 运行两次仿真...")
    
    print(f"\n  仿真1: 原始模型")
    print(f"    模型: {original_xml}")
    data_original = run_simulation(
        original_xml, T, N, wf, a, b, c_pol, q0, sim_time, control_dt, visualize
    )
    print(f"    ✓ 记录了 {len(data_original['time'])} 个数据点")
    
    print(f"\n  仿真2: 校准模型")
    print(f"    模型: {calibrated_xml}")
    data_calibrated = run_simulation(
        calibrated_xml, T, N, wf, a, b, c_pol, q0, sim_time, control_dt, visualize
    )
    print(f"    ✓ 记录了 {len(data_calibrated['time'])} 个数据点")
    
    # 3. 对比tau曲线
    print(f"\n[3/3] 对比tau曲线...")
    
    time_vec = data_original['time']
    tau_original = data_original['tau']
    tau_calibrated = data_calibrated['tau']
    
    # 计算误差
    tau_error = tau_original - tau_calibrated
    
    print("Tau对比分析")
    print("\n关节 |  RMSE(Nm)  |  最大(Nm)  |  平均(Nm)  |  相对误差%")
    
    for j in range(6):
        rmse = np.sqrt(np.mean(tau_error[:, j]**2))
        max_err = np.max(np.abs(tau_error[:, j]))
        mean_err = np.mean(np.abs(tau_error[:, j]))
        
        tau_mean = np.mean(np.abs(tau_original[:, j]))
        rel_err = (rmse / tau_mean * 100) if tau_mean > 1e-6 else 0
        
        print(f"Joint{j+1} | {rmse:10.6f} | {max_err:10.6f} | {mean_err:10.6f} | {rel_err:9.2f}%")
    
    avg_rmse = np.mean([np.sqrt(np.mean(tau_error[:, j]**2)) for j in range(6)])
    
    print("-"*65)
    print(f"平均   | {avg_rmse:10.6f}")
    
    print("\n结论:")
    if avg_rmse < 0.01:
        print("  ✅ 误差极小！两个模型tau曲线几乎完全一致！")
    elif avg_rmse < 0.1:
        print("  ✅ 误差很小！两个模型tau曲线基本一致！")
    elif avg_rmse < 0.5:
        print("  ✓ 误差较小，两个模型tau曲线接近")
    else:
        print("  ⚠️ 误差较大，两个模型tau曲线有明显差异")
    
    print("="*70)
    
    # 4. 绘制对比图
    print(f"\n[绘图] 生成tau对比图...")
    
    fig, axes = plt.subplots(3, 2, figsize=(18, 14))
    axes = axes.flatten()
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for j in range(6):
        ax = axes[j]
        
        # 绘制两条tau曲线
        ax.plot(time_vec, tau_original[:, j], '-', 
               linewidth=2, color=colors[j], alpha=0.7, label='Original Model')
        ax.plot(time_vec, tau_calibrated[:, j], '--', 
               linewidth=1.8, color='red', alpha=0.8, label='Calibrated Model')
        
        # 绘制误差（右侧y轴）
        ax2 = ax.twinx()
        ax2.plot(time_vec, tau_error[:, j], '-', 
                linewidth=1.0, color='green', alpha=0.5)
        ax2.set_ylabel('Error (Nm)', color='green', fontsize=9)
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.axhline(y=0, color='gray', linestyle=':', linewidth=0.5)
        
        # 设置标签
        ax.set_xlabel('Time (s)', fontsize=10)
        ax.set_ylabel('Torque (Nm)', fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', fontsize=9)
        
        # 误差统计
        rmse = np.sqrt(np.mean(tau_error[:, j]**2))
        max_err = np.max(np.abs(tau_error[:, j]))
        rel_err = (rmse / np.mean(np.abs(tau_original[:, j])) * 100) if np.mean(np.abs(tau_original[:, j])) > 1e-6 else 0
        
        # 根据误差设置背景颜色
        if rmse < 0.01:
            color_box = 'lightgreen'
            status = '✓✓✓ '
        elif rmse < 0.1:
            color_box = 'lightgreen'
            status = '✓✓ '
        elif rmse < 0.5:
            color_box = 'lightyellow'
            status = '✓ '
        else:
            color_box = 'lightcoral'
            status = '⚠'
        
        ax.set_title(f'Joint {j+1} {status}\nRMSE: {rmse:.4f} Nm ({rel_err:.2f}%)', 
                    fontsize=11, fontweight='bold')
        
        # 误差文本框
        ax2.text(0.98, 0.98, f'Max Err:\n{max_err:.4f} Nm',
                transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.7))
    
    # 总标题
    fig.suptitle('Tau Comparison: Original vs Calibrated Model\n' + 
                f'(Using test.py Control Logic) - Avg RMSE: {avg_rmse:.6f} Nm',
                fontsize=15, fontweight='bold', y=0.996)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存
    os.makedirs('diagram', exist_ok=True)
    output_path = 'diagram/model_comparison_tau.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存图表: {output_path}")
    
    # 显示图表
    print(f"\n  正在打开图表...")
    plt.show()
    
    # 5. 保存结果
    print(f"\n[保存] 保存对比结果...")
    
    os.makedirs('results', exist_ok=True)
    
    # 保存完整数据
    results = {
        'time': time_vec,
        'tau_original': tau_original,
        'tau_calibrated': tau_calibrated,
        'tau_error': tau_error,
        'q_original': data_original['q'],
        'q_calibrated': data_calibrated['q'],
        'qdot_original': data_original['qdot'],
        'qdot_calibrated': data_calibrated['qdot']
    }
    
    import pickle
    result_path = 'results/model_comparison_results.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"  ✓ 保存结果: {result_path}")
    
    # 保存CSV摘要
    try:
        import pandas as pd
        
        summary_data = []
        for j in range(6):
            rmse = np.sqrt(np.mean(tau_error[:, j]**2))
            max_err = np.max(np.abs(tau_error[:, j]))
            mean_err = np.mean(np.abs(tau_error[:, j]))
            rel_err = (rmse / np.mean(np.abs(tau_original[:, j])) * 100) if np.mean(np.abs(tau_original[:, j])) > 1e-6 else 0
            
            summary_data.append({
                'Joint': f'Joint{j+1}',
                'RMSE (Nm)': rmse,
                'Max Error (Nm)': max_err,
                'Mean Error (Nm)': mean_err,
                'Relative Error (%)': rel_err
            })
        
        # 添加平均值
        summary_data.append({
            'Joint': 'Average',
            'RMSE (Nm)': avg_rmse,
            'Max Error (Nm)': np.mean([np.max(np.abs(tau_error[:, j])) for j in range(6)]),
            'Mean Error (Nm)': np.mean([np.mean(np.abs(tau_error[:, j])) for j in range(6)]),
            'Relative Error (%)': np.mean([
                (np.sqrt(np.mean(tau_error[:, j]**2)) / np.mean(np.abs(tau_original[:, j])) * 100) 
                if np.mean(np.abs(tau_original[:, j])) > 1e-6 else 0 
                for j in range(6)
            ])
        })
        
        df = pd.DataFrame(summary_data)
        summary_path = 'results/model_comparison_summary.csv'
        df.to_csv(summary_path, index=False)
        print(f"  ✓ 保存摘要: {summary_path}")
        
    except ImportError:
        print("  pandas未安装,跳过CSV保存")
    
    print("✅ 对比完成！")
    print(f"\n查看结果:")
    print(f"  - 图表: diagram/model_comparison_tau.png")
    print(f"  - 数据: results/model_comparison_results.pkl")
    print(f"  - 摘要: results/model_comparison_summary.csv")
    
    if avg_rmse < 0.1:
        print(f"\n🎉 验证通过！两个模型tau曲线高度一致！")
    
    print("="*70)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
