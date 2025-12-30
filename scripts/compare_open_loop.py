#!/usr/bin/env python3
"""
正确的模型对比方法：开环逆动力学对比

这个脚本修正了原始compare_models.py的问题：
1. 使用相同的期望轨迹状态(q, qd, qdd)
2. 不运行闭环仿真（避免轨迹divergence）
3. 直接对比两个模型在相同状态下的逆动力学计算结果
"""

import numpy as np
import mujoco
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


def compute_inverse_dynamics_open_loop(model_path, q_traj, qd_traj, qdd_traj):
    """
    开环计算逆动力学
    
    Args:
        model_path: 模型XML路径
        q_traj: 位置轨迹 (6, N)
        qd_traj: 速度轨迹 (6, N)
        qdd_traj: 加速度轨迹 (6, N)
    
    Returns:
        tau_traj: 力矩轨迹 (N, 6)
    """
    print(f"    加载模型: {model_path}")
    model = mujoco.MjModel.from_xml_path(model_path)
    data = mujoco.MjData(model)
    
    n_samples = q_traj.shape[1]
    n_joints = model.nu
    tau_traj = np.zeros((n_samples, n_joints))
    
    print(f"    计算 {n_samples} 个时间点的逆动力学...")
    
    for i in range(n_samples):
        data.qpos[:n_joints] = q_traj[:, i]
        data.qvel[:n_joints] = qd_traj[:, i]
        data.qacc[:n_joints] = qdd_traj[:, i]
        
        mujoco.mj_inverse(model, data)
        
        tau_traj[i] = data.qfrc_inverse[:n_joints].copy()
        
        if (i + 1) % 5000 == 0:
            print(f"      进度: {i+1}/{n_samples} ({100*(i+1)/n_samples:.1f}%)")
    
    print(f"    ✓ 完成")
    return tau_traj


def main():
    
    # 配置
    original_xml = "models/mjcf/manipulator/airbot_play_force/_play_force.xml"
    calibrated_xml = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml"
    mat_file = "models/ptrnSrch_N7T25QR-5.mat"
    
    sim_time = 25.0
    dt = 0.001
    
    # 检查文件
    for path in [original_xml, calibrated_xml, mat_file]:
        if not os.path.exists(path):
            print(f"❌ 文件不存在: {path}")
            return 1
    
    print("\n[1/4] 生成期望轨迹...")
    
    mat_data = loadmat(mat_file)
    traj_par = mat_data['traj_par'][0, 0]
    
    T = float(traj_par['T'][0, 0])
    N = int(traj_par['N'][0, 0])
    wf = float(traj_par['wf'][0, 0])
    a = mat_data['a']
    b = mat_data['b']
    c_pol = mat_data['c_pol']
    q0 = traj_par['q0'].flatten()
    
    print(f"  轨迹参数: T={T}s, N={N}, wf={wf:.4f}")
    
    time_vec = np.arange(0, sim_time, dt)
    n_samples = len(time_vec)
    
    print(f"  生成 {n_samples} 个采样点...")
    q_des, qd_des, qdd_des = mixed_trajectory_calculator(
        time_vec, T, N, wf, a, b, c_pol, q0
    )
    
    print(f"  ✓ 轨迹形状: q={q_des.shape}, qd={qd_des.shape}, qdd={qdd_des.shape}")
    
    # 2. 计算原始模型的逆动力学
    print("\n[2/4] 计算原始模型的逆动力学...")
    tau_original = compute_inverse_dynamics_open_loop(
        original_xml, q_des, qd_des, qdd_des
    )
    
    # 3. 计算校准模型的逆动力学
    print("\n[3/4] 计算校准模型的逆动力学...")
    tau_calibrated = compute_inverse_dynamics_open_loop(
        calibrated_xml, q_des, qd_des, qdd_des
    )
    
    # 4. 对比分析
    print("\n[4/4] 对比分析...")
    
    tau_error = tau_original - tau_calibrated
    
    print("\n" + "="*60)
    print("开环逆动力学对比结果")
    print("="*60)
    print("\n关节 |  RMSE(Nm)  |  最大(Nm)  |  平均(Nm)  |  相对误差%")
    print("-" * 60)
    
    rmse_list = []
    for j in range(6):
        rmse = np.sqrt(np.mean(tau_error[:, j]**2))
        max_err = np.max(np.abs(tau_error[:, j]))
        mean_err = np.mean(np.abs(tau_error[:, j]))
        
        tau_mean = np.mean(np.abs(tau_original[:, j]))
        rel_err = (rmse / tau_mean * 100) if tau_mean > 1e-6 else 0
        
        rmse_list.append(rmse)
        
        # 根据误差大小显示状态标记
        if rmse < 1e-5:
            status = "✓✓✓"  # 极好
        elif rmse < 1e-4:
            status = "✓✓ "  # 很好
        elif rmse < 1e-3:
            status = "✓  "  # 好
        elif rmse < 0.01:
            status = "~  "  # 一般
        else:
            status = "⚠  "  # 需要检查
        
        print(f"Joint{j+1} {status}| {rmse:10.6e} | {max_err:10.6e} | {mean_err:10.6e} | {rel_err:9.2f}%")
    
    avg_rmse = np.mean(rmse_list)
    print("-" * 60)
    print(f"平均      | {avg_rmse:10.6e}")
    print("="*60)
    
    # 评估结果
    print("\n结果评估:")
    if avg_rmse < 1e-5:
        print("  ✓✓✓ 优秀！参数估计非常准确（数值精度级别）")
    elif avg_rmse < 1e-4:
        print("  ✓✓  很好！参数估计准确")
    elif avg_rmse < 1e-3:
        print("  ✓   良好，参数估计基本准确")
    elif avg_rmse < 0.01:
        print("  ~   一般，可能需要检查参数估计或转换")
    else:
        print("  ⚠   较大误差，需要检查:")
        print("      1. 惯性参数转换是否正确（是否添加了 .T）")
        print("      2. parse_urdf.py 的转换逻辑")
        print("      3. 估计参数的约束条件")
    
    # 5. 绘制对比图
    print("\n[绘图] 生成tau对比图...")
    
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
        rmse = rmse_list[j]
        max_err = np.max(np.abs(tau_error[:, j]))
        rel_err = (rmse / np.mean(np.abs(tau_original[:, j])) * 100) if np.mean(np.abs(tau_original[:, j])) > 1e-6 else 0
        
        # 根据误差设置背景颜色
        if rmse < 1e-5:
            color_box = 'lightgreen'
            status = '✓✓✓'
        elif rmse < 1e-4:
            color_box = 'lightgreen'
            status = '✓✓'
        elif rmse < 1e-3:
            color_box = 'lightyellow'
            status = '✓'
        else:
            color_box = 'lightcoral'
            status = '⚠'
        
        ax.set_title(f'Joint {j+1} {status}\nRMSE: {rmse:.4e} Nm ({rel_err:.2f}%)', 
                    fontsize=11, fontweight='bold')
        
        # 误差文本框
        ax2.text(0.98, 0.98, f'Max Err:\n{max_err:.4e} Nm',
                transform=ax2.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor=color_box, alpha=0.7))
    
    # 总标题
    fig.suptitle('Open-Loop Inverse Dynamics Comparison\n' + 
                f'Original vs Calibrated Model - Avg RMSE: {avg_rmse:.6e} Nm',
                fontsize=15, fontweight='bold', y=0.996)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    # 保存
    os.makedirs('diagram', exist_ok=True)
    output_path = 'diagram/model_comparison_open_loop.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"  ✓ 保存图表: {output_path}")
    
    # 6. 保存结果
    print("\n[保存] 保存结果数据...")
    
    os.makedirs('results', exist_ok=True)
    
    # 保存完整数据
    results = {
        'time': time_vec,
        'q_des': q_des.T,  # (N, 6)
        'qd_des': qd_des.T,
        'qdd_des': qdd_des.T,
        'tau_original': tau_original,
        'tau_calibrated': tau_calibrated,
        'tau_error': tau_error,
        'rmse_per_joint': rmse_list,
        'avg_rmse': avg_rmse
    }
    
    import pickle
    result_path = 'results/model_comparison_open_loop.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(results, f)
    
    print(f"  ✓ 保存结果: {result_path}")
    
    # 保存CSV摘要
    try:
        import pandas as pd
        
        summary_data = []
        for j in range(6):
            rmse = rmse_list[j]
            max_err = np.max(np.abs(tau_error[:, j]))
            mean_err = np.mean(np.abs(tau_error[:, j]))
            rel_err = (rmse / np.mean(np.abs(tau_original[:, j])) * 100) if np.mean(np.abs(tau_original[:, j])) > 1e-6 else 0
            
            summary_data.append({
                'Joint': f'Joint{j+1}',
                'RMSE (Nm)': f'{rmse:.6e}',
                'Max Error (Nm)': f'{max_err:.6e}',
                'Mean Error (Nm)': f'{mean_err:.6e}',
                'Relative Error (%)': f'{rel_err:.2f}'
            })
        
        # 添加平均值
        summary_data.append({
            'Joint': 'Average',
            'RMSE (Nm)': f'{avg_rmse:.6e}',
            'Max Error (Nm)': f'{np.mean([np.max(np.abs(tau_error[:, j])) for j in range(6)]):.6e}',
            'Mean Error (Nm)': f'{np.mean([np.mean(np.abs(tau_error[:, j])) for j in range(6)]):.6e}',
            'Relative Error (%)': f'{np.mean([rmse_list[j]/np.mean(np.abs(tau_original[:, j]))*100 if np.mean(np.abs(tau_original[:, j])) > 1e-6 else 0 for j in range(6)]):.2f}'
        })
        
        df = pd.DataFrame(summary_data)
        summary_path = 'results/model_comparison_open_loop_summary.csv'
        df.to_csv(summary_path, index=False)
        print(f"  ✓ 保存摘要: {summary_path}")
        
    except ImportError:
        print("  ⚠  pandas未安装,跳过CSV保存")
    
    print("\n" + "="*60)
    print("✓ 对比完成！")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())