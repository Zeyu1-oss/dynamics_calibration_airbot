#!/usr/bin/env python3
"""
分析轨迹数据，绘制对比曲线，诊断抖动问题
"""

import sys
import csv
import numpy as np
import matplotlib.pyplot as plt


def load_csv_data(csv_file):
    """加载CSV数据"""
    data = {
        'time': [],
        'q': [],
        'qv': [],
        'tau': [],
        'q_des': [],
        'qv_des': []
    }
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        has_tau = 'tau1' in header
        
        for row in reader:
            required_cols = 31 if has_tau else 25
            if len(row) < required_cols:
                continue
            try:
                data['time'].append(float(row[0]))
                data['q'].append([float(row[i]) for i in range(1, 7)])
                data['qv'].append([float(row[i]) for i in range(7, 13)])
                if has_tau:
                    data['tau'].append([float(row[i]) for i in range(13, 19)])
                    data['q_des'].append([float(row[i]) for i in range(19, 25)])
                    data['qv_des'].append([float(row[i]) for i in range(25, 31)])
                else:
                    data['tau'].append([0.0] * 6)
                    data['q_des'].append([float(row[i]) for i in range(13, 19)])
                    data['qv_des'].append([float(row[i]) for i in range(19, 25)])
            except:
                continue
    
    return data


def plot_analysis(data):
    """绘制详细分析图"""
    time_array = np.array(data['time'])
    q_actual = np.array(data['q'])
    qv_actual = np.array(data['qv'])
    tau_actual = np.array(data['tau'])
    q_des = np.array(data['q_des'])
    qv_des = np.array(data['qv_des'])
    
    n_joints = 6
    fig = plt.figure(figsize=(24, 14))
    fig.suptitle('', fontsize=16, fontweight='bold')
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    for joint_idx in range(n_joints):
        # 位置对比
        ax_q = plt.subplot(n_joints, 5, joint_idx * 5 + 1)
        ax_q.plot(time_array, q_des[:, joint_idx], label='exp',
                  linestyle='-', linewidth=2, color=colors[joint_idx], alpha=0.7)
        ax_q.plot(time_array, q_actual[:, joint_idx], label='real',
                  linestyle='--', linewidth=1.5, color='red', alpha=0.8)
        ax_q.set_ylabel(f'关节{joint_idx+1}\n position (rad)', fontsize=9)
        ax_q.grid(True, alpha=0.3)
        ax_q.legend(fontsize=7)
        if joint_idx == 0:
            ax_q.set_title('traj tracking', fontweight='bold')
        
        # 位置误差
        ax_qe = plt.subplot(n_joints, 5, joint_idx * 5 + 2)
        q_error = q_actual[:, joint_idx] - q_des[:, joint_idx]
        ax_qe.plot(time_array, q_error, color='red', linewidth=1.5)
        ax_qe.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax_qe.set_ylabel(f'joint{joint_idx+1}\n error (rad)', fontsize=9)
        ax_qe.grid(True, alpha=0.3)
        rmse = np.sqrt(np.mean(q_error**2))
        max_err = np.max(np.abs(q_error))
        ax_qe.text(0.02, 0.98, f'RMSE: {rmse:.4f}\nMax: {max_err:.4f}',
                   transform=ax_qe.transAxes, fontsize=7, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))
        if joint_idx == 0:
            ax_qe.set_title('pos error', fontweight='bold')
        
        # 速度对比
        ax_qv = plt.subplot(n_joints, 5, joint_idx * 5 + 3)
        ax_qv.plot(time_array, qv_des[:, joint_idx], label='exp',
                   linestyle='-', linewidth=2, color=colors[joint_idx], alpha=0.7)
        ax_qv.plot(time_array, qv_actual[:, joint_idx], label='real',
                   linestyle='--', linewidth=1.5, color='red', alpha=0.8)
        ax_qv.set_ylabel(f'joint{joint_idx+1}\n v (rad/s)', fontsize=9)
        ax_qv.grid(True, alpha=0.3)
        ax_qv.legend(fontsize=7)
        if joint_idx == 0:
            ax_qv.set_title('v tracking', fontweight='bold')
        
        # 速度误差（抖动检测）
        ax_qve = plt.subplot(n_joints, 5, joint_idx * 5 + 4)
        qv_error = qv_actual[:, joint_idx] - qv_des[:, joint_idx]
        ax_qve.plot(time_array, qv_error, color='red', linewidth=1.5)
        ax_qve.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax_qve.set_ylabel(f'joint{joint_idx+1}\n v error (rad/s)', fontsize=9)
        ax_qve.grid(True, alpha=0.3)
        qv_rmse = np.sqrt(np.mean(qv_error**2))
        qv_std = np.std(qv_error)
        ax_qve.text(0.02, 0.98, f'RMSE: {qv_rmse:.4f}\nStd: {qv_std:.4f}',
                    transform=ax_qve.transAxes, fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.5))
        if joint_idx == 0:
            ax_qve.set_title('v error', fontweight='bold')
        
        # 力矩
        ax_tau = plt.subplot(n_joints, 5, joint_idx * 5 + 5)
        ax_tau.plot(time_array, tau_actual[:, joint_idx], color=colors[joint_idx], linewidth=1.5)
        ax_tau.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
        ax_tau.set_ylabel(f'joint{joint_idx+1}\n torque (Nm)', fontsize=9)
        ax_tau.grid(True, alpha=0.3)
        tau_mean = np.mean(tau_actual[:, joint_idx])
        tau_max = np.max(np.abs(tau_actual[:, joint_idx]))
        ax_tau.text(0.02, 0.98, f'Mean: {tau_mean:.4f}\nMax: {tau_max:.4f}',
                    transform=ax_tau.transAxes, fontsize=7, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
        if joint_idx == 0:
            ax_tau.set_title('torque', fontweight='bold')
        
        if joint_idx == n_joints - 1:
            ax_q.set_xlabel('time (s)', fontsize=9)
            ax_qe.set_xlabel('time (s)', fontsize=9)
            ax_qv.set_xlabel('time (s)', fontsize=9)
            ax_qve.set_xlabel('time (s)', fontsize=9)
            ax_tau.set_xlabel('time (s)', fontsize=9)
    
    plt.tight_layout()
    output_file = 'trajectory_analysis.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ 分析图已保存: {output_file}")
    
    # 打印诊断报告
    print("\n" + "="*60)
    print("抖动诊断报告")
    print("="*60)
    for joint_idx in range(n_joints):
        q_error = q_actual[:, joint_idx] - q_des[:, joint_idx]
        qv_error = qv_actual[:, joint_idx] - qv_des[:, joint_idx]
        qv_std = np.std(qv_error)
        
        print(f"\n关节 {joint_idx+1}:")
        print(f"  位置RMSE: {np.sqrt(np.mean(q_error**2)):.6f} rad")
        print(f"  速度RMSE: {np.sqrt(np.mean(qv_error**2)):.6f} rad/s")
        print(f"  速度误差标准差: {qv_std:.6f} rad/s")
        
        if qv_std > 0.1:
            print(f"  ⚠️  警告：速度误差波动大，可能存在抖动！")
        elif qv_std > 0.05:
            print(f"  ⚠️  注意：速度误差有一定波动")
        else:
            print(f"  ✓  速度跟踪良好")


def main():
    if len(sys.argv) < 2:
        print("用法: python3 plot_trajectory.py <trajectory_log.csv>")
        return 1
    
    csv_file = sys.argv[1]
    print(f"加载数据: {csv_file}")
    
    data = load_csv_data(csv_file)
    print(f"数据点数: {len(data['time'])}")
    
    if len(data['time']) == 0:
        print("错误：没有有效数据")
        return 1
    
    plot_analysis(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())

