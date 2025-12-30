import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

df_raw = pd.read_csv('trajectory_log.csv')
df_proc = pd.read_csv('results/data_csv/trajectory_processed.csv', header=None)

time_raw = df_raw['time'].values
q_raw = df_raw[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].values
qd_raw = df_raw[['qv1', 'qv2', 'qv3', 'qv4', 'qv5', 'qv6']].values
tau_raw = df_raw[['tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6']].values

time_proc = df_proc.iloc[:, 0].values
q_proc = df_proc.iloc[:, 1:7].values
qd_proc = df_proc.iloc[:, 7:13].values
q2d_proc = df_proc.iloc[:, 13:19].values
tau_proc = df_proc.iloc[:, 19:25].values

fig, axes = plt.subplots(6, 4, figsize=(20, 18))

for j in range(6):
    axes[j, 0].plot(time_raw, q_raw[:, j], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[j, 0].plot(time_proc, q_proc[:, j], 'r-', linewidth=1.5, label='Filtered')
    axes[j, 0].set_ylabel(f'J{j+1} q (rad)')
    axes[j, 0].legend(fontsize=8)
    axes[j, 0].grid(True, alpha=0.3)
    if j == 0:
        axes[j, 0].set_title('Position', fontweight='bold')
    if j == 5:
        axes[j, 0].set_xlabel('Time (s)')
    
    axes[j, 1].plot(time_raw, qd_raw[:, j], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[j, 1].plot(time_proc, qd_proc[:, j], 'r-', linewidth=1.5, label='Filtered')
    axes[j, 1].set_ylabel(f'J{j+1} qd (rad/s)')
    axes[j, 1].legend(fontsize=8)
    axes[j, 1].grid(True, alpha=0.3)
    if j == 0:
        axes[j, 1].set_title('Velocity', fontweight='bold')
    if j == 5:
        axes[j, 1].set_xlabel('Time (s)')
    
    axes[j, 2].plot(time_proc, q2d_proc[:, j], 'g-', linewidth=1.2)
    axes[j, 2].set_ylabel(f'J{j+1} qdd (rad/s²)')
    axes[j, 2].grid(True, alpha=0.3)
    if j == 0:
        axes[j, 2].set_title('Acceleration', fontweight='bold')
    if j == 5:
        axes[j, 2].set_xlabel('Time (s)')
    
    axes[j, 3].plot(time_raw, tau_raw[:, j], 'b-', alpha=0.5, linewidth=0.8, label='Raw')
    axes[j, 3].plot(time_proc, tau_proc[:, j], 'r-', linewidth=1.5, label='Filtered')
    axes[j, 3].set_ylabel(f'J{j+1} τ (Nm)')
    axes[j, 3].legend(fontsize=8)
    axes[j, 3].grid(True, alpha=0.3)
    if j == 0:
        axes[j, 3].set_title('Torque', fontweight='bold')
    if j == 5:
        axes[j, 3].set_xlabel('Time (s)')

plt.tight_layout()
plt.savefig('diagram/processed_data_comparison.png', dpi=300, bbox_inches='tight')

