"""
数据处理工具模块
用于解析和过滤机器人轨迹数据
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

def parse_ur_data(file_path, start_idx, end_idx):
    """
    解析UR机器人数据文件
    
    参数:
        file_path: CSV数据文件路径
        start_idx: 起始索引
        end_idx: 结束索引
    
    返回:
        traj: dict, 包含 't', 'q', 'qd', 'q2d', 'i' 等字段
    """
    # 读取CSV文件
    df = pd.read_csv(file_path)
    
    # 提取数据段
    df = df.iloc[start_idx:end_idx]
    
    traj = {}
    
    # 时间
    if 'time' in df.columns:
        traj['t'] = df['time'].values
    elif 't' in df.columns:
        traj['t'] = df['t'].values
    else:
        # 如果没有时间列，创建一个
        traj['t'] = np.arange(len(df)) * 0.002  # 假设500Hz
    
    # 关节角度 q (假设列名为 q1, q2, ..., q6)
    q_cols = [f'q{i}' for i in range(1, 7)]
    if all(col in df.columns for col in q_cols):
        traj['q'] = df[q_cols].values
    else:
        # 尝试其他可能的列名
        q_cols_alt = [f'actual_q_{i}' for i in range(6)]
        if all(col in df.columns for col in q_cols_alt):
            traj['q'] = df[q_cols_alt].values
        else:
            raise ValueError(f"未找到关节角度列，期望: {q_cols}")
    
    # 关节速度 qd
    qd_cols = [f'qd{i}' for i in range(1, 7)]
    if all(col in df.columns for col in qd_cols):
        traj['qd'] = df[qd_cols].values
    else:
        qd_cols_alt = [f'actual_qd_{i}' for i in range(6)]
        if all(col in df.columns for col in qd_cols_alt):
            traj['qd'] = df[qd_cols_alt].values
        else:
            # 如果没有速度数据，用数值微分计算
            print("警告: 未找到速度数据，使用数值微分计算")
            dt = np.mean(np.diff(traj['t']))
            traj['qd'] = np.gradient(traj['q'], dt, axis=0)
    
    # 关节加速度 q2d (通常需要数值微分或估计)
    q2d_cols = [f'q2d{i}' for i in range(1, 7)]
    if all(col in df.columns for col in q2d_cols):
        traj['q2d'] = df[q2d_cols].values
    else:
        # 使用数值微分计算加速度
        dt = np.mean(np.diff(traj['t']))
        if 'qd' in traj:
            traj['q2d'] = np.gradient(traj['qd'], dt, axis=0)
        else:
            traj['q2d'] = np.gradient(np.gradient(traj['q'], dt, axis=0), dt, axis=0)
    
    # 关节力矩/电流 i
    i_cols = [f'i{i}' for i in range(1, 7)]
    tau_cols = [f'tau{i}' for i in range(1, 7)]
    
    if all(col in df.columns for col in i_cols):
        traj['i'] = df[i_cols].values
    elif all(col in df.columns for col in tau_cols):
        traj['i'] = df[tau_cols].values
    else:
        # 尝试其他可能的列名
        i_cols_alt = [f'actual_current_{i}' for i in range(6)]
        tau_cols_alt = [f'target_moment_{i}' for i in range(6)]
        
        if all(col in df.columns for col in i_cols_alt):
            traj['i'] = df[i_cols_alt].values
        elif all(col in df.columns for col in tau_cols_alt):
            traj['i'] = df[tau_cols_alt].values
        else:
            raise ValueError(f"未找到力矩/电流列，期望: {i_cols} 或 {tau_cols}")
    
    return traj


def filter_data(traj, window_length=21, polyorder=3):
    """
    对轨迹数据进行滤波处理
    
    参数:
        traj: dict, 轨迹数据
        window_length: Savitzky-Golay滤波窗口长度
        polyorder: 多项式阶数
    
    返回:
        traj: 添加了滤波后字段的轨迹数据
    """
    # 确保window_length是奇数
    if window_length % 2 == 0:
        window_length += 1
    
    # 确保window_length不大于数据长度
    n_samples = len(traj['t'])
    if window_length > n_samples:
        window_length = n_samples if n_samples % 2 == 1 else n_samples - 1
        print(f"警告: 滤波窗口长度调整为 {window_length}")
    
    # 滤波速度
    if 'qd' in traj:
        traj['qd_fltrd'] = np.zeros_like(traj['qd'])
        for i in range(6):
            traj['qd_fltrd'][:, i] = savgol_filter(
                traj['qd'][:, i], 
                window_length, 
                polyorder
            )
    else:
        traj['qd_fltrd'] = traj['qd']
    
    # 估计加速度（从滤波后的速度）
    dt = np.mean(np.diff(traj['t']))
    traj['q2d_est'] = np.gradient(traj['qd_fltrd'], dt, axis=0)
    
    # 再次滤波加速度
    for i in range(6):
        traj['q2d_est'][:, i] = savgol_filter(
            traj['q2d_est'][:, i],
            window_length,
            polyorder
        )
    
    # 滤波力矩/电流
    if 'i' in traj:
        traj['i_fltrd'] = np.zeros_like(traj['i'])
        for i in range(6):
            traj['i_fltrd'][:, i] = savgol_filter(
                traj['i'][:, i],
                window_length,
                polyorder
            )
    else:
        traj['i_fltrd'] = traj['i']
    
    return traj


def downsample_data(traj, factor=10):
    """
    对轨迹数据进行降采样
    
    参数:
        traj: dict, 轨迹数据
        factor: 降采样因子
    
    返回:
        traj_down: 降采样后的轨迹数据
    """
    traj_down = {}
    
    for key, value in traj.items():
        if isinstance(value, np.ndarray):
            traj_down[key] = value[::factor]
        else:
            traj_down[key] = value
    
    return traj_down

