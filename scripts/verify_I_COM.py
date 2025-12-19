#!/usr/bin/env python3
"""
验证估计参数转换为I_COM后是否正定
"""

import numpy as np
import pickle
import sys

def vec2skew(v):
    """将向量转换为斜对称矩阵"""
    return np.array([
        [0, -v[2], v[1]],
        [v[2], 0, -v[0]],
        [-v[1], v[0], 0]
    ])

def check_I_COM(pi_s, method_name):
    """检查转换后的I_COM是否正定"""
    print(f"\n{'='*70}")
    print(f"检查方法: {method_name}")
    print(f"{'='*70}")
    
    n_links = len(pi_s) // 10
    all_pass = True
    
    for link_idx in range(n_links):
        param_idx = link_idx * 10
        
        # 提取参数
        Ixx_origin = pi_s[param_idx]
        Ixy_origin = pi_s[param_idx + 1]
        Ixz_origin = pi_s[param_idx + 2]
        Iyy_origin = pi_s[param_idx + 3]
        Iyz_origin = pi_s[param_idx + 4]
        Izz_origin = pi_s[param_idx + 5]
        
        mx = pi_s[param_idx + 6]
        my = pi_s[param_idx + 7]
        mz = pi_s[param_idx + 8]
        mass = pi_s[param_idx + 9]
        
        # 计算COM
        com = np.array([mx, my, mz]) / mass if mass > 1e-6 else np.zeros(3)
        
        # 构建I_Origin
        I_origin = np.array([
            [Ixx_origin, Ixy_origin, Ixz_origin],
            [Ixy_origin, Iyy_origin, Iyz_origin],
            [Ixz_origin, Iyz_origin, Izz_origin]
        ])
        
        # 应用平行轴定理
        com_skew = vec2skew(com)
        I_com = I_origin - mass * (com_skew.T @ com_skew)
        
        # 检查正定性
        eig_vals = np.linalg.eigvalsh(I_com)
        is_positive_definite = np.all(eig_vals > 0)
        
        # 检查三角不等式
        Ixx_com = I_com[0, 0]
        Iyy_com = I_com[1, 1]
        Izz_com = I_com[2, 2]
        
        margin1 = Ixx_com + Iyy_com - Izz_com
        margin2 = Ixx_com + Izz_com - Iyy_com
        margin3 = Iyy_com + Izz_com - Ixx_com
        min_margin = min(margin1, margin2, margin3)
        triangle_pass = min_margin > 1e-5
        
        status = "✓" if (is_positive_definite and triangle_pass) else "✗"
        all_pass = all_pass and is_positive_definite and triangle_pass
        
        print(f"\n  {status} link{link_idx+1}:")
        print(f"      mass = {mass:.6f} kg")
        print(f"      COM = [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")
        print(f"      I_Origin对角: [{Ixx_origin:.4e}, {Iyy_origin:.4e}, {Izz_origin:.4e}]")
        print(f"      I_COM对角:    [{Ixx_com:.4e}, {Iyy_com:.4e}, {Izz_com:.4e}]")
        print(f"      特征值: [{eig_vals[0]:.4e}, {eig_vals[1]:.4e}, {eig_vals[2]:.4e}] {'✓' if is_positive_definite else '✗'}")
        print(f"      三角余量: {min_margin:.4e} {'✓' if triangle_pass else '✗'}")
        
        if not is_positive_definite:
            print(f"      ⚠️  I_COM不正定！")
        if not triangle_pass:
            print(f"      ⚠️  三角不等式不满足！")
    
    return all_pass

def main():
    pkl_path = 'results/estimation_results.pkl'
    
    print("="*70)
    print("验证平行轴转换后的I_COM正定性")
    print("="*70)
    print(f"\n加载: {pkl_path}")
    
    try:
        with open(pkl_path, 'rb') as f:
            results = pickle.load(f)
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        sys.exit(1)
    
    print(f"✓ 成功加载估计结果")
    
    # 检查所有方法
    methods = {
        'OLS': results.get('sol_ols'),
        'PC-OLS': results.get('sol_pc_ols'),
        'PC-OLS-REG': results.get('sol_pc_reg')
    }
    
    summary = {}
    for method_name, sol in methods.items():
        if sol is None:
            print(f"\n⚠️  {method_name}: 无数据")
            continue
        
        pi_s = sol['pi_s']
        if pi_s is None or len(pi_s) == 0:
            print(f"\n⚠️  {method_name}: pi_s为空")
            continue
        
        all_pass = check_I_COM(pi_s, method_name)
        summary[method_name] = "✓ 全部通过" if all_pass else "✗ 有问题"
    
    # 总结
    print(f"\n{'='*70}")
    print("总结")
    print(f"{'='*70}")
    for method_name, status in summary.items():
        print(f"  {method_name}: {status}")
    
    print(f"\n{'='*70}")
    print("结论:")
    print("  如果有 ✗，说明当前估计结果是用旧版parameter_estimation.py生成的，")
    print("  没有I_COM约束。需要重新运行parameter_estimation.py。")
    print(f"{'='*70}")

if __name__ == '__main__':
    main()

