#!/usr/bin/env python3
"""
生成校准后的MuJoCo模型（包含fullinertia）

从估计参数生成新的XML文件，完整设置所有10个参数：
- fullinertia="Ixx Ixy Ixz Iyy Iyz Izz" (6个)
- pos="x y z" (3个，重心位置)
- mass="m" (1个)
"""

import numpy as np
import pickle
import sys
import os
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def create_calibrated_xml(estimation_pkl_path, original_xml_path, output_xml_path, method='PC-OLS-REG'):
    """创建校准后的XML模型"""
    print("生成校准后的MuJoCo模型")
    
    # 1. 加载估计参数
    print("\n[1/4] 加载估计参数...")
    with open(estimation_pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    if method == 'PC-OLS-REG':
        sol = results['sol_pc_reg']
    elif method == 'PC-OLS':
        sol = results['sol_pc_ols']
    else:
        sol = results['sol_ols']
    
    if sol.get('pi_s') is None:
        print("  ⚠️  警告: 没有pi_s，无法生成完整模型")
        return None
    
    pi_s = sol['pi_s']
    pi_fr = sol.get('pi_fr', None)
    
    n_links = len(pi_s) // 10
    print(f"  ✓ 加载了 {len(pi_s)} 个标准参数 ({n_links} 个link)")
    if pi_fr is not None:
        print(f"  ✓ 加载了 {len(pi_fr)} 个摩擦参数 ({len(pi_fr)//3} 个关节)")
    
    # 2. 解析原始XML
    print("\n[2/4] 解析原始XML...")
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    bodies = root.findall('.//body')
    
    # 根据pi_s的长度动态确定link数量
    n_links_in_params = len(pi_s) // 10
    link_names = [f'link{i+1}' for i in range(n_links_in_params)]
    
    print(f"  找到 {len(bodies)} 个body元素")
    print(f"  将更新 {len(link_names)} 个link的参数")
    
    # 3. 更新每个link的参数
    print("\n[3/4] 更新link参数...")
    
    updated_count = 0
    for i, link_name in enumerate(link_names):
        param_idx = i * 10
        
        if param_idx + 9 >= len(pi_s):
            print(f"  ⚠️  参数不足，跳过 {link_name}")
            continue
        
        # ============================================================
        # 步骤1: 读取估计参数（link frame）
        # ============================================================
        # pi_s中存储的是 I_Origin（相对于link frame origin的6个惯性参数）
        # 这是因为parse_urdf和参数估计算法都基于link frame
        Ixx_link, Ixy_link, Ixz_link = pi_s[param_idx:param_idx + 3]
        Iyy_link, Iyz_link = pi_s[param_idx + 3:param_idx + 5]
        Izz_link = pi_s[param_idx + 5]
        
        # 一阶矩 h = m * r_com 和质量
        mx, my, mz = pi_s[param_idx + 6:param_idx + 9]
        mass = pi_s[param_idx + 9]
        
        # ============================================================
        # 步骤2: 计算COM位置
        # ============================================================
        # r_com = h / m
        com = np.array([mx, my, mz]) / mass if mass > 1e-6 else np.zeros(3)
        
        # ============================================================
        # 步骤3: 平行轴定理转换（link frame → COM frame）
        # ============================================================
        # MuJoCo需要的是相对于COM的惯性张量 I_COM
        # 转换公式：I_COM = I_link - m * [r_com]×^T * [r_com]×
        # 其中 [r_com]× 是r_com的斜对称矩阵
        
        def vec2skew(v):
            """向量转斜对称矩阵：[v]× = [[0, -vz, vy], [vz, 0, -vx], [-vy, vx, 0]]"""
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
        
        # 构建I_link矩阵（link frame）
        I_link = np.array([
            [Ixx_link, Ixy_link, Ixz_link],
            [Ixy_link, Iyy_link, Iyz_link],
            [Ixz_link, Iyz_link, Izz_link]
        ])
        
        # 应用平行轴定理
        com_skew = vec2skew(com)
        I_com = I_link - mass * (com_skew.T @ com_skew)
        
        # 提取I_COM的6个参数（COM frame）
        Ixx = I_com[0, 0]
        Ixy = I_com[0, 1]
        Ixz = I_com[0, 2]
        Iyy = I_com[1, 1]
        Iyz = I_com[1, 2]
        Izz = I_com[2, 2]
        
        # 检查转换后的 I_COM（这是MuJoCo实际使用的）
        eig_vals = np.linalg.eigvalsh(I_com)
        
        # 正定性检查
        if np.any(eig_vals <= 0):
            print(f"  ⚠️  {link_name} 惯性矩阵不正定！特征值: {eig_vals}")
            print(f"      MuJoCo编译会失败，跳过此link")
            continue
        
        # 三角不等式检查（MuJoCo要求）
        margin1 = (Ixx + Iyy) - Izz
        margin2 = (Ixx + Izz) - Iyy
        margin3 = (Iyy + Izz) - Ixx
        min_margin = min(margin1, margin2, margin3)
        
        if min_margin < -1e-8:
            print(f"  ⚠️  {link_name} 违反三角不等式！余量: {min_margin:.4e}")
            print(f"      Ixx+Iyy-Izz={margin1:.4e}, Ixx+Izz-Iyy={margin2:.4e}, Iyy+Izz-Ixx={margin3:.4e}")
            continue
        
        # 查找对应的body元素
        body = None
        for b in bodies:
            if b.get('name') == link_name:
                body = b
                break
        
        if body is None:
            print(f"  ⚠️  找不到body: {link_name}")
            continue
        
        # 查找或创建inertial元素
        inertial = body.find('inertial')
        if inertial is None:
            inertial = ET.SubElement(body, 'inertial')
        
        # 设置质量
        inertial.set('mass', f'{mass:.10f}')
        
        # 设置重心位置
        inertial.set('pos', f'{com[0]:.10f} {com[1]:.10f} {com[2]:.10f}')
        
        # ✅ 修复: 转换为MuJoCo的fullinertia顺序
        # URDF顺序: Ixx, Ixy, Ixz, Iyy, Iyz, Izz
        # MuJoCo顺序: Ixx, Iyy, Izz, Ixy, Ixz, Iyz
        fullinertia_mujoco = f'{Ixx:.10e} {Iyy:.10e} {Izz:.10e} {Ixy:.10e} {Ixz:.10e} {Iyz:.10e}'
        inertial.set('fullinertia', fullinertia_mujoco)
        
        # 移除旧的diaginertia属性（如果存在）
        if 'diaginertia' in inertial.attrib:
            del inertial.attrib['diaginertia']
        
        print(f"  ✓ {link_name}:")
        print(f"      mass = {mass:.6f} kg")
        print(f"      COM = [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")
        print(f"      I_link (link frame): [{Ixx_link:.4e}, {Iyy_link:.4e}, {Izz_link:.4e}] (对角)")
        print(f"      I_COM (COM frame): [{Ixx:.4e}, {Iyy:.4e}, {Izz:.4e}] (对角)")
        print(f"      MuJoCo fullinertia: [{Ixx:.4e}, {Iyy:.4e}, {Izz:.4e}, {Ixy:.4e}, {Ixz:.4e}, {Iyz:.4e}]")
        print(f"      特征值: [{eig_vals[0]:.4e}, {eig_vals[1]:.4e}, {eig_vals[2]:.4e}] ✓ 正定")
        print(f"      三角余量: {min_margin:.4e} ✓")
        
        updated_count += 1
    
    # 4. 更新摩擦参数
    updated_friction_count = 0
    
    if pi_fr is not None:
        print("\n[4/4] 更新joint摩擦参数...")
        
        joints = root.findall('.//joint')
        
        # 根据pi_fr的长度动态确定关节数量 (每个关节2个参数)
        n_joints_in_params = len(pi_fr) // 2
        joint_names = [f'joint{i+1}' for i in range(n_joints_in_params)]
        print(f"  将更新 {len(joint_names)} 个joint的摩擦参数")
        
        for i, joint_name in enumerate(joint_names):
            friction_idx = i * 2
            
            if friction_idx + 1 >= len(pi_fr):
                print(f"  ⚠️  摩擦参数不足，跳过 {joint_name}")
                continue
            
            # 提取2个摩擦参数
            Fv = pi_fr[friction_idx]       # 粘性摩擦
            Fc = pi_fr[friction_idx + 1]   # 库伦摩擦
            
            # 查找对应的joint元素
            joint = None
            for j in joints:
                if j.get('name') == joint_name:
                    joint = j
                    break
            
            if joint is None:
                print(f"  ⚠️  找不到joint: {joint_name}")
                continue
            
            # 设置摩擦参数
            joint.set('damping', f'{max(0, Fv):.10e}')  # 确保非负
            joint.set('frictionloss', f'{max(0, Fc):.10e}')  # 确保非负
            
            print(f"  ✓ {joint_name}:")
            print(f"      damping (Fv) = {Fv:.6e}")
            print(f"      frictionloss (Fc) = {Fc:.6e}")
            
            updated_friction_count += 1
    
    # 保存新XML
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    
    print("✅ 校准模型已保存")
    print(f"  文件: {output_xml_path}")
    print(f"  更新了 {updated_count} 个link的惯性参数")
    if pi_fr is not None:
        print(f"  更新了 {updated_friction_count} 个joint的摩擦参数")
    print("="*70)
    
    return output_xml_path
def main():
    
    # 配置
    estimation_pkl = "results/estimation_results.pkl"
    original_xml = "models/mjcf/manipulator/airbot_play_force/_play_force.xml"
    output_xml = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml"
    
    try:
        create_calibrated_xml(estimation_pkl, original_xml, output_xml, method='PC-OLS-REG')
        
        print("下一步操作：")
        print(f"\n1. 查看生成的模型:")
        print(f"   cat {output_xml}")
        
        print(f"\n2. 在validate_simulation.py中使用新模型:")
        print(f"   xml_path = '{output_xml}'")
        
        print(f"\n3. 验证新模型:")
        print(f"   python scripts/validate_simulation.py")
        
        print("\n" + "="*70)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

