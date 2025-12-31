#!/usr/bin/env python3


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
    
    if 'sol' in results:
        sol = results['sol']
        saved_method = results.get('method', 'unknown')
        print(f"  ✓ 加载的估计方法: {saved_method}")
        if method and method != saved_method:
            print(f"  ⚠️  警告: 请求的方法({method})与保存的方法({saved_method})不一致")
    else:
        if method == 'PC-OLS-REG':
            sol = results['sol_pc_reg']
        elif method == 'PC-OLS':
            sol = results['sol_pc_ols']
        else:
            sol = results['sol_ols']
        print(f"  ✓ 使用的方法: {method}")
    
    if sol.get('pi_s') is None:
        print("  ⚠️  警告: 没有pi_s，无法生成完整模型")
        return None
    
    pi_s = sol['pi_s']
    pi_fr = sol.get('pi_fr', None)
    
    n_links = len(pi_s) // 10
    print(f"  ✓ 加载了 {len(pi_s)} 个标准参数 ({n_links} 个link)")
    if pi_fr is not None:
        n_joints_fr = len(pi_fr) // 2 
        print(f"  ✓ 加载了 {len(pi_fr)} 个摩擦参数 ({n_joints_fr} 个关节 × 2参数/关节)")
    
    # 2. 解析原始XML
    print("\n[2/4] 解析原始XML...")
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    bodies = root.findall('.//body')
    
    # 强制指定 6 个连杆，确保完整性
    n_links_to_process = 6
    link_names = [f'link{i+1}' for i in range(n_links_to_process)]
    
    print(f"  找到 {len(bodies)} 个body元素")
    print(f"  准备更新 {n_links_to_process} 个link的参数 (link1-6)")
    
    # 3. 更新每个link的参数
    print("\n[3/4] 更新link参数...")
    
    updated_count = 0
    for i, link_name in enumerate(link_names):
        param_idx = i * 10
        
        if param_idx + 9 >= len(pi_s):
            print(f"  ⚠️  参数不足，跳过 {link_name}")
            continue
        
        # 提取10个参数
        # pi_s中存储的是 I_Origin（相对于link frame origin）
        # 因为parse_urdf做了转换：I_vec = I_COM - m*skew(r_com)^2
        Ixx_origin, Ixy_origin, Ixz_origin = pi_s[param_idx:param_idx + 3]
        Iyy_origin, Iyz_origin = pi_s[param_idx + 3:param_idx + 5]
        Izz_origin = pi_s[param_idx + 5]
        
        # 一阶矩和质量
        mx, my, mz = pi_s[param_idx + 6:param_idx + 9]
        mass = pi_s[param_idx + 9]
        
        # 计算COM位置
        com = np.array([mx, my, mz]) / mass if mass > 1e-6 else np.zeros(3)
        
        # ✅ 平行轴定理：将 I_Origin 转换回 I_COM（MuJoCo需要）
        # I_COM = I_Origin - m * skew(r_com)^T @ skew(r_com)
        def vec2skew(v):
            """将向量转换为斜对称矩阵"""
            return np.array([
                [0, -v[2], v[1]],
                [v[2], 0, -v[0]],
                [-v[1], v[0], 0]
            ])
        
        com_skew = vec2skew(com)
        I_origin = np.array([
            [Ixx_origin, Ixy_origin, Ixz_origin],
            [Ixy_origin, Iyy_origin, Iyz_origin],
            [Ixz_origin, Iyz_origin, Izz_origin]
        ])
        
        # I_origin = I_COM - m * [r]× @ [r]×  (parse_urdf.py)
        # I_com = I_origin + mass * (com_skew @ com_skew) 
        I_com = I_origin + mass * (com_skew @ com_skew)   
        # I_com = I_origin 
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
        print(f"      I_Origin (估计): [{Ixx_origin:.4e}, {Iyy_origin:.4e}, {Izz_origin:.4e}] (对角)")
        print(f"      I_COM (转换后): [{Ixx:.4e}, {Iyy:.4e}, {Izz:.4e}] (对角)")
        print(f"      MuJoCo fullinertia: [{Ixx:.4e}, {Iyy:.4e}, {Izz:.4e}, {Ixy:.4e}, {Ixz:.4e}, {Iyz:.4e}]")
        print(f"      特征值: [{eig_vals[0]:.4e}, {eig_vals[1]:.4e}, {eig_vals[2]:.4e}] ✓ 正定")
        print(f"      三角余量: {min_margin:.4e} ✓")
        
        updated_count += 1
    
    # 4. 更新摩擦参数
    updated_friction_count = 0
    
    if pi_fr is not None:
        print("\n[4/4] 更新joint摩擦参数...")
        
        joints = root.findall('.//joint')
        
        # 强制指定 6 个关节
        n_joints_to_process = 6
        joint_names = [f'joint{i+1}' for i in range(n_joints_to_process)]
        print(f"  将更新 {n_joints_to_process} 个joint的摩擦参数 (提取前2项)")
        
        for i, joint_name in enumerate(joint_names):
            # 每个关节在 pi_fr 中占 3 个位置 [Fv, Fc, F0]
            friction_idx = i * 2
            
            if friction_idx + 1 >= len(pi_fr):
                print(f"  ⚠️  参数索引 {friction_idx+1} 超出范围 (pi_fr 长度 {len(pi_fr)})，跳过 {joint_name}")
                continue
            
            # 提取前两个参数
            Fv = pi_fr[friction_idx]       # 第一个：粘性摩擦 (damping)
            Fc = pi_fr[friction_idx + 1]   # 第二个：库伦摩擦 (frictionloss)
            
            # 查找对应的joint元素
            joint = None
            for j in joints:
                if j.get('name') == joint_name:
                    joint = j
                    break
            
            if joint is None:
                print(f"  ⚠️  XML中找不到joint: {joint_name}")
                continue
            
            # 设置 MuJoCo 属性
            joint.set('damping', f'{max(0, Fv):.10e}')
            joint.set('frictionloss', f'{max(0, Fc):.10e}')
            
            print(f"  ✓ {joint_name}: damping={Fv:.4e}, frictionloss={Fc:.4e}")
            
            updated_friction_count += 1
    
    # 保存新XML
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    
    print("\n" + "="*60)
    print("✓ 校准模型生成完成！")
    print("="*60)
    print(f"输出文件: {output_xml_path}")
    print(f"更新了 {updated_count}/{len(link_names)} 个link")
    if pi_fr is not None:
        print(f"更新了 {updated_friction_count}/{len(joint_names)} 个joint的摩擦参数")
    print("="*60)
    
    return output_xml_path
def main():
    
    # 配置
    estimation_pkl = "results/estimation_results.pkl"
    original_xml = "models/mjcf/manipulator/airbot_play_force/_play_force.xml"
    output_xml = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml"
    
    try:
        create_calibrated_xml(estimation_pkl, original_xml, output_xml, method='PC-OLS-REG')
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

