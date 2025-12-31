#!/usr/bin/env python3
"""
生成包含电机反射惯量的校准模型
修改自 create_calibrated_xml.py
"""

import numpy as np
import pickle
import sys
import os
import xml.etree.ElementTree as ET

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def create_calibrated_xml_with_motor(estimation_pkl_path, original_xml_path, output_xml_path, method='PC-OLS-REG'):
    """创建包含电机动力学的校准XML"""
    print("生成校准后的MuJoCo模型（包含电机反射惯量）")
    
    # 1. 加载估计参数
    print("\n[1/4] 加载估计参数...")
    with open(estimation_pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    if 'sol' in results:
        sol = results['sol']
        saved_method = results.get('method', 'unknown')
        include_motor = results.get('include_motor_dynamics', False)
        
        print(f"  ✓ 加载的估计方法: {saved_method}")
        print(f"  ✓ 包含电机动力学: {include_motor}")
        
        if method and method != saved_method:
            print(f"  ⚠️  警告: 请求的方法({method})与保存的方法({saved_method})不一致")
    else:
        print(f"  ⚠️  旧版格式，假设不包含电机动力学")
        sol = results
        include_motor = False
    
    if sol.get('pi_s') is None:
        print("  ⚠️  警告: 没有pi_s，无法生成完整模型")
        return None
    
    pi_s = sol['pi_s']
    
    # 检查摩擦/电机参数
    if 'pi_fr_motor' in sol:
        pi_fr_motor = sol['pi_fr_motor']
        n_fr_motor_params = len(pi_fr_motor)
        print(f"  ✓ 加载了 {n_fr_motor_params} 个摩擦+电机参数")
        
        if n_fr_motor_params == 18:
            print(f"      → 6关节 × 3参数 (Fv, Fc, I_motor)")
            has_motor = True
        elif n_fr_motor_params == 12:
            print(f"      → 6关节 × 2参数 (Fv, Fc) 无电机")
            has_motor = False
        else:
            print(f"  ⚠️  未知参数结构: {n_fr_motor_params}")
            has_motor = False
    elif 'pi_fr' in sol:
        pi_fr_motor = sol['pi_fr']
        n_fr_motor_params = len(pi_fr_motor)
        print(f"  ✓ 加载了 {n_fr_motor_params} 个摩擦参数")
        has_motor = False
    else:
        print("  ⚠️  没有摩擦/电机参数")
        pi_fr_motor = None
        has_motor = False
    
    n_links = len(pi_s) // 10
    print(f"  ✓ 加载了 {len(pi_s)} 个标准参数 ({n_links} 个link)")
    
    # 2. 解析原始XML
    print("\n[2/4] 解析原始XML...")
    tree = ET.parse(original_xml_path)
    root = tree.getroot()
    bodies = root.findall('.//body')
    
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
        Ixx_origin, Ixy_origin, Ixz_origin = pi_s[param_idx:param_idx + 3]
        Iyy_origin, Iyz_origin = pi_s[param_idx + 3:param_idx + 5]
        Izz_origin = pi_s[param_idx + 5]
        
        mx, my, mz = pi_s[param_idx + 6:param_idx + 9]
        mass = pi_s[param_idx + 9]
        
        # 计算COM位置
        com = np.array([mx, my, mz]) / mass if mass > 1e-6 else np.zeros(3)
        
        # 平行轴定理：I_Origin → I_COM
        def vec2skew(v):
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
        
        # ✅ 正确的转换（有 .T）
        I_com = I_origin + mass * (com_skew @ com_skew)   
        
        # 提取转换后的惯性参数
        Ixx = I_com[0, 0]
        Ixy = I_com[0, 1]
        Ixz = I_com[0, 2]
        Iyy = I_com[1, 1]
        Iyz = I_com[1, 2]
        Izz = I_com[2, 2]
        
        # 检查物理约束
        eig_vals = np.linalg.eigvalsh(I_com)
        
        if np.any(eig_vals <= 0):
            print(f"  ⚠️  {link_name} 惯性矩阵不正定！特征值: {eig_vals}") 
            continue
        
        margin1 = (Ixx + Iyy) - Izz
        margin2 = (Ixx + Izz) - Iyy
        margin3 = (Iyy + Izz) - Ixx
        min_margin = min(margin1, margin2, margin3)
        
        if min_margin < -1e-8:
            print(f"  ⚠️  {link_name} 违反三角不等式！余量: {min_margin:.4e}")
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
        
        # MuJoCo顺序: Ixx, Iyy, Izz, Ixy, Ixz, Iyz
        fullinertia_mujoco = f'{Ixx:.10e} {Iyy:.10e} {Izz:.10e} {Ixy:.10e} {Ixz:.10e} {Iyz:.10e}'
        inertial.set('fullinertia', fullinertia_mujoco)
        
        # 移除旧的diaginertia属性
        if 'diaginertia' in inertial.attrib:
            del inertial.attrib['diaginertia']
        
        print(f"  ✓ {link_name}:")
        print(f"      mass = {mass:.6f} kg")
        print(f"      COM = [{com[0]:.6f}, {com[1]:.6f}, {com[2]:.6f}]")
        print(f"      I_COM: [{Ixx:.4e}, {Iyy:.4e}, {Izz:.4e}] (对角)")
        print(f"      特征值: [{eig_vals[0]:.4e}, {eig_vals[1]:.4e}, {eig_vals[2]:.4e}] ✓")
        
        updated_count += 1
    
    # 4. 更新摩擦和电机参数
    updated_friction_count = 0
    updated_motor_count = 0
    
    if pi_fr_motor is not None:
        print("\n[4/4] 更新joint摩擦和电机参数...")
        
        joints = root.findall('.//joint')
        n_joints_to_process = 6
        joint_names = [f'joint{i+1}' for i in range(n_joints_to_process)]
        
        if has_motor:
            print(f"  将更新 {n_joints_to_process} 个joint (Fv, Fc, armature)")
            params_per_joint = 3
        else:
            print(f"  将更新 {n_joints_to_process} 个joint (Fv, Fc)")
            params_per_joint = 2
        
        for i, joint_name in enumerate(joint_names):
            friction_idx = i * params_per_joint
            
            if friction_idx + params_per_joint - 1 >= len(pi_fr_motor):
                print(f"  ⚠️  参数不足，跳过 {joint_name}")
                continue
            
            # 提取参数
            Fv = pi_fr_motor[friction_idx]       # 粘性摩擦
            Fc = pi_fr_motor[friction_idx + 1]   # 库伦摩擦
            
            if has_motor:
                I_motor = pi_fr_motor[friction_idx + 2]  # 电机反射惯量
            else:
                I_motor = None
            
            # 查找对应的joint元素
            joint = None
            for j in joints:
                if j.get('name') == joint_name:
                    joint = j
                    break
            
            if joint is None:
                print(f"  ⚠️  XML中找不到joint: {joint_name}")
                continue
            
            # 设置 MuJoCo 摩擦属性（强制非负）
            joint.set('damping', f'{max(0, Fv):.10e}')
            joint.set('frictionloss', f'{max(0, Fc):.10e}')
            
            # 设置电机反射惯量（如果有）
            if I_motor is not None and I_motor > 0:
                joint.set('armature', f'{I_motor:.10e}')
                print(f"  ✓ {joint_name}: damping={Fv:.4e}, frictionloss={Fc:.4e}, armature={I_motor:.4e} kg·m²")
                updated_motor_count += 1
            else:
                print(f"  ✓ {joint_name}: damping={Fv:.4e}, frictionloss={Fc:.4e}")
            
            updated_friction_count += 1
    
    # 保存新XML
    tree.write(output_xml_path, encoding='utf-8', xml_declaration=True)
    
    print("\n" + "="*60)
    print("✓ 校准模型生成完成！")
    print("="*60)
    print(f"输出文件: {output_xml_path}")
    print(f"更新了 {updated_count}/{len(link_names)} 个link")
    print(f"更新了 {updated_friction_count}/{len(joint_names)} 个joint的摩擦参数")
    if has_motor:
        print(f"更新了 {updated_motor_count}/{len(joint_names)} 个joint的电机反射惯量")
    print("="*60)
    
    return output_xml_path


def main():
    print("生成包含电机动力学的校准模型")
    
    estimation_pkl = "results/estimation_results_with_motor.pkl"
    original_xml = "models/mjcf/manipulator/airbot_play_force/_play_force.xml"
    output_xml = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml"
    
    if not os.path.exists(estimation_pkl):
        print(f"\n❌ 估计结果文件不存在: {estimation_pkl}")
        print("   请先运行: python parameter_estimation_with_motor.py")
        return 1
    
    if not os.path.exists(original_xml):
        print(f"\n❌ 原始XML文件不存在: {original_xml}")
        return 1
    
    try:
        create_calibrated_xml_with_motor(
            estimation_pkl, 
            original_xml, 
            output_xml, 
            method='PC-OLS-REG'
        )
        
        
    except Exception as e:
        import traceback
        print(f"\n❌ 生成失败:")
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())