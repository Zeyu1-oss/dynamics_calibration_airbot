"""
动力学参数验证模块
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端，避免在无GUI环境下卡住
import matplotlib.pyplot as plt
import sys
import os
import pickle

def validate_dynamic_params(path_to_data, idx, drv_gains, baseQR, pi_b, pi_fr, 
                           plot=True, save_csv=False, output_prefix='validation'):
    """
    验证估计的动力学参数（高效批量版本）
    
    功能:
    - 加载验证数据并过滤
    - 批量构建观测矩阵（和parameter_estimation.py一样快）
    - 使用估计参数预测力矩（矩阵乘法）
    - 计算相对残差误差 (RRE)
    - 绘制对比图
    - 保存详细结果到CSV
    
    Args:
        path_to_data: 验证数据路径 (CSV文件)
        idx: [start, end] 数据范围
        drv_gains: 驱动增益 (6,)
        baseQR: QR分解结果字典
        pi_b: 基础参数
        pi_fr: 摩擦参数
        plot: 是否绘制对比图
        save_csv: 是否保存详细结果到CSV
        output_prefix: 输出文件前缀
    
    Returns:
        rre: (6,) 每个关节的相对残差误差 (%)
        results: 字典，包含详细结果
    """
    
    print("\n" + "="*60)
    print("开始验证动力学参数...")
    print("="*60)
    
    # 导入数据处理函数
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from dynamics.parameter_estimation import (
        parse_ur_data, 
        filter_data,
        build_observation_matrices  
    )
    
    # 1. 加载和处理验证数据
    print(f"\n步骤 1/3: 加载验证数据...")
    print(f"  数据文件: {path_to_data}")
    print(f"  数据范围: [{idx[0]}, {idx[1]}]")
    
    vldtn_traj = parse_ur_data(path_to_data, idx[0], idx[1])
    vldtn_traj = filter_data(vldtn_traj)
    
    n_samples = len(vldtn_traj['t'])
    print(f"  ✓ 加载了 {n_samples} 个验证样本")
    
    # 2. 构建观测矩阵（批量，高效！）
    print(f"\n步骤 2/3: 构建观测矩阵（批量处理）...")
    
    Tau_measured, Wb = build_observation_matrices(vldtn_traj, baseQR, drv_gains)
    
    print(f"  ✓ 观测矩阵构建完成: {Wb.shape}")
    
    # 3. 预测力矩（一次矩阵乘法，超快！）
    print(f"\n步骤 3/3: 预测力矩...")
    
    # 合并参数
    params = np.concatenate([pi_b, pi_fr])
    
    # 预测力矩（批量）
    Tau_predicted = Wb @ params
    
    # 重塑为 (6, n_samples) 方便分析
    tau_msrd = Tau_measured.reshape(-1, 6).T  # (6, n_samples)
    tau_pred = Tau_predicted.reshape(-1, 6).T  # (6, n_samples)
    
    print(f"  ✓ 力矩预测完成")
    
    # 4. 计算相对残差误差
    print(f"\n=== 验证结果 ===")
    
    rre = np.zeros(6)
    rmse = np.zeros(6)
    mae = np.zeros(6)
    max_error = np.zeros(6)
    
    for j in range(6):
        residual = tau_msrd[j, :] - tau_pred[j, :]
        rre[j] = 100 * np.linalg.norm(residual) / np.linalg.norm(tau_msrd[j, :])
        rmse[j] = np.sqrt(np.mean(residual**2))
        mae[j] = np.mean(np.abs(residual))
        max_error[j] = np.max(np.abs(residual))
    
    # 打印结果
    print("\n关节 |  RRE(%)  |  RMSE(Nm)  |  MAE(Nm)  | Max误差(Nm)")
    print("-"*60)
    for j in range(6):
        print(f"Joint{j+1} | {rre[j]:7.3f}  | {rmse[j]:9.4f}  | {mae[j]:8.4f}  | {max_error[j]:10.4f}")
    print("-"*60)
    print(f"平均   | {np.mean(rre):7.3f}  | {np.mean(rmse):9.4f}  | {np.mean(mae):8.4f}  | {np.mean(max_error):10.4f}")
    
    # 5. 绘图
    if plot:
        print(f"\n绘制对比图...")
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for j in range(6):
            ax = axes[j]
            
            # 绘制测量值和预测值
            ax.plot(vldtn_traj['t'], tau_msrd[j, :], 'b-', 
                   linewidth=1.5, label='Measured', alpha=0.8)
            ax.plot(vldtn_traj['t'], tau_pred[j, :], 'r--', 
                   linewidth=1.2, label='Predicted', alpha=0.8)
            
            # 绘制残差（右侧y轴）
            ax2 = ax.twinx()
            residual = tau_msrd[j, :] - tau_pred[j, :]
            ax2.plot(vldtn_traj['t'], residual, 'g-', 
                    linewidth=0.8, alpha=0.5, label='Residual')
            ax2.set_ylabel('Residual (Nm)', color='g', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='g')
            
            # 设置标题和标签
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Torque (Nm)', fontsize=10)
            ax.set_title(f'Joint {j+1} - RRE: {rre[j]:.2f}%, RMSE: {rmse[j]:.3f} Nm', 
                        fontsize=11, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=9)
            
        plt.tight_layout()
        plt.savefig(f'{output_prefix}_comparison.png', dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存对比图到: {output_prefix}_comparison.png")
        plt.close()  # 关闭图表，不显示（避免卡住）
    
    # 6. 保存详细结果到CSV
    if save_csv:
        try:
            import pandas as pd
            
            # 创建数据框
            df = pd.DataFrame({
                'time': vldtn_traj['t']
            })
            
            for j in range(6):
                df[f'tau_measured_j{j+1}'] = tau_msrd[j, :]
                df[f'tau_predicted_j{j+1}'] = tau_pred[j, :]
                df[f'residual_j{j+1}'] = tau_msrd[j, :] - tau_pred[j, :]
            
            csv_path = f'{output_prefix}_detailed.csv'
            df.to_csv(csv_path, index=False)
            print(f"  ✓ 保存详细结果到: {csv_path}")
            
            # 保存统计摘要
            summary_df = pd.DataFrame({
                'Joint': [f'Joint{j+1}' for j in range(6)] + ['Average'],
                'RRE (%)': list(rre) + [np.mean(rre)],
                'RMSE (Nm)': list(rmse) + [np.mean(rmse)],
                'MAE (Nm)': list(mae) + [np.mean(mae)],
                'Max Error (Nm)': list(max_error) + [np.mean(max_error)]
            })
            
            summary_path = f'{output_prefix}_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"  ✓ 保存统计摘要到: {summary_path}")
        except ImportError:
            print("  ⚠️  pandas未安装，跳过CSV保存")
    
    # 整理返回结果
    results = {
        'rre': rre,
        'rmse': rmse,
        'mae': mae,
        'max_error': max_error,
        'tau_measured': tau_msrd,
        'tau_predicted': tau_pred,
        'time': vldtn_traj['t']
    }
    
    print("\n" + "="*60)
    print("✅ 验证完成!")
    print("="*60)
    
    return rre, results


def load_estimation_results(pkl_path='estimation_results.pkl'):
    """
    
    从pkl文件加载估计结果
    
    Args:
        pkl_path: pkl文件路径
    
    Returns:
        results: 包含sol_ols和sol_pc的字典
    """
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"估计结果文件不存在: {pkl_path}")
    
    print(f"从 {pkl_path} 加载估计结果...")
    with open(pkl_path, 'rb') as f:
        results = pickle.load(f)
    
    print("✓ 加载成功")
    print(f"  包含方法: {list(results.keys())}")
    
    return results


def main():
    """
    主函数：加载估计结果并在验证数据上进行验证
    """
    import h5py
    
    print("="*70)
    print("动力学参数验证脚本")
    print("="*70)
    
    # 1. 加载baseQR
    print("\n步骤 1/4: 加载baseQR...")
    mat_filename = 'models/baseQR_standard.mat'
    
    try:
        with h5py.File(mat_filename, 'r') as f:
            baseQR_group = f['baseQR']
            bb = np.array(baseQR_group['numberOfBaseParameters']).flatten()[0]
            E_full = np.array(baseQR_group['permutationMatrix']).T
            beta = np.array(baseQR_group['beta']).T
            motorDynamicsIncluded = bool(np.array(baseQR_group['motorDynamicsIncluded']).flatten()[0])
            
            baseQR = {
                'numberOfBaseParameters': int(bb),
                'permutationMatrixFull': E_full,
                'beta': beta,
                'motorDynamicsIncluded': motorDynamicsIncluded
            }
        
        if baseQR['beta'].ndim == 1:
            baseQR['beta'] = baseQR['beta'].reshape(-1, 1)
        
        print(f"  ✓ 基础参数数量: {baseQR['numberOfBaseParameters']}")
    
    except Exception as e:
        print(f"  ❌ 加载baseQR失败: {e}")
        sys.exit(1)
    
    # 2. 加载估计结果
    print("\n步骤 2/4: 加载估计参数...")
    pkl_path = 'estimation_results.pkl'
    
    try:
        estimation_results = load_estimation_results(pkl_path)
    except Exception as e:
        print(f"  ❌ 加载估计结果失败: {e}")
        print("  请先运行 parameter_estimation.py 进行参数估计")
        sys.exit(1)
    
    # 3. 设置验证参数
    print("\n步骤 3/4: 设置验证参数...")
    drv_gains = np.ones(6)
    validation_data_path = 'vali.csv'  # 使用相同或不同的数据集进行验证
    idx = [0, 2500]  # 可以使用不同的数据范围
    
    print(f"  验证数据: {validation_data_path}")
    print(f"  数据范围: {idx}")
    
    # 4. 对每种方法进行验证
    print("\n步骤 4/4: 验证估计参数...")
    
    validation_results = {}
    
    # 验证OLS方法
    if 'sol_ols' in estimation_results and estimation_results['sol_ols'] is not None:
        print("\n" + "-"*70)
        print("验证方法 1: OLS")
        print("-"*70)
        
        sol_ols = estimation_results['sol_ols']
        
        rre_ols, results_ols = validate_dynamic_params(
            path_to_data=validation_data_path,
            idx=idx,
            drv_gains=drv_gains,
            baseQR=baseQR,
            pi_b=sol_ols['pi_b'],
            pi_fr=sol_ols['pi_fr'],
            plot=True,
            save_csv=True,
            output_prefix='validation_OLS'
        )
        
        validation_results['OLS'] = {
            'rre': rre_ols,
            'results': results_ols
        }
    
    # 验证PC-OLS-REG方法
    if 'sol_pc' in estimation_results and estimation_results['sol_pc'] is not None:
        print("\n" + "-"*70)
        print("验证方法 2: PC-OLS-REG")
        print("-"*70)
        
        sol_pc = estimation_results['sol_pc']
        
        rre_pc, results_pc = validate_dynamic_params(
            path_to_data=validation_data_path,
            idx=idx,
            drv_gains=drv_gains,
            baseQR=baseQR,
            pi_b=sol_pc['pi_b'],
            pi_fr=sol_pc['pi_fr'],
            plot=True,
            save_csv=True,
            output_prefix='validation_PC-OLS-REG'
        )
        
        validation_results['PC-OLS-REG'] = {
            'rre': rre_pc,
            'results': results_pc
        }
    
    # 5. 对比不同方法的验证结果
    if len(validation_results) > 1:
        print("\n" + "="*70)
        print("方法对比")
        print("="*70)
        
        print("\n平均相对残差误差 (RRE %):")
        print("  方法           | 平均RRE")
        print("  " + "-"*40)
        for method, result in validation_results.items():
            avg_rre = np.mean(result['rre'])
            print(f"  {method:14s} | {avg_rre:7.3f}%")
        
        # 绘制对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(validation_results.keys())
        n_methods = len(methods)
        n_joints = 6
        
        x = np.arange(n_joints)
        width = 0.8 / n_methods
        
        for i, method in enumerate(methods):
            rre = validation_results[method]['rre']
            offset = (i - n_methods/2 + 0.5) * width
            ax.bar(x + offset, rre, width, label=method, alpha=0.8)
        
        ax.set_xlabel('Joint', fontsize=12)
        ax.set_ylabel('RRE (%)', fontsize=12)
        ax.set_title('Validation: Relative Residual Error by Method', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'J{i+1}' for i in range(n_joints)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig('validation_comparison_methods.png', dpi=300, bbox_inches='tight')
        print("\n  ✓ 保存方法对比图到: validation_comparison_methods.png")
        plt.close()  # 关闭图表，不显示（避免卡住）
    
    # 保存验证结果
    with open('validation_results.pkl', 'wb') as f:
        pickle.dump(validation_results, f)
    
    print("\n" + "="*70)
    print("✅ 所有验证完成！结果已保存到 validation_results.pkl")
    print("="*70)
    
    return validation_results


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("❌ 错误: 需要安装 Pandas 库: pip install pandas")
        sys.exit(1)
    
    main()