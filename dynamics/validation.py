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
                           pi_s=None, plot=True, save_csv=False, output_prefix='validation'):
    """
    1. 基参数验证（pi_b）: tau = Wb @ [pi_b; pi_fr]
    2. 完整参数验证（pi_s）: tau = Y_std @ pi_s + Y_friction @ pi_fr
    
    Args:
        path_to_data: 验证数据路径 (CSV文件)
        idx: [start, end] 数据范围
        drv_gains: 驱动增益 (6,)
        baseQR: QR分解结果字典
        pi_b: 基础参数
        pi_fr: 摩擦参数
        pi_s: 标准参数（可选，如果提供则进行双重验证）
        plot: 是否绘制对比图
        save_csv: 是否保存详细结果到CSV
        output_prefix: 输出文件前缀
    
    Returns:
        rre: (6,) 每个关节的相对残差误差 (%)
        results: 字典，包含详细结果
    """
    
    print("开始验证动力学参数...")
    
    # 导入数据处理函数
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from dynamics.parameter_estimation import (
        parse_ur_data, 
        filter_data,
        build_observation_matrices  
    )
    
    # 1. 加载和处理验证数据
    print(f"\n 1/3: 加载验证数据...")
    
    vldtn_traj = parse_ur_data(path_to_data, idx[0], idx[1])
    vldtn_traj = filter_data(vldtn_traj)
    
    n_samples = len(vldtn_traj['t'])
    print(f"  ✓ 加载了 {n_samples} 个验证样本")
    
    # 2. 构建观测矩阵（批量，高效！）
    print(f"\n 2/3: 构建观测矩阵（批量处理）...")
    
    Tau_measured, Wb = build_observation_matrices(vldtn_traj, baseQR, drv_gains)
    
    print(f"  ✓ 观测矩阵构建完成: {Wb.shape}")
    
    # 3. 预测力矩（一次矩阵乘法，超快！）
    print(f"\n 3/3: 预测力矩...")
    
    # 方法1: 基参数验证
    params_base = np.concatenate([pi_b, pi_fr])
    Tau_predicted_base = Wb @ params_base
    
    print(f"  ✓ 基参数预测: tau = Wb @ [pi_b; pi_fr]")
    
    # 方法2: 完整参数验证（如果提供了pi_s）
    if pi_s is not None:
        print(f"  计算完整参数预测: tau = Y_std @ pi_s + Y_friction @ pi_fr")
        
        # Y_std @ E1 = W_dyn，所以 Y_std = W_dyn @ inv(E1)
        
        n_base = baseQR['numberOfBaseParameters']
        W_dyn = Wb[:, :n_base]  # 动力学部分（已映射到基参数）
        Y_friction = Wb[:, n_base:]  # 摩擦部分（18列）
        
        # 从基参数空间恢复到标准参数空间
        E1 = baseQR['permutationMatrixFull'][:, :n_base]
        
        # Y_std @ E1 = W_dyn，
        
        # 替代方案：直接重新调用standard_regressor
        print("    重新构建标准回归矩阵Y_std...")
        from dynamics.parameter_estimation import build_observation_matrices
        
        # 为了简单，暂时使用E的逆映射：pi_s = E @ [pi_b; pi_d]
        try:
            from oct2py import Oct2Py
            oc = Oct2Py()
            oc.addpath('matlab')
            oc.addpath('autogen')
            
            n_samples_total = vldtn_traj['q'].shape[0]
            
            try:
                print(f"    尝试批量计算Y_std（{n_samples_total}个样本）...")
                Y_std_total = oc.standard_regressor_airbot_batched(
                    vldtn_traj['q'],
                    vldtn_traj['qd_fltrd'],
                    vldtn_traj['q2d_est']
                )
                print(f"    ✓ 批量计算完成！")
            except:
                print(f"    批量函数不可用，逐个计算（这会花几分钟）...")
                Y_std_list = []
                
                for i in range(n_samples_total):
                    if (i + 1) % 500 == 0:
                        print(f"      进度: {i+1}/{n_samples_total} ({100*(i+1)/n_samples_total:.1f}%)")
                    
                    q_col = vldtn_traj['q'][i, :].reshape(-1, 1)
                    qd_col = vldtn_traj['qd_fltrd'][i, :].reshape(-1, 1)
                    qdd_col = vldtn_traj['q2d_est'][i, :].reshape(-1, 1)
                    
                    Yi = oc.standard_regressor_airbot(q_col, qd_col, qdd_col)
                    Y_std_list.append(Yi)
                
                Y_std_total = np.vstack(Y_std_list)
                print(f"    ✓ 逐个计算完成")
            
            oc.exit()
            
            print(f"    ✓ 标准回归矩阵: {Y_std_total.shape}")
            
            Tau_predicted_std = Y_std_total @ pi_s + Y_friction @ pi_fr
            
            print(f"    ✓ 完整参数预测完成")
            
        except Exception as e:
            print(f"    ⚠️ 无法构建Y_std: {e}")
            print(f"    跳过完整参数验证（只使用基参数验证）")
            Tau_predicted_std = None
    else:
        Tau_predicted_std = None
    
    tau_msrd = Tau_measured.reshape(-1, 6).T  # (6, n_samples)
    tau_pred = Tau_predicted_base.reshape(-1, 6).T  # (6, n_samples) - 基参数预测
    
    if Tau_predicted_std is not None:
        tau_pred_std = Tau_predicted_std.reshape(-1, 6).T  # (6, n_samples) - 完整参数预测
    else:
        tau_pred_std = None
    
    print(f"  ✓ 力矩预测完成")
    
    # 4. 计算相对残差误差
    
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
    
    # 打印基参数验证结果
    print("\n【方法1;基参数验证】 tau = Wb @ [pi_b; pi_fr]")
    print("\n关节 |  RRE(%)  |  RMSE(Nm)  |  MAE(Nm)  | Max误差(Nm)")
    for j in range(6):
        print(f"Joint{j+1} | {rre[j]:7.3f}  | {rmse[j]:9.4f}  | {mae[j]:8.4f}  | {max_error[j]:10.4f}")
    print(f"平均   | {np.mean(rre):7.3f}  | {np.mean(rmse):9.4f}  | {np.mean(mae):8.4f}  | {np.mean(max_error):10.4f}")
    
    # 完整参数验证（如果提供了pi_s）
    if tau_pred_std is not None:
        rre_std = np.zeros(6)
        rmse_std = np.zeros(6)
        mae_std = np.zeros(6)
        max_error_std = np.zeros(6)
        
        for j in range(6):
            residual_std = tau_msrd[j, :] - tau_pred_std[j, :]
            rre_std[j] = 100 * np.linalg.norm(residual_std) / np.linalg.norm(tau_msrd[j, :])
            rmse_std[j] = np.sqrt(np.mean(residual_std**2))
            mae_std[j] = np.mean(np.abs(residual_std))
            max_error_std[j] = np.max(np.abs(residual_std))
        
        print("\n【方法2:完整参数验证】 tau = Y_std @ pi_s + Y_friction @ pi_fr")
        print("\n关节 |  RRE(%)  |  RMSE(Nm)  |  MAE(Nm)  | Max误差(Nm)")
        for j in range(6):
            print(f"Joint{j+1} | {rre_std[j]:7.3f}  | {rmse_std[j]:9.4f}  | {mae_std[j]:8.4f}  | {max_error_std[j]:10.4f}")
        print(f"平均   | {np.mean(rre_std):7.3f}  | {np.mean(rmse_std):9.4f}  | {np.mean(mae_std):8.4f}  | {np.mean(max_error_std):10.4f}")
        
        # 对比两种方法
        print("\n【两种方法对比】")
        print("关节 | 基参数RRE | 完整参数RRE | 差异")
        for j in range(6):
            diff = abs(rre[j] - rre_std[j])
            print(f"Joint{j+1} | {rre[j]:10.3f}% | {rre_std[j]:12.3f}% | {diff:6.3f}%")
        print("-"*50)
        avg_diff = np.mean([abs(rre[j] - rre_std[j]) for j in range(6)])
        print(f"平均   | {np.mean(rre):10.3f}% | {np.mean(rre_std):12.3f}% | {avg_diff:6.3f}%")
        
        if avg_diff < 0.1:
            print("\n两种方法结果几乎完全一致！pi_b和pi_s映射正确！")
        elif avg_diff < 1.0:
            print("\n✓ 两种方法结果接近，参数估计良好")
        else:
            print("\n 两种方法结果有差异，可能存在映射问题")
    else:
        rre_std = None
        rmse_std = None
        mae_std = None
        max_error_std = None
        tau_pred_std = None
    
    # 5. 绘图
    if plot:
        
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        for j in range(6):
            ax = axes[j]
            ax.plot(vldtn_traj['t'], tau_msrd[j, :], 'b-', 
                   linewidth=1.5, label='Measured', alpha=0.8)
            ax.plot(vldtn_traj['t'], tau_pred[j, :], 'r--', 
                   linewidth=1.2, label='Predicted (pi_b)', alpha=0.8)
            if tau_pred_std is not None:
                ax.plot(vldtn_traj['t'], tau_pred_std[j, :], 'orange', 
                       linestyle=':', linewidth=1.5, label='Predicted (pi_s)', alpha=0.8)
            ax2 = ax.twinx()
            residual = tau_msrd[j, :] - tau_pred[j, :]
            ax2.plot(vldtn_traj['t'], residual, 'g-', 
                    linewidth=0.8, alpha=0.5, label='Residual (base)')
            ax2.set_ylabel('Residual (Nm)', color='g', fontsize=9)
            ax2.tick_params(axis='y', labelcolor='g')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.set_ylabel('Torque (Nm)', fontsize=10)
            title_str = f'Joint {j+1} - Base: RRE={rre[j]:.2f}%, RMSE={rmse[j]:.3f}Nm'
            if tau_pred_std is not None:
                title_str += f'\nStd: RRE={rre_std[j]:.2f}%, RMSE={rmse_std[j]:.3f}Nm'
            ax.set_title(title_str, fontsize=10, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left', fontsize=8)
            
        plt.tight_layout()
        
        os.makedirs('diagram', exist_ok=True)
        output_path = f'diagram/{output_prefix}_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ 保存对比图到: {output_path}")
        plt.close()  # 关闭图表，不显示（避免卡住）
    
    if save_csv:
        try:
            import pandas as pd
            df = pd.DataFrame({
                'time': vldtn_traj['t']
            })
            for j in range(6):
                df[f'tau_measured_j{j+1}'] = tau_msrd[j, :]
                df[f'tau_predicted_j{j+1}'] = tau_pred[j, :]
                df[f'residual_j{j+1}'] = tau_msrd[j, :] - tau_pred[j, :]
            os.makedirs('results', exist_ok=True)
            csv_path = f'results/{output_prefix}_detailed.csv'
            df.to_csv(csv_path, index=False)
            print(f"  ✓ 保存详细结果到: {csv_path}")
            summary_df = pd.DataFrame({
                'Joint': [f'Joint{j+1}' for j in range(6)] + ['Average'],
                'RRE (%)': list(rre) + [np.mean(rre)],
                'RMSE (Nm)': list(rmse) + [np.mean(rmse)],
                'MAE (Nm)': list(mae) + [np.mean(mae)],
                'Max Error (Nm)': list(max_error) + [np.mean(max_error)]
            })
            
            summary_path = f'results/{output_prefix}_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"  ✓ 保存统计摘要到: {summary_path}")
        except ImportError:
            print("  ⚠️  pandas未安装，跳过CSV保存")
    
    # 整理返回结果
    results = {
        'rre_base': rre,
        'rmse_base': rmse,
        'mae_base': mae,
        'max_error_base': max_error,
        'tau_measured': tau_msrd,
        'tau_predicted_base': tau_pred,
        'time': vldtn_traj['t']
    }
    
    # 如果有完整参数验证结果，也保存
    if tau_pred_std is not None:
        results['rre_std'] = rre_std
        results['rmse_std'] = rmse_std
        results['mae_std'] = mae_std
        results['max_error_std'] = max_error_std
        results['tau_predicted_std'] = tau_pred_std
    
    
    return rre, results


def load_estimation_results(pkl_path='estimation_results.pkl'):
    """
    
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
    
    return results


def main():
    """
    主函数：加载估计结果并在验证数据上进行验证
    """
    import h5py
    
    
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
    pkl_path = 'results/estimation_results.pkl'
    
    try:
        estimation_results = load_estimation_results(pkl_path)
    except Exception as e:
        print(f"  ❌ 加载估计结果失败: {e}")
        print("  请先运行 parameter_estimation.py 进行参数估计")
        sys.exit(1)
    
    # 3. 设置验证参数
    print("\n步骤 3/4: 设置验证参数...")
    drv_gains = np.ones(6)
    validation_data_path = 'results/data_csv/vali.csv'  # 使用相同或不同的数据集进行验证
    idx = [0, 2500]  # 可以使用不同的数据范围
    
    print(f"  验证数据: {validation_data_path}")
    print(f"  数据范围: {idx}")
    
    # 4. 对每种方法进行验证
    print("\n步骤 4/4: 验证估计参数...")
    
    validation_results = {}
    
    # 验证OLS方法
    if 'sol_ols' in estimation_results and estimation_results['sol_ols'] is not None:
        print("验证方法 1: OLS")
        
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
            'rre_base': rre_ols,
            'results': results_ols
        }
    
    # 验证PC-OLS方法
    if 'sol_pc_ols' in estimation_results and estimation_results['sol_pc_ols'] is not None:
        print("验证方法 2: PC-OLS")
        
        sol_pc_ols = estimation_results['sol_pc_ols']
        
        rre_pc_ols, results_pc_ols = validate_dynamic_params(
            path_to_data=validation_data_path,
            idx=idx,
            drv_gains=drv_gains,
            baseQR=baseQR,
            pi_b=sol_pc_ols['pi_b'],
            pi_fr=sol_pc_ols['pi_fr'],
            pi_s=sol_pc_ols.get('pi_s'),
            plot=True,
            save_csv=True,
            output_prefix='validation_PC-OLS'
        )
        
        validation_results['PC-OLS'] = {
            'rre_base': rre_pc_ols,
            'results': results_pc_ols
        }
        
        if results_pc_ols.get('rre_std') is not None:
            validation_results['PC-OLS']['rre_std'] = results_pc_ols['rre_std']
    
    # 验证PC-OLS-REG方法（双重验证：基参数 + 完整参数）
    if 'sol_pc_reg' in estimation_results and estimation_results['sol_pc_reg'] is not None:
        print("验证方法 3: PC-OLS-REG (双重验证)")
        
        sol_pc_reg = estimation_results['sol_pc_reg']
        
        # 传入pi_s进行双重验证
        rre_pc_reg, results_pc_reg = validate_dynamic_params(
            path_to_data=validation_data_path,
            idx=idx,
            drv_gains=drv_gains,
            baseQR=baseQR,
            pi_b=sol_pc_reg['pi_b'],
            pi_fr=sol_pc_reg['pi_fr'],
            pi_s=sol_pc_reg.get('pi_s'),  # 传入完整参数
            plot=True,
            save_csv=True,
            output_prefix='validation_PC-OLS-REG'
        )
        
        validation_results['PC-OLS-REG'] = {
            'rre_base': rre_pc_reg,  # 基参数验证的RRE
            'results': results_pc_reg
        }
        
        # 如果有完整参数验证结果，也记录
        if results_pc_reg.get('rre_std') is not None:
            validation_results['PC-OLS-REG']['rre_std'] = results_pc_reg['rre_std']
    
    # 5. 对比不同方法的验证结果
    if len(validation_results) > 1:
        print("方法对比")
        
        print("\n平均相对残差误差 (RRE %):")
        print("  方法           | 平均RRE")
        for method, result in validation_results.items():
            avg_rre = np.mean(result['rre_base'])
            print(f"  {method:14s} | {avg_rre:7.3f}%")
        
        # 绘制对比图
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = list(validation_results.keys())
        n_methods = len(methods)
        n_joints = 6
        
        x = np.arange(n_joints)
        width = 0.8 / n_methods
        
        for i, method in enumerate(methods):
            rre = validation_results[method]['rre_base']  # 使用基参数的RRE
            offset = (i - n_methods/2 + 0.5) * width
            ax.bar(x + offset, rre, width, label=f'{method} (base)', alpha=0.8)
        
        ax.set_xlabel('Joint', fontsize=12)
        ax.set_ylabel('RRE (%)', fontsize=12)
        ax.set_title('Validation: Relative Residual Error by Method', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([f'J{i+1}' for i in range(n_joints)])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        os.makedirs('diagram', exist_ok=True)
        output_path = 'diagram/validation_comparison_methods.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"\n  ✓ 保存方法对比图到: {output_path}")
        plt.close()  # 关闭图表，不显示（避免卡住）
    
    # 保存验证结果
    os.makedirs('results', exist_ok=True)
    result_path = 'results/validation_results.pkl'
    with open(result_path, 'wb') as f:
        pickle.dump(validation_results, f)
    
    print(f"✅ 所有验证完成！结果已保存到 {result_path}")
    
    return validation_results


if __name__ == "__main__":
    try:
        import pandas as pd
    except ImportError:
        print("❌ 错误: 需要安装 Pandas 库: pip install pandas")
        sys.exit(1)
    
    main()