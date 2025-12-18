
import numpy as np
from scipy.linalg import qr
import sympy as sp

# 方法 1: 使用 Oct2Py 调用 MATLAB 函数 (最简单)

def base_params_qr_oct2py(includeMotorDynamics=False, matlab_dir='autogen'):
    # Returns:
    #     pi_lgr_base: 基础参数的符号表达式
    #     baseQR: 包含QR分解结果的字典
    print("使用QR分解计算基础参数 (Oct2Py版本)")
    np.random.seed(42) 
    try:
        from oct2py import Oct2Py
        oc = Oct2Py()
        oc.addpath(matlab_dir)
        print(f"✓ Oct2Py初始化成功，已添加路径: {matlab_dir}")
    except ImportError:
        raise ImportError("需要安装 oct2py: pip install oct2py")
    
    # 设置关节限制
    print("\n步骤 1/5: 设置关节限制")
    
    q_min = np.deg2rad([180, 100, -20, 100, 90, 100])
    q_max = np.deg2rad([120, 10, 180, 100, 90, 100])
    qd_max = np.array([2, 2, 2, 2, 2, 2])
    q2d_max = np.array([2, 2, 2, 2, 2, 2])
    
    print(f"  位置范围: [{np.rad2deg(q_min[0]):.1f}°, {np.rad2deg(q_max[0]):.1f}°] (关节1示例)")
    print(f"  速度上限: {qd_max[0]} rad/s")
    print(f"  加速度上限: {q2d_max[0]} rad/s²")
    
    # 定义符号参数
    print("\n步骤 2/5: 定义符号参数")
    
    # 每个连杆的标准动力学参数
    m = [sp.symbols(f'm{i}', real=True) for i in range(1, 7)]
    hx = [sp.symbols(f'h{i}_x', real=True) for i in range(1, 7)]
    hy = [sp.symbols(f'h{i}_y', real=True) for i in range(1, 7)]
    hz = [sp.symbols(f'h{i}_z', real=True) for i in range(1, 7)]
    ixx = [sp.symbols(f'i{i}_xx', real=True) for i in range(1, 7)]
    ixy = [sp.symbols(f'i{i}_xy', real=True) for i in range(1, 7)]
    ixz = [sp.symbols(f'i{i}_xz', real=True) for i in range(1, 7)]
    iyy = [sp.symbols(f'i{i}_yy', real=True) for i in range(1, 7)]
    iyz = [sp.symbols(f'i{i}_yz', real=True) for i in range(1, 7)]
    izz = [sp.symbols(f'i{i}_zz', real=True) for i in range(1, 7)]
    
    if includeMotorDynamics:
        im = [sp.symbols(f'im{i}', real=True) for i in range(1, 7)]
    
    # 构建参数向量
    pi_lgr_sym = []
    for i in range(6):
        if includeMotorDynamics:
            pi_lgr_sym.extend([
                ixx[i], ixy[i], ixz[i], iyy[i], iyz[i], izz[i],
                hx[i], hy[i], hz[i], m[i], im[i]
            ])
            nLnkPrms = 11
        else:
            pi_lgr_sym.extend([
                ixx[i], ixy[i], ixz[i], iyy[i], iyz[i], izz[i],
                hx[i], hy[i], hz[i], m[i]
            ])
            nLnkPrms = 10
    
    pi_lgr_sym = sp.Matrix(pi_lgr_sym)
    nParams = len(pi_lgr_sym)
    
    print(f"  每个连杆参数数量: {nLnkPrms}")
    print(f"  总参数数量: {nParams}")
    print(f"  包含电机动力学: {includeMotorDynamics}")
    
    # 生成观测矩阵 W
    print("\n步骤 3/5: 生成观测矩阵 (调用MATLAB函数)")
    
    W = []
    n_samples = 150
    
    print(f"  采样数量: {n_samples}")
    print("  采样策略:")
    print("    - 20% 完全随机")
    print("    - 20% 低速运动")
    print("    - 20% 高加速度")
    print("    - 20% 匀速运动")
    print("    - 20% 静止状态")
    
    for i in range(n_samples):
        if (i + 1) % 30 == 0:
            print(f"  进度: {i+1}/{n_samples}")
        
        # 多样化采样策略
        r_type = i % 5
        
        if r_type == 0:  # 完全随机
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = -qd_max + 2 * qd_max * np.random.rand(6)
            q2d_rnd = -q2d_max + 2 * q2d_max * np.random.rand(6)
            
        elif r_type == 1:  # 低速运动
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = 0.1 * (-qd_max + 2 * qd_max * np.random.rand(6))
            q2d_rnd = np.zeros(6)
            
        elif r_type == 2:  # 高加速度
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = np.zeros(6)
            q2d_rnd = -q2d_max + 2 * q2d_max * np.random.rand(6)
            
        elif r_type == 3:  # 匀速运动
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = 0.5 * qd_max * (2 * np.random.rand(6) - 1)
            q2d_rnd = np.zeros(6)
            
        else:  # 静止状态
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = np.zeros(6)
            q2d_rnd = np.zeros(6)
        
        # 调用MATLAB函数计算回归矩阵
        q_col = q_rnd.reshape(-1, 1)
        qd_col = qd_rnd.reshape(-1, 1)
        q2d_col = q2d_rnd.reshape(-1, 1)
        
        if includeMotorDynamics:
            Y = oc.regressorWithMotorDynamics(q_col, qd_col, q2d_col)
        else:
            Y = oc.standard_regressor_airbot(q_col, qd_col, q2d_col)
        
        W.append(Y)
    
    W = np.vstack(W)
    print(f"\n  观测矩阵 W 形状: {W.shape}")
    
    print("\n步骤 4/5: 执行QR分解")
    
    # QR分解: W*E = Q*R
    Q, R, E = qr(W, pivoting=True, mode='economic')
    
    # 计算秩
    tolerance = 1e-10
    bb = np.sum(np.abs(np.diag(R)) > tolerance)
    
    print(f"  矩阵秩: {bb}")
    print(f"  标准参数数量: {nParams}")
    print(f"  基础参数数量: {bb}")
    print(f"  减少参数: {nParams - bb} ({(nParams-bb)/nParams*100:.1f}%)")
    
    # 提取 R1 和 R2
    R1 = R[:bb, :bb]
    R2 = R[:bb, bb:]
    
    # 计算 beta: W2 = W1 * beta
    beta = np.linalg.solve(R1, R2)
    beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0  # 去除数值误差
    
    print(f"  Beta 矩阵形状: {beta.shape}")
    
    # 验证关系
    E_full = np.eye(nParams)[:, E]  # 转换为完整的置换矩阵
    W1 = W @ E_full[:, :bb]
    W2 = W @ E_full[:, bb:]
    error = np.linalg.norm(W2 - W1 @ beta)
    
    print(f"  验证误差: {error:.2e}")
    assert error < 1e-6, f"W2 = W1*beta 关系验证失败，误差: {error}"
    print("  ✓ 验证通过")
    
    # -----------------------------------------------------------------------
    # 计算基础参数
    # -----------------------------------------------------------------------
    print("\n步骤 5/5: 计算基础参数")
    
    # 构建置换矩阵 (符号版本)
    E_sym = sp.zeros(nParams, nParams)
    for i, col_idx in enumerate(E):
        E_sym[col_idx, i] = 1
    
    # 独立参数和依赖参数
    pi1 = E_sym[:, :bb].T @ pi_lgr_sym
    pi2 = E_sym[:, bb:].T @ pi_lgr_sym
    
    # 基础参数
    beta_sym = sp.Matrix(beta)
    pi_lgr_base = pi1 + beta_sym @ pi2
    
    print(f"  基础参数向量长度: {len(pi_lgr_base)}")
    print(f"  示例基础参数:")
    for i in range(min(3, len(pi_lgr_base))):
        print(f"    π_base[{i}] = {pi_lgr_base[i]}")
    
    baseQR = {
        'numberOfBaseParameters': bb,
        'permutationMatrix': E,
        'permutationMatrixFull': E_full,
        'beta': beta,
        'motorDynamicsIncluded': includeMotorDynamics,
        'standardParameters': pi_lgr_sym,
        'baseParameters': pi_lgr_base,
        'R1': R1,
        'R2': R2,
        'observationMatrix': W,
        'tolerance': tolerance
    }
    
    print("✅ QR分解完成!")
    print("="*70)
    print(f"结果摘要:")
    print(f"  - 标准参数: {nParams}")
    print(f"  - 基础参数: {bb}")
    print(f"  - 参数减少: {nParams - bb} ({(nParams-bb)/nParams*100:.1f}%)")
    print("="*70 + "\n")
    
    # 清理
    oc.exit()
    return pi_lgr_base, baseQR


# 方法 2: 纯Python版本 (使用Python生成的regressor)

def base_params_qr_python(includeMotorDynamics=False):
    print("使用QR分解计算基础参数 (纯Python版本)")
    
    try:
        import sys
        sys.path.insert(0, 'matlab')
        from standard_regressor_airbot import standard_regressor_airbot
        print("✓ 成功导入 Python 版本的 standard_regressor_airbot")
    except ImportError as e:
        raise ImportError(f"无法导入 standard_regressor_airbot: {e}\n"
                         "请先运行 generate_rb_regressor_complete.py 生成函数")
    
    # -----------------------------------------------------------------------
    # 设置关节限制
    # -----------------------------------------------------------------------
    print("\n步骤 1/5: 设置关节限制")
    
    q_min = np.deg2rad([180, 100, -20, 100, 90, 100])
    q_max = np.deg2rad([120, 10, 180, 100, 90, 100])
    qd_max = np.array([2, 2, 2, 2, 2, 2])
    q2d_max = np.array([2, 2, 2, 2, 2, 2])
    
    # -----------------------------------------------------------------------
    # 定义符号参数 (与上面相同)
    # -----------------------------------------------------------------------
    print("\n步骤 2/5: 定义符号参数")
    
    m = [sp.symbols(f'm{i}', real=True) for i in range(1, 7)]
    hx = [sp.symbols(f'h{i}_x', real=True) for i in range(1, 7)]
    hy = [sp.symbols(f'h{i}_y', real=True) for i in range(1, 7)]
    hz = [sp.symbols(f'h{i}_z', real=True) for i in range(1, 7)]
    ixx = [sp.symbols(f'i{i}_xx', real=True) for i in range(1, 7)]
    ixy = [sp.symbols(f'i{i}_xy', real=True) for i in range(1, 7)]
    ixz = [sp.symbols(f'i{i}_xz', real=True) for i in range(1, 7)]
    iyy = [sp.symbols(f'i{i}_yy', real=True) for i in range(1, 7)]
    iyz = [sp.symbols(f'i{i}_yz', real=True) for i in range(1, 7)]
    izz = [sp.symbols(f'i{i}_zz', real=True) for i in range(1, 7)]
    
    if includeMotorDynamics:
        im = [sp.symbols(f'im{i}', real=True) for i in range(1, 7)]
        nLnkPrms = 11
    else:
        nLnkPrms = 10
    
    pi_lgr_sym = []
    for i in range(6):
        if includeMotorDynamics:
            pi_lgr_sym.extend([
                ixx[i], ixy[i], ixz[i], iyy[i], iyz[i], izz[i],
                hx[i], hy[i], hz[i], m[i], im[i]
            ])
        else:
            pi_lgr_sym.extend([
                ixx[i], ixy[i], ixz[i], iyy[i], iyz[i], izz[i],
                hx[i], hy[i], hz[i], m[i]
            ])
    
    pi_lgr_sym = sp.Matrix(pi_lgr_sym)
    nParams = len(pi_lgr_sym)
    
    print(f"  总参数数量: {nParams}")
    
    # -----------------------------------------------------------------------
    # 生成观测矩阵 W (使用Python函数)
    # -----------------------------------------------------------------------
    print("\n步骤 3/5: 生成观测矩阵 (使用Python函数)")
    
    W = []
    n_samples = 150
    
    for i in range(n_samples):
        if (i + 1) % 30 == 0:
            print(f"  进度: {i+1}/{n_samples}")
        
        r_type = i % 5
        
        if r_type == 0:
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = -qd_max + 2 * qd_max * np.random.rand(6)
            q2d_rnd = -q2d_max + 2 * q2d_max * np.random.rand(6)
        elif r_type == 1:
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = 0.1 * (-qd_max + 2 * qd_max * np.random.rand(6))
            q2d_rnd = np.zeros(6)
        elif r_type == 2:
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = np.zeros(6)
            q2d_rnd = -q2d_max + 2 * q2d_max * np.random.rand(6)
        elif r_type == 3:
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = 0.5 * qd_max * (2 * np.random.rand(6) - 1)
            q2d_rnd = np.zeros(6)
        else:
            q_rnd = q_min + (q_max - q_min) * np.random.rand(6)
            qd_rnd = np.zeros(6)
            q2d_rnd = np.zeros(6)
        
        # 调用Python函数
        Y = standard_regressor_airbot(*q_rnd, *qd_rnd, *q2d_rnd)
        W.append(Y)
    
    W = np.vstack(W)
    print(f"\n  观测矩阵 W 形状: {W.shape}")
    
    # -----------------------------------------------------------------------
    # QR分解 (与上面相同的逻辑)
    # -----------------------------------------------------------------------
    print("\n步骤 4/5: 执行QR分解")
    
    Q, R, E = qr(W, pivoting=True, mode='economic')
    tolerance = 1e-10
    bb = np.sum(np.abs(np.diag(R)) > tolerance)
    
    print(f"  基础参数数量: {bb}/{nParams}")
    
    R1 = R[:bb, :bb]
    R2 = R[:bb, bb:]
    beta = np.linalg.solve(R1, R2)
    beta[np.abs(beta) < np.sqrt(np.finfo(float).eps)] = 0
    
    # 验证
    E_full = np.eye(nParams)[:, E]
    W1 = W @ E_full[:, :bb]
    W2 = W @ E_full[:, bb:]
    error = np.linalg.norm(W2 - W1 @ beta)
    assert error < 1e-6, f"验证失败，误差: {error}"
    print(f"  ✓ 验证通过 (误差: {error:.2e})")
    
    # -----------------------------------------------------------------------
    # 计算基础参数
    # -----------------------------------------------------------------------
    print("\n步骤 5/5: 计算基础参数")
    
    E_sym = sp.zeros(nParams, nParams)
    for i, col_idx in enumerate(E):
        E_sym[col_idx, i] = 1
    
    pi1 = E_sym[:, :bb].T @ pi_lgr_sym
    pi2 = E_sym[:, bb:].T @ pi_lgr_sym
    beta_sym = sp.Matrix(beta)
    pi_lgr_base = pi1 + beta_sym @ pi2
    
    baseQR = {
        'numberOfBaseParameters': bb,
        'permutationMatrix': E,
        'permutationMatrixFull': E_full,
        'beta': beta,
        'motorDynamicsIncluded': includeMotorDynamics,
        'standardParameters': pi_lgr_sym,
        'baseParameters': pi_lgr_base,
        'R1': R1,
        'R2': R2,
        'observationMatrix': W,
        'tolerance': tolerance
    }
    
    print("\n" + "="*70)
    print("✅ QR分解完成!")
    print("="*70)
    
    return pi_lgr_base, baseQR


def verify_base_params(baseQR, n_tests=10):
    """
    验证基础参数的正确性
    随机生成测试数据，比较全参数和基础参数的力矩输出
    """
    print("\n" + "="*70)
    print("验证基础参数")
    print("="*70)
    
    pass



def main():
    
    use_oct2py = True # True: 使用Oct2Py, False: 使用纯Python
    
    if use_oct2py:
        pi_base, baseQR = base_params_qr_oct2py(
            includeMotorDynamics=False,
            matlab_dir='matlab'
        )
    else:
        pi_base, baseQR = base_params_qr_python(
            includeMotorDynamics=False
        )
    
    # 打印结果
    print("结果分析")
    print(f"标准参数数量: {len(baseQR['standardParameters'])}")
    print(f"基础参数数量: {baseQR['numberOfBaseParameters']}")
    print(f"参数减少: {len(baseQR['standardParameters']) - baseQR['numberOfBaseParameters']}")
    
    # 保存结果
    import pickle
    with open('base_params_qr_result.pkl', 'wb') as f:
        pickle.dump({'pi_base': pi_base, 'baseQR': baseQR}, f)
    print("\n✅ 结果已保存到: base_params_qr_result.pkl")


if __name__ == "__main__":
    main()