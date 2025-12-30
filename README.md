# Robot Dynamic Parameter Calibration

机器人动力学参数校准工具包，用于从实验数据估计机器人动力学参数，并生成物理一致的MuJoCo仿真模型。

## 功能特性

- **多种参数估计方法**：OLS（普通最小二乘）、PC-OLS（物理一致性最小二乘）、PC-OLS-REG（带正则化的物理一致性）
- **物理一致性约束**：确保估计的参数满足物理定律（质量>0，惯性矩阵半正定，三角不等式）
- **摩擦模型**：支持粘性摩擦、库伦摩擦（Fv, Fc,）
- **MuJoCo模型生成**：自动生成校准后的MuJoCo XML模型
- **验证工具**：对比原始模型和校准模型的仿真结果

## 项目结构

```
calibration_airbot/
├── dynamics/              # 参数估计核心模块
│   ├── parameter_estimation.py    # 主估计函数
│   ├── validation.py             # 验证函数
│   ├── parameter_estimation_with_motordynamics.py    # 主估计函数(包含armature估计)
│   ├── validation.py             # 验证函数
│   └── base_params_qr.py         # 基础参数QR分解
├── scripts/              # 工具脚本
│   ├── test.py                   # 生成仿真数据
│   ├── create_calibrated_model.py # 生成校准模型
│   ├── create_calibrated_model.py # 生成校准模型（如果使用parameter_estimation_with_motordynamics.py会更新armature）
│   └── compare_two_models.py     # 模型对比
├── utils/                # 工具函数
│   ├── parse_urdf.py             # URDF解析
│   └── data_processing.py        # 数据处理
├── state_machine_demo/    # 真实机器人控制示例
│   ├── basic_usage.py            # 基础使用示例
│   ├── csv_pvt_control.py         # CSV轨迹跟踪（PVT控制）
│   ├── state_control.py           # 状态机控制器封装
│   ├── plot_trajectory.py         # 轨迹可视化
│   ├── real_data/                 # 真实机器人采集的数据
│   └── resources/                 # URDF和mesh资源
├── matlab/               # MATLAB函数（通过Oct2Py调用）
│   └── standard_regressor_airbot.m
├── models/               # 模型文件
│   ├── baseQR_standard.mat      # QR分解结果
│   ├── mjcf/                    # MuJoCo模型
│   └── ptrnSrch_N7T25QR-**.mat  # 激励轨迹参数（傅里叶级数+多项式参数，456不会有碰撞风险）
└── results/              # 结果输出
    ├── data_csv/                 # 处理后的数据
    ├── data/                     # 真实机器人数据（转换后）
    └── estimation_results.pkl    # 估计结果
```

## 安装依赖

```bash
# Python依赖
pip install numpy scipy cvxpy pandas matplotlib h5py

# Oct2Py（用于调用MATLAB函数）
pip install oct2py

# MuJoCo Python绑定
pip install mujoco

# 可选：MOSEK求解器（提高优化精度）
# 访问 https://www.mosek.com/products/academic-licenses/ 获取免费学术许可证
pip install Mosek

# 真实机器人控制（可选）
# 安装 airbot_state_machine 包
pip install airbot_state_machine-0.1.4.dev11+g996d249-py3-none-any.whl
# 或从源码安装
```

## 快速开始

### 1. 生成仿真数据

```bash
python scripts/test.py
```

生成的数据保存在 `results/data_csv/` 目录。

### 2. 参数估计

```bash
python dynamics/parameter_estimation.py
```

**配置选项**（在 `parameter_estimation.py` 的 `main()` 函数中）：

```python
METHOD = 'PC-OLS-REG'      # 可选: 'OLS', 'PC-OLS', 'PC-OLS-REG'
LAMBDA_REG = 5e-4          # 正则化系数（仅PC-OLS-REG）
data_paths = 'results/data_csv/vali——0fre.csv'
data_ranges = [0, 2500]
```

### 3. 生成校准模型

```bash
python scripts/create_calibrated_model.py
```

输出：`models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml`

### 4. 验证结果

```bash
python dynamics/validation.py
```

### 5. 对比模型

```bash
python scripts/compare_two_models.py
```

## 数据格式

### 输入数据格式（CSV）

```
time, q1, q2, q3, q4, q5, q6, qd1, qd2, qd3, qd4, qd5, qd6, i1, i2, i3, i4, i5, i6, ...
```

- `time`: 时间戳（秒）
- `q1-q6`: 关节位置（弧度）
- `qd1-qd6`: 关节速度（弧度/秒）
- `i1-i6`: 关节力矩/电流（Nm）

**注意**：加速度 `qdd` 会自动通过数值微分计算，无需提供。

### 输出数据格式

估计结果保存在 `results/estimation_results.pkl`：

```python
{
    'sol': {
        'pi_b': np.array,      # 基础参数
        'pi_fr': np.array,     # 摩擦参数
        'pi_s': np.array,      # 标准参数（60维）
        'masses': np.array      # 估计的质量
    },
    'method': 'PC-OLS-REG',
    'lambda_reg': 5e-4
}
```

## 参数估计方法

### OLS (Ordinary Least Squares)
- 无约束最小二乘
- 快速但不保证物理一致性

### PC-OLS (Physically Consistent OLS)
- 添加物理一致性约束（质量>0，惯性矩阵半正定）
- 保证参数物理合理

### PC-OLS-REG (PC-OLS with Regularization)
- PC-OLS + URDF参考参数正则化
- 在物理一致性和先验知识之间平衡

## 物理约束

### 质量约束
- 所有6个link的质量在URDF参考值的±20%范围内
- 最小质量：1g

### 惯性矩阵约束
- 正定性：所有特征值 > 0
- 三角不等式：Ixx + Iyy ≥ Izz（及对称形式）
- 通过4×4伪惯性矩阵D的半正定约束实现

### 摩擦参数约束
- Fv（粘性摩擦）> 0
- Fc（库伦摩擦）> 0
- armature> 0

## 配置说明

### 质量约束范围

在 `parameter_estimation.py` 中修改：

```python
error_range = 0.20  # ±20%误差范围
mass_urdf = np.array([0.607, 0.918, 0.7, 0.359, 0.403, 0.11])  # 6个link的参考质量
```


```

## 输出文件

- `results/estimation_results.pkl`: 估计参数
- `results/validation_results.pkl`: 验证结果
- `results/validation_*.csv`: 验证详细数据
- `diagram/validation_*.png`: 验证对比图
- `diagram/model_comparison_*.png`: 模型对比图

## 故障排除



