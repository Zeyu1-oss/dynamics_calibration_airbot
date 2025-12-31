# Robot Dynamic Parameter Calibration

机器人动力学参数校准工具包，用于从实验数据估计机器人动力学参数，并生成物理一致的MuJoCo仿真模型。


## 功能

- **多种参数估计方法**：OLS（普通最小二乘）、PC-OLS（物理一致性最小二乘）、PC-OLS-REG（带正则化的物理一致性）
- **物理一致性约束**：确保估计的参数满足物理定律（质量>0，惯性矩阵半正定，三角不等式）
- **摩擦模型**：支持粘性摩擦、库伦摩擦（Fv, Fc）和电机动力学（armature）
- **MuJoCo模型生成**：自动生成校准后的MuJoCo XML模型
- **验证工具**：驗證生成的仿真模型和真實機器人的動力學系統一致性
- **真实机器人控制**：支持从CSV轨迹文件执行PVT控制，采集实验数据

## 项目结构

```
calibration_airbot/
├── dynamics/              # 参数估计核心模块
│   ├── parameter_estimation.py              # 主估计函数（不含电机动力学）
│   ├── parameter_estimation_withmotordynamics.py  # 主估计函数（包含armature估计）
│   ├── validation.py                        # 验证函数
│   └── base_params_qr.py                    # 基础参数QR分解
├── scripts/              # 工具脚本
│   ├── test_simulation.py                   # 生成仿真數據
│   ├── create_calibrated_model.py          # 生成校准模型（不含电机动力学）
│   ├── create_calibrated_model_withmotor.py  # 生成校准模型（包含电机动力学）
│   ├── compare_two_models.py                # 仿真模型对比
├── utils/                # 工具函数
│   ├── parse_urdf.py                        # URDF解析
│   └── data_processing.py                   # 数据处理
│   └── read_pkl.py                 
├── state_machine_demo/    # 真实机器人控制示例
│   ├── basic_usage.py                       # 基础使用示例
│   ├── csv_pvt_control.py                   # CSV轨迹跟踪（PVT控制）
│   ├── csv_mit_control.py                   # CSV轨迹跟踪（MIT控制）
│   ├── state_control.py                     # 状态机控制器封装
│   ├── plot_trajectory.py                   # 轨迹可视化
│   ├── real_data/                           # 真实机器人采集的数据
│   ├── real_data_converted/                 # 转换后的数据
│   └── resources/                           # URDF和mesh资源
├── autogen/              
│   ├── M_mtrx_fcn.py                        # 质量矩阵函数
│   ├── C_mtrx_fcn.py                        # 科里奥利矩阵函数
│   ├── G_vctr_fcn.py                        # 重力向量函数
│   └── F_vctr_fcn.py                        # 摩擦向量函数
├── matlab/               # MATLAB函数（通过Oct2Py调用）
│   ├── standard_regressor_airbot.m          # 标准回归器
│   ├── standard_regressor_airbot_batched.m  # 批量标准回归器
│   ├── M_mtrx_fcn.m                         # 质量矩阵
│   ├── C_mtrx_fcn.m                         # 科里奥利矩阵
│   ├── G_vctr_fcn.m                         # 重力向量
│   └── frictionRegressor.m                  # 摩擦回归器
├── models/               # 模型文件
│   ├── baseQR_standard.mat                  # QR分解结果
│   ├── mjcf/                                # MuJoCo模型
│   │   └── manipulator/
│   │       └── airbot_play_force/
│   │           ├── _play_force.xml          # 原始模型
│   │           └── _play_force_calibrated.xml  # 校准模型
│   ├── urdf/                                # URDF模型
│   └── models/exciting_trajectory/ptrnSrch_N7T25QR-*.mat      # 激励轨迹参数（傅里叶级数+多项式）
└── results/              # 结果输出
    ├── data_csv/                            # 仿真測試保存的数据
    ├── estimation_results.pkl              # 估计结果s（不含电机）
    ├── estimation_results_with_motor.pkl   # 估计结果（含电机）
```

## 安装依赖


### Python依赖

```bash
# 基础科学计算库
pip install numpy scipy pandas matplotlib h5py

# 优化求解器
pip install cvxpy

# Oct2Py（用于调用MATLAB/Octave函数）
pip install oct2py

# MuJoCo Python绑定
pip install mujoco

```

### 真实机器人控制


```bash
# 安装 airbot_state_machine 包
pip install airbot_state_machine-0.1.4.dev11+g996d249-py3-none-any.whl

```

### 安装Octave（如果未安装）

```bash
# Ubuntu/Debian
sudo apt-get install octave

# 验证安装
octave --version
```

## 快速开始

### 真实机器人数据采集

使用 `state_machine_demo/csv_pvt_control.py` 从真实机器人采集数据，轨迹数据在results/data_csv/，建议都跑一边，辨识的时候用-5和-6
**输出格式**：
采集的数据保存在 `state_machine_demo/real_data/` 目录，包含：
- 实际关节位置、速度、力矩
- 期望关节位置、速度
- 时间戳生成的数据
- 生成的数据第一行需要删掉。

```bash
cd state_machine_demo
python csv_pvt_control.py
```
**参数说明**：
- `--csv`: 输入轨迹文件,先運行仿真scripts/test_simulation.py（CSV包含激勵軌跡的位置速度）
- `--can`: CAN接口名称（默认：can0）
- `--eef`: 末端执行器类型（默认：none）
- `--duration`: 执行时长（秒，默认：完整轨迹）
- `--output`: 输出文件路径（默认：自动生成）
- `--tau-filter`: 力矩滤波窗口大小（默认：1不滤波）
- `--no-read`: 仅发送控制命令，不读取反馈（用于测试）

由于电机tau=k*I的k参数不准确，需要多采集几组数据来估计gain，改argument里面的csv轨迹参数来获得多组数据。
运行cali_gains.py可以运行原始mjcf仿真和真实数据做最小二乘法得到近似的j2和j3的电流drive gains，对j2和j3数据处理后自动保存至results/unified_corrected_j2j3

![原始urdf和真实数据对比图（j2和j3差异较大）](diagram/image.png)

```bash
cd ..
python scripts/cali_gains.py
```




### 参数辨识

需要先采集真实数据，改679行-695行真实数据路径

```bash
python dynamics/parameter_estimation_withmotordynamics.py
```
- 结果保存至`results/estimation_results_with_motor.pkl`: 估计参数（含电机）


### 生成校准模型

根据估计结果生成校准后的MuJoCo airbot模型：

```bash

# 含电机动力学
python scripts/create_calibrated_model_withmotor.py


```
校准模型生成会：
1. 加载估计的标准参数 `pi_s`（60维，6个link × 10参数/link）
2. 解析原始MuJoCo XML模型
3. 更新每个link的质量和惯性矩阵
4. 更新摩擦参数
5. 更新电机参数

### validation
```bash
python scripts/vali_sim.py
```
![result](diagram/results.png)
#### 选择估计方法

**OLS (Ordinary Least Squares)**
- 优点：计算快速
- 缺点：不保证物理一致性
- 适用：快速测试、初步估计

**PC-OLS (Physically Consistent OLS)**
- 优点：保证物理一致性
- 缺点：可能偏离URDF参考值
- 适用：无先验知识的情况

**PC-OLS-REG (PC-OLS with Regularization)**
- 优点：物理一致性 + 参考先验
- 缺点：需要调整正则化系数
- 适用：**推荐方法**，有URDF参考模型时




### 验证流程

验证包括：
1. **力矩预测验证**：使用估计参数预测力矩，与实际力矩对比
3. **开环验证**：对比原始模型和校准模型的开环响应

## 数据格式

### 输入数据格式（CSV）

标准CSV格式，包含以下列：

```
time, q1, q2, q3, q4, q5, q6, qd1, qd2, qd3, qd4, qd5, qd6, tau1, tau2, tau3, tau4, tau5, tau6
```

- `time`: 时间戳（秒）
- `q1-q6`: 关节位置（弧度）
- `qd1-qd6`: 关节速度（弧度/秒）
- `tau1-tau6`: 关节力矩

**注意**：
- 加速度 `qdd` 会自动通过数值微分计算，无需提供
- 数据应包含足够的激励（建议使用激励轨迹）

### 输出数据格式

估计结果保存在 `results/estimation_results.pkl`：

```python
{
    'sol': {
        'pi_b': np.array,      # 基础参数（最小参数集）
        'pi_fr': np.array,     # 摩擦参数+电机（12维：6关节×3参数）
        'pi_s': np.array,      # 标准参数（60维：6link×10参数）
        'masses': np.array     # 估计的质量（6维）
    },
    'method': 'PC-OLS-REG',
    'lambda_reg': 5e-4,
    'stats': {
        'rmse': float,         # 均方根误差
        'condition_number': float,  # 条件数
        ...
    }
}
```

## 参数估计方法

### OLS (Ordinary Least Squares)

无约束最小二乘估计：

```
min ||Wb * pi_b - tau||²
```

- **优点**：计算快速，无约束
- **缺点**：不保证物理一致性，可能得到负质量等不合理参数
- **适用场景**：快速测试、初步估计

### PC-OLS (Physically Consistent OLS)

带物理约束的最小二乘：

```
min ||Wb * pi_b - tau||²
s.t. 物理约束（质量>0，惯性矩阵正定等）
```

- **优点**：保证物理一致性
- **缺点**：可能偏离URDF参考值较远
- **适用场景**：无先验知识，需要物理一致性保证

### PC-OLS-REG (PC-OLS with Regularization)

带正则化的物理一致性估计：

```
min ||Wb * pi_b - tau||² + λ * ||pi_s - pi_urdf||²
s.t. 物理约束
```

- **优点**：物理一致性 + 参考先验，通常效果最好
- **缺点**：需要调整正则化系数 λ
- **适用场景**：**推荐方法**，有URDF参考模型时


## 物理约束

### 质量约束

- 所有6个link的质量在URDF参考值的±10%范围内
- 最小质量：1g（防止数值问题）
- 可在 `parameter_estimation.py` 中修改 `error_range` 参数

### 惯性矩阵约束

通过4×4伪惯性矩阵D的半正定约束实现：

1. **正定性**：所有特征值 > 0
2. **三角不等式**：
   - Ixx + Iyy ≥ Izz
   - Ixx + Izz ≥ Iyy
   - Iyy + Izz ≥ Ixx

这些约束确保惯性矩阵物理合理。

### 摩擦参数约束

- Fv（粘性摩擦系数）> 0
- Fc（库伦摩擦系数）> 0
- armature（电机惯性）> 0（如果估计电机动力学）



### 校准模型

- `models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml`: 校准后的MuJoCo模型


