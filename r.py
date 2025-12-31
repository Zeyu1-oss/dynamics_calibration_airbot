import scipy.io
import numpy as np

# 加載文件
file_path = 'models/exciting_trajectory/ptrnSrch_N7T25motorQR.mat'
traj_data = scipy.io.loadmat(file_path, struct_as_record=False, squeeze_me=True)

# 1. 打印傅立葉正弦係數 a (通常是 6x7)
print("="*30)
print("正弦係數 (a) - 激勵頻率分量:")
print(traj_data['a'])
print(f"形狀: {traj_data['a'].shape}")

# 2. 打印傅立葉餘弦係數 b (通常是 6x7)
print("\n" + "="*30)
print("餘弦係數 (b) - 激勵頻率分量:")
print(traj_data['b'])
print(f"形狀: {traj_data['b'].shape}")

# 3. 打印邊界平滑多項式係數 c_pol (通常是 6x6)
# 這些係數決定了軌跡在 T=0 和 T=End 時的速度/加速度是否為 0
print("\n" + "="*30)
print("多項式係數 (c_pol) - 用於起始/終點平滑:")
print(traj_data['c_pol'])
print(f"形狀: {traj_data['c_pol'].shape}")

# 4. 打印軌跡元數據 (traj_par)
print("\n" + "="*30)
print("軌跡配置參數 (traj_par):")
tp = traj_data['traj_par']
print(f"週期 T: {tp.T} s")
print(f"諧波數量 N: {tp.N}")
print(f"基頻 wf: {tp.wf:.4f} rad/s")
print(f"初始位置 q0: {tp.q0}")
print(f"最大速度限制 qd_max: {tp.qd_max}")
print(f"最大加速度限制 q2d_max: {tp.q2d_max}")

# 5. 打印辨識矩陣信息 (如果包含在 baseQR 中)
if 'baseQR' in traj_data:
    print("\n" + "="*30)
    print("基礎參數矩陣 (baseQR):")
    bqr = traj_data['baseQR']
    print(f"基礎參數個數: {bqr.numberOfBaseParameters}")