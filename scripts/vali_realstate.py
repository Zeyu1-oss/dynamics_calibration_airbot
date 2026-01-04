#!/usr/bin/env python3
import numpy as np
import mujoco
import scipy.signal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ptrnSrch_N7T25QR-7(1).csv"
# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_opt(1).csv"
# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ga_N12T25(1).csv"
CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ptrnSrch_N7T25QR-6(1).csv"
# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ptrnSrch_N8T25QR_maxq1.csv"
# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ptrnSrch_N8T25QR-8.csv"
# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ptrnSrch_N8T25QR.csv"
# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ptrnSrch_N7T25QR-5(1).csv"
# CSV_PATH = "results/unified_corrected_j2j3/unified_corrected_vali_ga_N12T25(1).csv"
MODEL_XML_PATH = "models/mjcf/manipulator/airbot_play_force/_play_force_calibrated.xml"
SAVE_DIR = "diagram"
os.makedirs(SAVE_DIR, exist_ok=True)

data = np.loadtxt(CSV_PATH, delimiter=',')

t = data[:, 0]
q = data[:, 1:7]
qdot = data[:, 7:13]
tau_real = data[:, 13:19]

dt = np.mean(np.diff(t))
n_steps, n_joints = q.shape

qddot = np.zeros_like(qdot)
for j in range(n_joints):
    qddot[:, j] = np.gradient(qdot[:, j], dt)

fs = 200.0
fc = 20.0  # cutoff frequency (Hz)

b, a = scipy.signal.butter(
    N=2,
    Wn=fc / (fs / 2),
    btype='low'
)
qddot = scipy.signal.filtfilt(b, a, qddot, axis=0)

# MUJOCO INVERSE DYNAMICS
model = mujoco.MjModel.from_xml_path(MODEL_XML_PATH)
data_mj = mujoco.MjData(model)

tau_sim = np.zeros_like(tau_real)

for k in range(n_steps):
    data_mj.qpos[:n_joints] = q[k]
    data_mj.qvel[:n_joints] = qdot[k]
    data_mj.qacc[:n_joints] = qddot[k]

    mujoco.mj_inverse(model, data_mj)
    tau_sim[k] = data_mj.qfrc_inverse[:n_joints]

# ========================
# PLOT
# ========================
fig, axes = plt.subplots(n_joints, 2, figsize=(14, 3 * n_joints))
fig.suptitle("Torque Validation (Real-State Inverse Dynamics)", fontsize=16)

for j in range(n_joints):
    # Torque comparison
    ax = axes[j, 0]
    ax.plot(t, tau_real[:, j], label="Real", alpha=0.6)
    ax.plot(t, tau_sim[:, j], "--", label="MuJoCo")
    ax.set_ylabel(f"J{j+1} [Nm]")
    ax.grid(True)
    if j == 0:
        ax.legend()

    # Residual
    err = tau_sim[:, j] - tau_real[:, j]
    rmse = np.sqrt(np.mean(err**2))

    axe = axes[j, 1]
    axe.plot(t, err, color="r")
    axe.set_title(f"RMSE = {rmse:.4f} Nm")
    axe.grid(True)

axes[-1, 0].set_xlabel("Time [s]")
axes[-1, 1].set_xlabel("Time [s]")

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig(f"{SAVE_DIR}/torque_validation_real_state1.png", dpi=300)
plt.close()

print("✅ Validation finished. Figure saved.")