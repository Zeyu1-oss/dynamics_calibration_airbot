import numpy as np
import pandas as pd
import os

df = pd.read_csv('trajectory_log2.csv')

time = df['time'].values
q = df[['q1', 'q2', 'q3', 'q4', 'q5', 'q6']].values
qd = df[['qv1', 'qv2', 'qv3', 'qv4', 'qv5', 'qv6']].values
tau = df[['tau1', 'tau2', 'tau3', 'tau4', 'tau5', 'tau6']].values

output_data = np.column_stack([
    time,
    q,
    qd,
    tau,
    tau,
    tau
])

output_data[:, 0] = np.round(output_data[:, 0], 2)

os.makedirs('results/data_csv', exist_ok=True)
np.savetxt('results/data_csv/trajectory_raw.csv', output_data, delimiter=',', fmt='%.6f')

