import numpy as np
from numpy import sign

def F_vctr_fcn(qd1, qd2, qd3, qd4, qd5, qd6, pi_frcn_1, pi_frcn_2, pi_frcn_3, pi_frcn_4, pi_frcn_5, pi_frcn_6, pi_frcn_7, pi_frcn_8, pi_frcn_9, pi_frcn_10, pi_frcn_11, pi_frcn_12, pi_frcn_13, pi_frcn_14, pi_frcn_15, pi_frcn_16, pi_frcn_17, pi_frcn_18):
    # Common Subexpression Elimination (CSE) Results

    # Matrix Construction (tau_frcn is a 6x1 vector)
    vector = np.zeros((6, 1))
    vector[0, 0] = pi_frcn_1*qd1 + pi_frcn_2*sign(qd1) + pi_frcn_3
    vector[1, 0] = pi_frcn_4*qd2 + pi_frcn_5*sign(qd2) + pi_frcn_6
    vector[2, 0] = pi_frcn_7*qd3 + pi_frcn_8*sign(qd3) + pi_frcn_9
    vector[3, 0] = pi_frcn_10*qd4 + pi_frcn_11*sign(qd4) + pi_frcn_12
    vector[4, 0] = pi_frcn_13*qd5 + pi_frcn_14*sign(qd5) + pi_frcn_15
    vector[5, 0] = pi_frcn_16*qd6 + pi_frcn_17*sign(qd6) + pi_frcn_18
    return vector
