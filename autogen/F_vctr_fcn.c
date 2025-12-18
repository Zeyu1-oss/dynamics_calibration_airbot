#include "F_vctr_fcn.h"
#include <math.h>
void F_vctr_fcn(double pi_frcn_0, double pi_frcn_1, double pi_frcn_10, double pi_frcn_11, double pi_frcn_12, double pi_frcn_13, double pi_frcn_14, double pi_frcn_15, double pi_frcn_16, double pi_frcn_17, double pi_frcn_2, double pi_frcn_3, double pi_frcn_4, double pi_frcn_5, double pi_frcn_6, double pi_frcn_7, double pi_frcn_8, double pi_frcn_9, double qd0, double qd1, double qd2, double qd3, double qd4, double qd5, double *out_671629354517289065) {
   out_671629354517289065[0] = pi_frcn_0*qd0 + pi_frcn_1*(((qd0) > 0) - ((qd0) < 0)) + pi_frcn_2;
   out_671629354517289065[1] = pi_frcn_3*qd1 + pi_frcn_4*(((qd1) > 0) - ((qd1) < 0)) + pi_frcn_5;
   out_671629354517289065[2] = pi_frcn_6*qd2 + pi_frcn_7*(((qd2) > 0) - ((qd2) < 0)) + pi_frcn_8;
   out_671629354517289065[3] = pi_frcn_10*(((qd3) > 0) - ((qd3) < 0)) + pi_frcn_11 + pi_frcn_9*qd3;
   out_671629354517289065[4] = pi_frcn_12*qd4 + pi_frcn_13*(((qd4) > 0) - ((qd4) < 0)) + pi_frcn_14;
   out_671629354517289065[5] = pi_frcn_15*qd5 + pi_frcn_16*(((qd5) > 0) - ((qd5) < 0)) + pi_frcn_17;
}
