#!/usr/bin/env python3

import sys
import time
import csv
from pathlib import Path

from state_control import StateMachineController
from airbot_state_machine import robotic_arm

CONTROL_DT = 0.01  
EFFORT_LIMIT = [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]

JOINT_LIMITS = [
    (-3.151, 2.08),
    (-2.963, 0.181),
    (-0.094, 3.161),
    (-3.012, 3.012),
    (-1.859, 1.859),
    (-3.017, 3.017),
]

MAX_VELOCITY = [2, 2, 2, 2, 2, 2]
KP_GAINS = [80.0, 80.0, 80.0, 40.0, 40.0, 40.0]
KD_GAINS = [3.0, 3.0, 3.0, 0.3, 0.15, 0.2]

TAU_FF_SCALE = 0


def clamp_to_limits(q):
    q_clamped = list(q)
    for i in range(min(len(q), len(JOINT_LIMITS))):
        lower, upper = JOINT_LIMITS[i]
        q_clamped[i] = max(lower, min(upper, q[i]))
    return q_clamped


def load_trajectory_from_csv(csv_file):
    trajectory = []
    clipped_count = [0] * 6
    
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 31:
                continue
            
            try:
                t = float(row[0])
                q = [float(row[i]) for i in range(1, 7)]
                qv = [float(row[i]) for i in range(7, 13)]
                tau_ff = [float(row[i]) for i in range(25, 31)]
                
                q_clipped = []
                for i in range(6):
                    lower, upper = JOINT_LIMITS[i]
                    if q[i] < lower or q[i] > upper:
                        clipped_count[i] += 1
                        q_clipped.append(max(lower, min(upper, q[i])))
                    else:
                        q_clipped.append(q[i])
                
                trajectory.append((t, q_clipped, qv, tau_ff))
            except (ValueError, IndexError):
                continue
    
    total_clipped = sum(clipped_count)
    if total_clipped > 0:
        print(f"Clipped {total_clipped}/{len(trajectory)} trajectory points")
    
    return trajectory


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV Trajectory MIT Control")
    parser.add_argument("--csv", type=str, default="../vali——0fre.csv")
    parser.add_argument("--can", type=str, default="slcan0")
    parser.add_argument("--eef", type=str, default="none")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--output", type=str, default="./trajectory_log.csv")
    parser.add_argument("--no-read", action="store_true")
    parser.add_argument("--ff-scale", type=float, default=0.8, help="Feedforward torque scale (0-1)")
    args = parser.parse_args()
    
    global TAU_FF_SCALE
    TAU_FF_SCALE = args.ff_scale
    
    print(f"Control frequency: {1.0/CONTROL_DT:.0f}Hz, Mode: MIT")
    print(f"Feedforward torque scale: {TAU_FF_SCALE:.2f}")
    
    try:
        trajectory = load_trajectory_from_csv(args.csv)
        if len(trajectory) == 0:
            print("No valid trajectory data")
            return 1
        
        total_time = trajectory[-1][0] - trajectory[0][0]
        
        if args.duration is None:
            args.duration = total_time
        else:
            args.duration = min(args.duration, total_time)
        print(f"Duration: {args.duration:.3f}s")
        
    except Exception as e:
        print(f"Failed to load trajectory: {e}")
        return 1
    
    print("Initializing...")
    resources_dir = Path(__file__).parent / "resources"
    controller = StateMachineController(resources_dir)
    
    if not controller.initialize(args.can, args.eef):
        print("Initialization failed")
        return 1
    print("Initialized")
    
    joint_states = controller.get_joint_states()
    if not joint_states:
        print("Cannot read joint states")
        controller.shutdown()
        return 1
    
    current_q = joint_states.q
    q_start = trajectory[0][1]
    
    max_diff = max(abs(current_q[i] - q_start[i]) for i in range(6))
    print(f"Distance to start: {max_diff:.3f} rad")
    
    if max_diff > 0.01:
        print("Moving to start position (MIT mode)...")
        controller._arm.set_param("arm.control_mode", robotic_arm.ControlMode.MIT)
        controller._arm.set_param("immediate_update", 1)
        time.sleep(0.1)
        
        move_duration = max(15.0, max_diff * 10.0)
        move_steps = int(move_duration / CONTROL_DT)
        
        for step in range(move_steps):
            t_ratio = (step + 1) / move_steps
            s = 3 * t_ratio**2 - 2 * t_ratio**3
            
            q_interp = [current_q[i] + s * (q_start[i] - current_q[i]) for i in range(6)]
            qv_interp = [0.0] * 6
            tau_ff_zero = [0.0] * 6
            
            controller._arm.mit(q_interp, qv_interp, tau_ff_zero, KP_GAINS, KD_GAINS)
            time.sleep(CONTROL_DT)
        
        for i in range(100):
            qv_zero = [0.0] * 6
            tau_ff_zero = [0.0] * 6
            controller._arm.mit(q_start, qv_zero, tau_ff_zero, KP_GAINS, KD_GAINS)
            time.sleep(CONTROL_DT)
        
        js = controller.get_joint_states()
        if js:
            final_q = js.q
            final_diff = max(abs(final_q[i] - q_start[i]) for i in range(6))
            
            if final_diff > 0.1:
                print(f"Warning: Position error {final_diff:.4f} rad")
                response = input("Continue? (y/n): ")
                if response.lower() != 'y':
                    controller.shutdown()
                    return 1
            else:
                print("Reached start position")
    
    recorded_data = {
        'time': [],
        'q': [],
        'qv': [],
        'tau': [],
        'q_des': [],
        'qv_des': []
    }
    
    print("MIT control mode ready for trajectory execution...")
    
    print("Starting trajectory execution...")
    for i in range(3, 0, -1):
        print(f"{i}...", end='\r')
        time.sleep(1)
    print("Running!")
    
    start_time = time.time()
    step_count = 0
    
    try:
        while True:
            elapsed_time = time.time() - start_time
            
            if elapsed_time >= args.duration:
                break
            
            traj_idx = min(int(elapsed_time / CONTROL_DT), len(trajectory) - 1)
            
            if traj_idx >= len(trajectory):
                break
            
            _, q_des, qv_des, tau_ff = trajectory[traj_idx]
            q_des = clamp_to_limits(q_des)
            
            tau_feedforward = [tau * TAU_FF_SCALE for tau in tau_ff]
            
            controller._arm.mit(q_des, qv_des, tau_feedforward, KP_GAINS, KD_GAINS)
            
            if args.no_read:
                q_actual = [0.0] * 6
                qv_actual = [0.0] * 6
                tau_actual = [0.0] * 6
            else:
                js = controller.get_joint_states()
                if js:
                    q_actual = js.q
                    qv_actual = js.dq
                    tau_actual = js.tau
                else:
                    q_actual = [0.0] * 6
                    qv_actual = [0.0] * 6
                    tau_actual = [0.0] * 6
            
            recorded_data['time'].append(elapsed_time)
            recorded_data['q'].append(list(q_actual))
            recorded_data['qv'].append(list(qv_actual))
            recorded_data['tau'].append(list(tau_actual))
            recorded_data['q_des'].append(list(q_des))
            recorded_data['qv_des'].append(list(qv_des))
            
            step_count += 1
            
            if step_count % 100 == 0:
                if not args.no_read:
                    errors = [abs(q_actual[i] - q_des[i]) for i in range(6)]
                    max_err = max(errors)
                    print(f"t={elapsed_time:.2f}s | max_error={max_err:.4f} rad", end='\r')
            
            time.sleep(CONTROL_DT)
            
    except KeyboardInterrupt:
        print("\nInterrupted")
    
    try:
        js = controller.get_joint_states()
        if js:
            final_q = js.q
            qv_zero = [0.0] * 6
            tau_ff_zero = [0.0] * 6
            for i in range(50):
                controller._arm.mit(final_q, qv_zero, tau_ff_zero, KP_GAINS, KD_GAINS)
                time.sleep(CONTROL_DT)
    except Exception as e:
        print(f"Stop failed: {e}")
    
    if len(recorded_data['time']) > 0:
        print("\nSaving data...")
        try:
            with open(args.output, 'w') as f:
                f.write('time,q1,q2,q3,q4,q5,q6,qv1,qv2,qv3,qv4,qv5,qv6,')
                f.write('tau1,tau2,tau3,tau4,tau5,tau6,')
                f.write('q_des1,q_des2,q_des3,q_des4,q_des5,q_des6,')
                f.write('qv_des1,qv_des2,qv_des3,qv_des4,qv_des5,qv_des6\n')
                
                for i in range(len(recorded_data['time'])):
                    row = [recorded_data['time'][i]]
                    row.extend(recorded_data['q'][i])
                    row.extend(recorded_data['qv'][i])
                    row.extend(recorded_data['tau'][i])
                    row.extend(recorded_data['q_des'][i])
                    row.extend(recorded_data['qv_des'][i])
                    f.write(','.join(f'{x:.6f}' for x in row) + '\n')
            
            print(f"Saved to {args.output} ({len(recorded_data['time'])} points)")
        except Exception as e:
            print(f"Save failed: {e}")
    
    controller.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())

