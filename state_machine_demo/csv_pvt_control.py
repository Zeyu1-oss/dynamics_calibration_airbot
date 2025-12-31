#!/usr/bin/env python3

import sys
import time
import csv
import os
from pathlib import Path
from collections import deque

from state_control import StateMachineController

CONTROL_DT = 0.005 
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


class TorqueFilter:
    """Moving average filter for torque measurements"""
    def __init__(self, window_size=10, num_joints=6):
        self.window_size = window_size
        self.buffers = [deque(maxlen=window_size) for _ in range(num_joints)]
    
    def filter(self, tau):
        """Apply moving average filter to torque readings"""
        filtered = []
        for i, t in enumerate(tau):
            self.buffers[i].append(t)
            filtered.append(sum(self.buffers[i]) / len(self.buffers[i]))
        return filtered


def check_joint_limits(q):
    for i in range(min(len(q), len(JOINT_LIMITS))):
        lower, upper = JOINT_LIMITS[i]
        if q[i] < lower or q[i] > upper:
            return False, i, lower, upper, q[i]
    return True, None, None, None, None


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
            if len(row) < 13:
                continue
            
            try:
                t = float(row[0])
                q = [float(row[i]) for i in range(1, 7)]
                qv = [float(row[i]) for i in range(7, 13)]
                
                q_clipped = []
                for i in range(6):
                    lower, upper = JOINT_LIMITS[i]
                    if q[i] < lower or q[i] > upper:
                        clipped_count[i] += 1
                        q_clipped.append(max(lower, min(upper, q[i])))
                    else:
                        q_clipped.append(q[i])
                
                trajectory.append((t, q_clipped, qv))
            except (ValueError, IndexError):
                continue
    
    total_clipped = sum(clipped_count)
    if total_clipped > 0:
        for i in range(6):
            if clipped_count[i] > 0:
                print(f"      Joint{i+1}: {clipped_count[i]} points clipped")
        print(f"      Total: {total_clipped}/{len(trajectory)} trajectory points clipped")
    
    return trajectory


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="CSV Trajectory PVT Control")
    parser.add_argument("--csv", type=str, default="../results/data_csv/vali_ptrnSrch_N7T25QR-6.csv", help="CSV trajectory file path")
    parser.add_argument("--can", type=str, default="can0", help="CAN interface")
    parser.add_argument("--eef", type=str, default="none", help="End-effector type")
    parser.add_argument("--duration", type=float, default=None, help="Duration (seconds), default: full trajectory")
    parser.add_argument("--output", type=str, default=None, help="Output data file (default: auto-generated)")
    parser.add_argument("--no-read", action="store_true", help="Control-only mode: send commands without reading feedback")
    parser.add_argument("--tau-filter", type=int, default=1, help="Torque filter window size (default: 10)等於1不濾波")
    args = parser.parse_args()
    
    # Auto-generate output filename if not specified
    if args.output is None:
        # Create real_data directory
        real_data_dir = Path("./real_data")
        real_data_dir.mkdir(exist_ok=True)
        
        # Extract base name from input CSV
        input_basename = os.path.splitext(os.path.basename(args.csv))[0]
        args.output = str(real_data_dir / f"{input_basename}.csv")
    
    print(f"Input CSV: {args.csv}")
    print(f"Output CSV: {args.output}")
    
    
    try:
        trajectory = load_trajectory_from_csv(args.csv)
        if len(trajectory) == 0:
            return 1
        
        total_time = trajectory[-1][0] - trajectory[0][0]
        
        if args.duration is None:
            args.duration = total_time
        else:
            args.duration = min(args.duration, total_time)
        print(f"   - Duration: {args.duration:.3f}s")
        
    except Exception as e:
        return 1
    
    resources_dir = Path(__file__).parent / "resources"
    controller = StateMachineController(resources_dir)
    
    if not controller.initialize(args.can, args.eef):
        return 1
    
    joint_states = controller.get_joint_states()
    if joint_states:
        print(f"   Position: {[f'{p:.4f}' for p in joint_states.q]}")
        print(f"   Velocity: {[f'{v:.4f}' for v in joint_states.dq]}")
        print(f"   Torque: {[f'{t:.4f}' for t in joint_states.tau]}")
        current_q = joint_states.q
    else:
        controller.shutdown()
        return 1
    
    q_start = trajectory[0][1]
    
    print(f"   Current position: {[f'{p:.3f}' for p in current_q]}")
    print(f"   Target position: {[f'{p:.3f}' for p in q_start]}")
    
    max_diff = max(abs(current_q[i] - q_start[i]) for i in range(6))
    print(f"   Max position diff: {max_diff:.3f} rad ({max_diff*180/3.14159:.1f}°)")
    
    if max_diff > 0.01:
        print("\n   ⚠️  Need to move to trajectory start position")
        print("   ⚠️  Ensure no obstacles around the robot")
        print("   Moving in 3 seconds...")
        for i in range(3, 0, -1):
            print(f"   {i}...", end='\r')
            time.sleep(1)
        
        print("\n   Moving to start position...")
        move_duration = max(15.0, max_diff * 10.0)
        move_steps = int(move_duration / CONTROL_DT)
        
        print(f"   Move duration: {move_duration:.1f}s")
        
        for step in range(move_steps):
            t_ratio = (step + 1) / move_steps
            s = 3 * t_ratio**2 - 2 * t_ratio**3
            
            q_interp = [current_q[i] + s * (q_start[i] - current_q[i]) for i in range(6)]
            max_vel = [1.0, 1.0, 1.0, 0.8, 0.8, 0.8]
            
            controller.joint_pvt(q_interp, max_vel, EFFORT_LIMIT)
            time.sleep(CONTROL_DT)
            
            if step % 50 == 0:
                js = controller.get_joint_states()
                if js:
                    current_err = max(abs(js.q[i] - q_start[i]) for i in range(6))
                    print(f"   Progress: {s*100:.0f}% | Error: {current_err:.3f} rad", end='\r')
        
        print()
        
        print("   Holding target position...")
        for i in range(100):
            controller.joint_pvt(q_start, [0.5]*6, EFFORT_LIMIT)
            time.sleep(CONTROL_DT)
            if i % 20 == 0 and i > 0:
                js = controller.get_joint_states()
                if js:
                    err = max(abs(js.q[j] - q_start[j]) for j in range(6))
                    print(f"   Holding... Error: {err:.3f} rad", end='\r')
        
        js = controller.get_joint_states()
        if js:
            final_q = js.q
            final_diff = max(abs(final_q[i] - q_start[i]) for i in range(6))
            print(f"\n   Final error: {final_diff:.4f} rad ({final_diff*180/3.14159:.2f}°)")
            
            print("   Joint errors:")
            for i in range(6):
                err = abs(final_q[i] - q_start[i])
                status = "✓" if err < 0.1 else "✗"
                print(f"      Joint{i+1}: {err:.4f} rad ({err*180/3.14159:.1f}°) {status}")
            
            if final_diff > 0.1:
                print(f"\n   ⚠️  Warning: Failed to reach target, large error!")
                print(f"   This is usually due to CAN bus congestion or blocked joints 4-6")
                print(f"   Current position: {[f'{p:.3f}' for p in final_q]}")
                print(f"   Target position: {[f'{p:.3f}' for p in q_start]}")
                response = input("   Continue trajectory execution? (y/n): ")
                if response.lower() != 'y':
                    print("   User canceled")
                    controller.shutdown()
                    return 1
            else:
                print(f"\n   ✓ Reached start position")
    else:
        print("   ✓ Already at start position, no move needed")
    
    recorded_data = {
        'time': [],
        'q': [],
        'qv': [],
        'tau': [],
        'q_des': [],
        'qv_des': []
    }
    
    tau_filter = TorqueFilter(window_size=args.tau_filter)
    print(f"   ✓ Torque filter initialized (window={args.tau_filter})")
    
    print("\n6. Preparing trajectory execution...")
    print("   Robot will start moving!")
    print("   Ensure no obstacles around the robot")
    for i in range(5, 0, -1):
        print(f"   Starting in {i} seconds...", end='\r')
        time.sleep(1)
    print("\n   Executing trajectory!")
    
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
            
            _, q_des, qv_des = trajectory[traj_idx]
            
            safe, joint_idx, lower, upper, value = check_joint_limits(q_des)
            if not safe:
                print(f"   Desired: {value:.4f}, Limits: [{lower:.4f}, {upper:.4f}]")
                q_des = clamp_to_limits(q_des)
            
            max_vel_limits = [2.0] * 6
            
            controller.joint_pvt(q_des, max_vel_limits, EFFORT_LIMIT)
            
            if args.no_read:
                q_actual = [0.0] * 6
                qv_actual = [0.0] * 6
                tau_actual = [0.0] * 6
                tau_filtered = [0.0] * 6
            else:
                js = controller.get_joint_states()
                if js:
                    q_actual = js.q
                    qv_actual = js.dq
                    tau_actual = js.tau
                    tau_filtered = tau_filter.filter(tau_actual)
                else:
                    q_actual = [0.0] * 6
                    qv_actual = [0.0] * 6
                    tau_actual = [0.0] * 6
                    tau_filtered = [0.0] * 6
            
            recorded_data['time'].append(elapsed_time)
            recorded_data['q'].append(list(q_actual))
            recorded_data['qv'].append(list(qv_actual))
            recorded_data['tau'].append(list(tau_filtered))
            recorded_data['q_des'].append(list(q_des))
            recorded_data['qv_des'].append(list(qv_des))
            
            step_count += 1
            
            if step_count % 100 == 0:
                if args.no_read:
                    print(f"t={elapsed_time:.2f}s | [Control only] Cmd: q_des=[{', '.join(f'{q:.3f}' for q in q_des)}]")
                else:
                    errors = [abs(q_actual[i] - q_des[i]) for i in range(6)]
                    j456_status = "J4-6: " + ", ".join([f"J{i+1}err={errors[i]:.4f}" for i in range(3, 6)])
                    print(f"t={elapsed_time:.2f}s | "
                          f"q=[{', '.join(f'{q:.3f}' for q in q_actual)}] | "
                          f"{j456_status}")
            
            time.sleep(CONTROL_DT)
            
    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted (Ctrl+C)")
    
    try:
        js = controller.get_joint_states()
        if js:
            final_q = js.q
            controller.joint_pvt(final_q, [0.1]*6, EFFORT_LIMIT)
            time.sleep(0.5)
    except Exception as e:
        print(f"   ⚠️  Stop failed: {e}")
    
    if len(recorded_data['time']) > 0:
        print("\n8. Saving data...")
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
            
            print(f"   ✓ Data saved to: {args.output}")
            
        except Exception as e:
            print(f"   ⚠️  Save failed: {e}")
    
    try:
        controller.shutdown()
    except Exception as e:
        print(f"   ⚠️  Shutdown failed: {e}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

