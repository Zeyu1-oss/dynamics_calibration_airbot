import xml.etree.ElementTree as ET
import numpy as np


def vec2skewSymMat(v):
    """
    将 3D 向量转换为对应的斜对称矩阵 (Skew-symmetric matrix).
    
    [ 0, -v3,  v2 ]
    [ v3,  0, -v1 ]
    [-v2, v1,  0  ]
    """
    if isinstance(v, list):
        v = np.array(v).flatten()
    elif v.ndim > 1:
        v = v.flatten()
        
    v1, v2, v3 = v
    return np.array([
        [0, -v3, v2],
        [v3, 0, -v1],
        [-v2, v1, 0]
    ])

def inertiaMatrix2Vector(I):
    """
    将 3x3 惯性张量矩阵转换为标准的 6 维惯性向量 (Ixx, Ixy, Ixz, Iyy, Iyz, Izz).
    """
    I_vec = np.array([
        I[0, 0],  # Ixx
        I[0, 1],  # Ixy
        I[0, 2],  # Ixz
        I[1, 1],  # Iyy
        I[1, 2],  # Iyz
        I[2, 2]   # Izz
    ])
    return I_vec.reshape((6, 1))

# --- 主解析函数 ---

def parse_urdf(file_path):
    
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except FileNotFoundError:
        print(f"Error: File not found at {file_path}")
        return None
    except ET.ParseError:
        print(f"Error: Failed to parse XML file at {file_path}")
        return None

    # URDF 的根标签应该是 'robot'
    robot_element = root
    
    # 只读取前5个link (link1到link5)，跳过base_link
    NO_DOF = 5
    
    # 初始化存储机器人参数的字典
    robot = {
        'm': np.zeros(NO_DOF),          # link_mass (1x5)
        'k': np.zeros((3, NO_DOF)),     # axis_of_rot (3x5)
        'r_com': np.zeros((3, NO_DOF)), # com_pos (3x5)
        'I': np.zeros((3, 3, NO_DOF)),  # link_inertia (3x3x5)
        'h': np.zeros((3, NO_DOF)),     # link_mass * com_pos (3x5)
        'I_vec': np.zeros((6, NO_DOF)), # 修正后的惯量向量 (6x5)
        'pi': np.zeros((10, NO_DOF))    # 标准惯性参数向量 (10x5)
    }

    # 获取所有 joint 和 link 元素
    joints = robot_element.findall('joint')
    links = robot_element.findall('link')

    if len(joints) < NO_DOF:
        print(f"Warning: Found only {len(joints)} joints, expected {NO_DOF}.")

    for i in range(NO_DOF):
        if i >= len(joints):
            break
            
        joint_element = joints[i]
        link_index = i + 1  # 对应 link[i+1] 的惯性参数
        
        if link_index >= len(links):
             print(f"Error: Missing link element for index {link_index}. Skipping.")
             continue
             
        link_element = links[link_index]

        # 1. 提取 Joint 参数 (轴)
        try:
            axis_of_rot_str = joint_element.find('axis').attrib['xyz']
            # 将字符串 'x y z' 转换为 NumPy 向量
            axis_of_rot = np.array([float(x) for x in axis_of_rot_str.split()])
        except AttributeError:
            print(f"Warning: Joint {i+1} missing axis data.")
            continue

        # 2. 提取 Link 参数 (质量, COM, 惯性张量)
        try:
            inertial = link_element.find('inertial')
            link_mass = float(inertial.find('mass').attrib['value'])
            
            com_pos_str = inertial.find('origin').attrib['xyz']
            com_pos = np.array([float(x) for x in com_pos_str.split()])
            
            inertia = inertial.find('inertia').attrib
            ixx = float(inertia['ixx'])
            ixy = float(inertia['ixy'])
            ixz = float(inertia['ixz'])
            iyy = float(inertia['iyy'])
            iyz = float(inertia['iyz'])
            izz = float(inertia['izz'])
        except (AttributeError, KeyError):
            print(f"Warning: Link {link_index} missing inertial data. Skipping.")
            continue
        
        # 惯性张量
        link_inertia = np.array([
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz]
        ])
        
        # --- 计算和存储参数 ---
        
        com_vec2mat = vec2skewSymMat(com_pos)

        # 1. 存储基本参数 (列索引 i 对应第 i+1 个连杆)
        robot['m'][i] = link_mass
        robot['k'][:, i] = axis_of_rot
        robot['r_com'][:, i] = com_pos
        robot['I'][:, :, i] = link_inertia
        
        # 2. 存储 H 向量 (h = m * r_com)
        robot['h'][:, i] = link_mass * com_pos
        
        # 3. 存储修正后的惯性向量 I_vec，平行轴定理。
        I_new = link_inertia - link_mass * com_vec2mat @ com_vec2mat
        robot['I_vec'][:, i] = inertiaMatrix2Vector(I_new).flatten()
        
        # 4. 存储标准惯性参数向量 pi (10x1)
        # pi = [I_vec; h; m]
        # I_vec (6x1), h (3x1), m (1x1)
        
        pi_vector = np.concatenate([
            robot['I_vec'][:, i].reshape((6, 1)),
            robot['h'][:, i].reshape((3, 1)),
            np.array([[link_mass]])
        ]).flatten() # 10x1
        
        robot['pi'][:, i] = pi_vector
        
    # 原始 MATLAB 代码返回的是 xml2struct 的完整结构，这里我们只返回提取的参数
    # 如果需要，可以将完整的 XML 解析结果添加到 'robot' 字典中
    
    return robot
