import pickle
import pprint  # 用于美观地打印复杂数据结构

# 指定您的 pkl 文件路径
file_path = 'estimation_results.pkl' 

try:
    with open(file_path, 'rb') as f:
        # 使用 'rb' (read binary) 模式打开文件
        data = pickle.load(f)
        
    print(f"✅ 成功加载文件: {file_path}")
    print("-" * 30)
    
    # 使用 pprint 打印加载的数据结构
    print("📋 文件内容的数据结构 (部分展示):")
    pprint.pprint(data) 
    
except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {file_path}")
except Exception as e:
    print(f"❌ 加载文件时发生错误: {e}")
    print("提示: pkl 文件可能已损坏或使用不同 Python/pickle 版本保存。")