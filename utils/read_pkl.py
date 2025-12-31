import pickle
import pprint  

# 指定您的 pkl 文件路径
file_path = '../results/estimation_results.pkl' 

try:
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    pprint.pprint(data) 
    
except FileNotFoundError:
    print(f"❌ 错误: 找不到文件 {file_path}")
except Exception as e:
    print(f"❌ 加载文件时发生错误: {e}")
    print("提示: pkl 文件可能已损坏或使用不同 Python/pickle 版本保存。")