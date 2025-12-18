#!/usr/bin/env python3
"""
运行动力学参数验证的示例脚本

使用方法:
1. 先运行 parameter_estimation.py 生成 estimation_results.pkl
2. 然后运行此脚本进行验证

或者直接从Python中导入使用:
    from dynamics.validation import validate_dynamic_params, load_estimation_results
"""

import sys
import os

# 确保可以导入dynamics模块
sys.path.insert(0, os.path.dirname(__file__))

from dynamics.validation import main

if __name__ == "__main__":
    
    try:
        results = main()
        
        print("\n" + "="*70)
        print("验证完成！查看以下文件获取详细结果：")
        print("="*70)
        print("  📊 图表:")
        print("     - validation_OLS_comparison.png")
        print("     - validation_PC-OLS-REG_comparison.png")
        print("     - validation_comparison_methods.png")
        print("\n  📄 CSV文件:")
        print("     - validation_OLS_detailed.csv")
        print("     - validation_OLS_summary.csv")
        print("     - validation_PC-OLS-REG_detailed.csv")
        print("     - validation_PC-OLS-REG_summary.csv")
        print("\n  💾 结果文件:")
        print("     - validation_results.pkl")
        print("="*70)
        
    except FileNotFoundError as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 解决方法:")
        print("   1. 先运行参数估计: python dynamics/parameter_estimation.py")
        print("   2. 确保 estimation_results.pkl 文件存在")
        print("   3. 检查 vali.csv 数据文件是否存在")
        sys.exit(1)
    
    except Exception as e:
        print(f"\n❌ 验证过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

