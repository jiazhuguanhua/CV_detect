#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试修复后的检测器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_detector import OptimizedRedBallDetector
from config import BalancedConfig, StrictRedBallConfig, DetectorConfig

def quick_test():
    """快速测试"""
    print("选择测试配置:")
    print("1. 平衡配置 (推荐) - 能检测橙红色球体")
    print("2. 严格配置 - 只检测纯红色")
    print("3. 默认配置 - 宽松检测")
    
    choice = input("请选择 (1-3): ").strip()
    
    if choice == "1":
        config = BalancedConfig()
        config_name = "平衡配置"
    elif choice == "2":
        config = StrictRedBallConfig()
        config_name = "严格配置"
    else:
        config = DetectorConfig()
        config_name = "默认配置"
    
    print(f"\n使用 {config_name} 启动检测器...")
    detector = OptimizedRedBallDetector(config)
    
    print("配置参数:")
    print(f"  颜色范围1: H({config.LOWER_RED1[0]}-{config.UPPER_RED1[0]}), S({config.LOWER_RED1[1]}-{config.UPPER_RED1[1]}), V({config.LOWER_RED1[2]}-{config.UPPER_RED1[2]})")
    print(f"  最小面积: {config.MIN_AREA}")
    print(f"  最小置信度: {config.MIN_CONFIDENCE}")
    print(f"  Hough参数: param2={config.HOUGH_PARAM2}")
    
    print("\n按 'q' 退出检测器")
    detector.run()

if __name__ == "__main__":
    quick_test()
