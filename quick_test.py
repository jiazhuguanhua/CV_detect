#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试修复后的检测器
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_detector import OptimizedRedBallDetector
from config import StrictRedBallConfig

def quick_test():
    """快速测试"""
    print("快速测试严格配置的检测器...")
    
    # 使用严格配置
    config = StrictRedBallConfig()
    detector = OptimizedRedBallDetector(config)
    
    print("配置参数:")
    print(f"  颜色范围1: H({config.LOWER_RED1[0]}-{config.UPPER_RED1[0]}), S({config.LOWER_RED1[1]}-{config.UPPER_RED1[1]}), V({config.LOWER_RED1[2]}-{config.UPPER_RED1[2]})")
    print(f"  颜色范围2: H({config.LOWER_RED2[0]}-{config.UPPER_RED2[0]}), S({config.LOWER_RED2[1]}-{config.UPPER_RED2[1]}), V({config.LOWER_RED2[2]}-{config.UPPER_RED2[2]})")
    print(f"  最小面积: {config.MIN_AREA}")
    print(f"  最小置信度: {config.MIN_CONFIDENCE}")
    print(f"  Hough参数: dp={config.HOUGH_DP}, param2={config.HOUGH_PARAM2}")
    
    print("\n启动检测器...")
    print("如果仍有误检，请:")
    print("1. 运行 python color_calibrator.py 进行颜色校准")
    print("2. 确保光照条件良好")
    print("3. 使用纯红色球体进行测试")
    
    detector.run()

if __name__ == "__main__":
    quick_test()
