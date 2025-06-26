#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简单启动脚本 - 使用平衡配置
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_detector import OptimizedRedBallDetector
from config import BalancedConfig

if __name__ == "__main__":
    print("启动平衡配置的红色球体检测器...")
    config = BalancedConfig()
    detector = OptimizedRedBallDetector(config)
    detector.run()
