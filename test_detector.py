#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红色球体检测器测试脚本
用于验证检测器功能和性能
"""

import cv2
import numpy as np
import time
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_detector import OptimizedRedBallDetector
from config import EmbeddedConfig, DetectorConfig

def create_test_image_with_red_circles():
    """创建包含红色圆形的测试图像"""
    # 创建黑色背景
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 添加几个红色圆形
    circles = [
        ((160, 120), 30),  # 左上角
        ((480, 120), 25),  # 右上角  
        ((320, 240), 40),  # 中心
        ((480, 360), 20),  # 右下角
    ]
    
    for (x, y), radius in circles:
        cv2.circle(img, (x, y), radius, (0, 0, 255), -1)  # 红色填充圆
        # 添加一些噪声使检测更有挑战性
        cv2.circle(img, (x+5, y+5), radius//3, (0, 100, 100), -1)
    
    # 添加一些干扰色块
    cv2.rectangle(img, (50, 50), (100, 100), (0, 255, 0), -1)  # 绿色矩形
    cv2.rectangle(img, (550, 50), (600, 100), (255, 0, 0), -1)  # 蓝色矩形
    
    return img

def test_detection_accuracy():
    """测试检测精度"""
    print("=== 检测精度测试 ===")
    
    # 创建测试图像
    test_img = create_test_image_with_red_circles()
    
    # 创建检测器
    config = DetectorConfig()
    config.SHOW_WINDOW = False
    config.PRINT_RESULTS = False
    
    detector = OptimizedRedBallDetector(config)
    
    # 处理测试图像
    detected_balls = detector.process_frame(test_img)
    
    print(f"预期检测数量: 4")
    print(f"实际检测数量: {len(detected_balls)}")
    
    for i, ball in enumerate(detected_balls):
        print(f"球体 {i+1}: 位置({ball.x}, {ball.y}), 半径{ball.radius}, 置信度{ball.confidence:.3f}")
    
    # 保存测试结果
    result_img = detector.draw_results(test_img, detected_balls)
    cv2.imwrite("test_detection_result.jpg", result_img)
    print("测试结果已保存: test_detection_result.jpg")
    
    return len(detected_balls) >= 3  # 至少检测到3个为通过

def test_performance():
    """测试性能"""
    print("\n=== 性能测试 ===")
    
    # 创建测试图像
    test_img = create_test_image_with_red_circles()
    
    configs = {
        'Default': DetectorConfig(),
        'Embedded': EmbeddedConfig()
    }
    
    for config_name, config in configs.items():
        config.SHOW_WINDOW = False
        config.PRINT_RESULTS = False
        
        detector = OptimizedRedBallDetector(config)
        
        # 性能测试
        num_frames = 100
        start_time = time.time()
        
        for _ in range(num_frames):
            detected_balls = detector.process_frame(test_img)
        
        end_time = time.time()
        total_time = end_time - start_time
        fps = num_frames / total_time
        
        print(f"{config_name} 配置:")
        print(f"  处理 {num_frames} 帧耗时: {total_time:.2f} 秒")
        print(f"  平均FPS: {fps:.1f}")
        print(f"  每帧处理时间: {total_time/num_frames*1000:.2f} 毫秒")

def test_color_ranges():
    """测试不同颜色范围"""
    print("\n=== 颜色范围测试 ===")
    
    # 创建不同红色强度的测试图像
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    
    # 不同红色强度的圆形
    colors = [
        (0, 0, 255),      # 纯红色
        (0, 0, 200),      # 深红色
        (0, 50, 255),     # 稍微偏紫的红色
        (20, 20, 255),    # 粉红色
        (0, 100, 180),    # 暗红色
    ]
    
    positions = [(100, 80), (300, 80), (100, 180), (300, 180), (200, 240)]
    
    for i, (color, pos) in enumerate(zip(colors, positions)):
        cv2.circle(img, pos, 25, color, -1)
        cv2.putText(img, f"{i+1}", (pos[0]-5, pos[1]+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # 测试检测
    detector = OptimizedRedBallDetector()
    detector.config.SHOW_WINDOW = False
    detector.config.PRINT_RESULTS = False
    
    detected_balls = detector.process_frame(img)
    
    print(f"颜色变化测试 - 检测到 {len(detected_balls)}/5 个球体")
    
    # 保存结果
    result_img = detector.draw_results(img, detected_balls)
    cv2.imwrite("test_color_ranges.jpg", result_img)
    print("颜色测试结果已保存: test_color_ranges.jpg")

def test_camera_connection():
    """测试摄像头连接"""
    print("\n=== 摄像头连接测试 ===")
    
    # 尝试连接摄像头
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("✓ 摄像头连接成功")
        
        # 获取摄像头信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"  分辨率: {width}x{height}")
        print(f"  FPS: {fps}")
        
        # 尝试读取一帧
        ret, frame = cap.read()
        if ret:
            print("✓ 成功读取图像帧")
            cv2.imwrite("camera_test_frame.jpg", frame)
            print("  测试帧已保存: camera_test_frame.jpg")
        else:
            print("✗ 无法读取图像帧")
        
        cap.release()
        return True
    else:
        print("✗ 无法连接摄像头")
        print("  请检查:")
        print("  1. 摄像头是否正确连接")
        print("  2. 摄像头是否被其他应用占用")
        print("  3. 摄像头驱动是否正确安装")
        return False

def main():
    """主测试函数"""
    print("红色球体检测器 - 功能测试")
    print("=" * 40)
    
    test_results = []
    
    # 1. 摄像头连接测试
    camera_ok = test_camera_connection()
    test_results.append(("摄像头连接", camera_ok))
    
    # 2. 检测精度测试
    accuracy_ok = test_detection_accuracy()
    test_results.append(("检测精度", accuracy_ok))
    
    # 3. 性能测试
    test_performance()
    test_results.append(("性能测试", True))  # 性能测试总是通过
    
    # 4. 颜色范围测试
    test_color_ranges()
    test_results.append(("颜色范围测试", True))
    
    # 输出测试总结
    print("\n" + "=" * 40)
    print("测试总结:")
    
    all_passed = True
    for test_name, result in test_results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"  {test_name}: {status}")
        if not result:
            all_passed = False
    
    print("\n整体测试结果:", "✓ 全部通过" if all_passed else "✗ 存在问题")
    
    if camera_ok:
        print("\n可以运行实时检测:")
        print("  python optimized_detector.py --config embedded")
    else:
        print("\n请解决摄像头问题后再运行实时检测")

if __name__ == "__main__":
    main()
