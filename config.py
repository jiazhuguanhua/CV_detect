#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红色球体检测器配置文件
用于调整检测参数以适应不同的环境和硬件
"""

class DetectorConfig:
    """检测器配置类"""
    
    # 摄像头参数
    CAMERA_ID = 0
    TARGET_FPS = 30
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    
    # 颜色检测参数 (HSV) - 平衡的红色范围
    # 红色范围1 (0-15度) - 包含橙红色
    LOWER_RED1 = [0, 80, 80]   # 降低饱和度和亮度要求
    UPPER_RED1 = [15, 255, 255]  # 扩大色相范围包含橙红
    
    # 红色范围2 (165-180度)
    LOWER_RED2 = [165, 80, 80]   # 降低饱和度和亮度要求
    UPPER_RED2 = [180, 255, 255]
    
    # 球体大小参数
    MIN_RADIUS = 12      # 降低最小半径
    MAX_RADIUS = 120     # 增加最大半径
    MIN_AREA = 300       # 适中的最小面积
    
    # 性能优化参数
    RESIZE_FACTOR = 0.75  # 图像缩放因子 (0.5-1.0, 越小越快但精度降低)
    BLUR_KERNEL = 3       # 减小模糊核，保持边缘清晰
    MORPH_KERNEL_SIZE = 3 # 形态学操作核大小
    
    # Hough参数 - 平衡设置
    HOUGH_DP = 1          # 提高精度
    HOUGH_MIN_DIST = 35   # 适中的最小距离
    HOUGH_PARAM1 = 80     # 适中的边缘检测阈值
    HOUGH_PARAM2 = 35     # 适中的累加器阈值
    
    # 显示参数
    SHOW_WINDOW = True
    PRINT_RESULTS = True
    PRINT_INTERVAL = 30   # 打印间隔帧数
    
    # 高级参数
    USE_ROI = False       # 是否使用ROI优化
    ROI_MARGIN = 50       # ROI边缘
    MIN_CONFIDENCE = 0.4  # 降低最小置信度阈值

class EmbeddedConfig(DetectorConfig):
    """嵌入式设备优化配置"""
    
    # 降低分辨率提高性能
    FRAME_WIDTH = 320
    FRAME_HEIGHT = 240
    TARGET_FPS = 20
    
    # 更激进的缩放
    RESIZE_FACTOR = 0.6
    
    # 简化处理
    BLUR_KERNEL = 3
    HOUGH_DP = 3
    HOUGH_MIN_DIST = 20
    
    # 关闭显示窗口节省资源
    SHOW_WINDOW = False
    PRINT_INTERVAL = 60

class HighPerformanceConfig(DetectorConfig):
    """高性能配置"""
    
    # 更高分辨率
    FRAME_WIDTH = 1280
    FRAME_HEIGHT = 720
    TARGET_FPS = 60
    
    # 保持原始尺寸
    RESIZE_FACTOR = 1.0
    
    # 更精细的检测
    BLUR_KERNEL = 7
    HOUGH_DP = 1
    HOUGH_MIN_DIST = 50
    HOUGH_PARAM1 = 100
    HOUGH_PARAM2 = 20
    
    # 更严格的过滤
    MIN_CONFIDENCE = 0.5

class OutdoorConfig(DetectorConfig):
    """户外环境配置"""
    
    # 调整颜色范围适应阳光
    LOWER_RED1 = [0, 70, 70]
    UPPER_RED1 = [10, 255, 255]
    LOWER_RED2 = [170, 70, 70]
    UPPER_RED2 = [180, 255, 255]
    
    # 更大的模糊核处理光照变化
    BLUR_KERNEL = 7
    
    # 调整检测参数
    HOUGH_PARAM1 = 80
    HOUGH_PARAM2 = 40

class IndoorConfig(DetectorConfig):
    """室内环境配置"""
    
    # 适应室内较暗环境
    LOWER_RED1 = [0, 40, 40]
    UPPER_RED1 = [10, 255, 255]
    LOWER_RED2 = [170, 40, 40]
    UPPER_RED2 = [180, 255, 255]
    
    # 更敏感的检测
    HOUGH_PARAM2 = 25
    MIN_CONFIDENCE = 0.2

class StrictRedBallConfig(DetectorConfig):
    """严格红色球体检测配置 - 减少误检"""
    
    # 非常严格的颜色范围
    LOWER_RED1 = [0, 150, 150]   # 非常高的饱和度和亮度要求
    UPPER_RED1 = [6, 255, 255]   # 很窄的色相范围
    
    LOWER_RED2 = [174, 150, 150] # 非常高的饱和度和亮度要求
    UPPER_RED2 = [180, 255, 255] # 很窄的色相范围
    
    # 更严格的尺寸要求
    MIN_RADIUS = 20
    MAX_RADIUS = 60
    MIN_AREA = 1000  # 更大的最小面积
    
    # 超严格的Hough参数
    HOUGH_DP = 1
    HOUGH_MIN_DIST = 50
    HOUGH_PARAM1 = 120
    HOUGH_PARAM2 = 60    # 非常高的阈值
    
    # 最严格的置信度
    MIN_CONFIDENCE = 0.8
    
    # 更精细的处理
    BLUR_KERNEL = 3
    RESIZE_FACTOR = 1.0  # 不缩放，保持最高精度
