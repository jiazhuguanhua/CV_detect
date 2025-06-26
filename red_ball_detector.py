#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
高性能红色球体检测器
适用于嵌入式设备的轻量化实现
作者: GitHub Copilot
日期: 2025-06-26
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math

@dataclass
class DetectedBall:
    """检测到的球体信息"""
    x: int          # 中心x坐标
    y: int          # 中心y坐标
    radius: int     # 半径
    confidence: float  # 置信度
    area: float     # 面积

class RedBallDetector:
    """高性能红色球体检测器"""
    
    def __init__(self, camera_id: int = 0, target_fps: int = 30):
        """
        初始化检测器
        
        Args:
            camera_id: 摄像头ID
            target_fps: 目标帧率
        """
        self.camera_id = camera_id
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        
        # 摄像头参数
        self.cap = None
        self.frame_width = 640
        self.frame_height = 480
        
        # HSV颜色范围 (针对红色优化)
        # 红色在HSV中分为两个范围
        self.lower_red1 = np.array([0, 50, 50])
        self.upper_red1 = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 50, 50])
        self.upper_red2 = np.array([180, 255, 255])
        
        # 检测参数
        self.min_radius = 10
        self.max_radius = 100
        self.min_area = 200
        
        # 性能优化参数
        self.resize_factor = 0.75  # 图像缩放因子
        self.blur_kernel = 5       # 高斯模糊核大小
        self.morph_kernel = np.ones((3, 3), np.uint8)
        
        # Hough圆检测参数 (优化后)
        self.hough_dp = 2
        self.hough_min_dist = 30
        self.hough_param1 = 50
        self.hough_param2 = 30
        
        # 多线程相关
        self.frame_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=5)
        self.running = False
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # ROI优化
        self.use_roi = False
        self.roi_margin = 50
        self.last_detections = []
        
    def initialize_camera(self) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"错误: 无法打开摄像头 {self.camera_id}")
                return False
            
            # 设置摄像头参数
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 减少缓冲延迟
            
            # 验证设置
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            
            print(f"摄像头初始化成功: {actual_width}x{actual_height} @ {actual_fps}fps")
            return True
            
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        预处理帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            处理后的掩码和缩放因子
        """
        # 缩放图像以提高处理速度
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)
            scale_factor = self.resize_factor
        else:
            scale_factor = 1.0
        
        # 高斯模糊减少噪声
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        
        # 转换到HSV颜色空间
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, self.lower_red1, self.upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 形态学操作消除噪声
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.morph_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.morph_kernel)
        
        return mask, scale_factor
    
    def detect_circles_hough(self, mask: np.ndarray, scale_factor: float) -> List[DetectedBall]:
        """
        使用霍夫变换检测圆形
        
        Args:
            mask: 二值掩码
            scale_factor: 缩放因子
            
        Returns:
            检测到的球体列表
        """
        balls = []
        
        # 霍夫圆检测
        circles = cv2.HoughCircles(
            mask,
            cv2.HOUGH_GRADIENT,
            dp=self.hough_dp,
            minDist=self.hough_min_dist,
            param1=self.hough_param1,
            param2=self.hough_param2,
            minRadius=int(self.min_radius * scale_factor),
            maxRadius=int(self.max_radius * scale_factor)
        )
        
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            
            for (x, y, r) in circles:
                # 转换回原始坐标
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                orig_r = int(r / scale_factor)
                
                # 计算面积和置信度
                area = math.pi * orig_r * orig_r
                
                if area >= self.min_area:
                    # 简单的置信度计算：基于半径和面积的合理性
                    confidence = min(1.0, area / (math.pi * self.max_radius * self.max_radius))
                    
                    ball = DetectedBall(
                        x=orig_x,
                        y=orig_y,
                        radius=orig_r,
                        confidence=confidence,
                        area=area
                    )
                    balls.append(ball)
        
        return balls
    
    def detect_contours(self, mask: np.ndarray, scale_factor: float) -> List[DetectedBall]:
        """
        使用轮廓检测圆形（备用方法）
        
        Args:
            mask: 二值掩码
            scale_factor: 缩放因子
            
        Returns:
            检测到的球体列表
        """
        balls = []
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area >= self.min_area * scale_factor * scale_factor:
                # 计算最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                
                # 转换回原始坐标
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                orig_r = int(radius / scale_factor)
                orig_area = area / (scale_factor * scale_factor)
                
                if self.min_radius <= orig_r <= self.max_radius:
                    # 计算圆度（衡量形状接近圆的程度）
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * math.pi * area / (perimeter * perimeter)
                        confidence = min(1.0, circularity)
                        
                        if confidence > 0.3:  # 过滤掉形状不太圆的对象
                            ball = DetectedBall(
                                x=orig_x,
                                y=orig_y,
                                radius=orig_r,
                                confidence=confidence,
                                area=orig_area
                            )
                            balls.append(ball)
        
        return balls
    
    def process_frame(self, frame: np.ndarray) -> List[DetectedBall]:
        """
        处理单帧图像
        
        Args:
            frame: 输入帧
            
        Returns:
            检测到的球体列表
        """
        # 预处理
        mask, scale_factor = self.preprocess_frame(frame)
        
        # 使用霍夫变换检测
        balls_hough = self.detect_circles_hough(mask, scale_factor)
        
        # 如果霍夫变换没有找到足够的圆，使用轮廓检测作为补充
        if len(balls_hough) == 0:
            balls_contour = self.detect_contours(mask, scale_factor)
            balls = balls_contour
        else:
            balls = balls_hough
        
        # 按置信度排序
        balls.sort(key=lambda b: b.confidence, reverse=True)
        
        return balls
    
    def update_fps(self):
        """更新FPS计算"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_results(self, frame: np.ndarray, balls: List[DetectedBall]) -> np.ndarray:
        """
        在图像上绘制检测结果
        
        Args:
            frame: 输入帧
            balls: 检测到的球体列表
            
        Returns:
            绘制结果的图像
        """
        result_frame = frame.copy()
        
        for i, ball in enumerate(balls):
            # 绘制圆形边界
            cv2.circle(result_frame, (ball.x, ball.y), ball.radius, (0, 255, 0), 2)
            cv2.circle(result_frame, (ball.x, ball.y), 2, (0, 0, 255), 3)
            
            # 绘制信息文本
            info_text = f"Ball {i+1}: ({ball.x},{ball.y}) R={ball.radius}"
            confidence_text = f"Conf: {ball.confidence:.2f}"
            
            cv2.putText(result_frame, info_text, (ball.x - 50, ball.y - ball.radius - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(result_frame, confidence_text, (ball.x - 50, ball.y - ball.radius - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # 绘制统计信息
        stats_text = f"Count: {len(balls)} | FPS: {self.current_fps:.1f}"
        cv2.putText(result_frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return result_frame
    
    def print_detection_results(self, balls: List[DetectedBall]):
        """打印检测结果到控制台"""
        if balls:
            print(f"\n=== 检测结果 (FPS: {self.current_fps:.1f}) ===")
            print(f"检测到 {len(balls)} 个红色球体:")
            
            for i, ball in enumerate(balls, 1):
                print(f"球体 {i}:")
                print(f"  位置: ({ball.x}, {ball.y})")
                print(f"  半径: {ball.radius} 像素")
                print(f"  面积: {ball.area:.1f} 像素²")
                print(f"  置信度: {ball.confidence:.3f}")
                print(f"  距离中心: {math.sqrt(ball.x**2 + ball.y**2):.1f} 像素")
        else:
            print(f"未检测到红色球体 (FPS: {self.current_fps:.1f})")
    
    def run(self, show_window: bool = True, print_results: bool = True):
        """
        运行检测器
        
        Args:
            show_window: 是否显示窗口
            print_results: 是否打印结果到控制台
        """
        if not self.initialize_camera():
            return
        
        print("红色球体检测器启动...")
        print("按 'q' 键退出, 按 's' 键保存截图")
        
        self.running = True
        frame_count = 0
        
        try:
            while self.running:
                frame_start_time = time.time()
                
                # 读取帧
                ret, frame = self.cap.read()
                if not ret:
                    print("警告: 无法读取摄像头帧")
                    continue
                
                # 处理帧
                balls = self.process_frame(frame)
                
                # 更新FPS
                self.update_fps()
                
                # 打印结果
                if print_results and frame_count % 30 == 0:  # 每30帧打印一次
                    self.print_detection_results(balls)
                
                # 显示结果
                if show_window:
                    result_frame = self.draw_results(frame, balls)
                    cv2.imshow('Red Ball Detector', result_frame)
                    
                    # 处理按键
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        # 保存截图
                        timestamp = int(time.time())
                        filename = f"detection_result_{timestamp}.jpg"
                        cv2.imwrite(filename, result_frame)
                        print(f"截图已保存: {filename}")
                
                # 帧率控制
                frame_elapsed = time.time() - frame_start_time
                if frame_elapsed < self.frame_time:
                    time.sleep(self.frame_time - frame_elapsed)
                
                frame_count += 1
                
        except KeyboardInterrupt:
            print("\n检测器被用户中断")
        except Exception as e:
            print(f"检测器运行错误: {e}")
        finally:
            self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        self.running = False
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("检测器已停止")

def main():
    """主函数"""
    # 创建检测器实例
    detector = RedBallDetector(camera_id=0, target_fps=30)
    
    # 运行检测器
    detector.run(show_window=True, print_results=True)

if __name__ == "__main__":
    main()
