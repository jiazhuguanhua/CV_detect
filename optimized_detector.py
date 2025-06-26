#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
优化版红色球体检测器
集成配置管理和性能优化
"""

import cv2
import numpy as np
import time
import threading
import queue
from typing import List, Tuple, Optional
from dataclasses import dataclass
import math
import argparse
from config import DetectorConfig, EmbeddedConfig, HighPerformanceConfig, OutdoorConfig, IndoorConfig, StrictRedBallConfig, BalancedConfig

@dataclass
class DetectedBall:
    """检测到的球体信息"""
    x: int
    y: int
    radius: int
    confidence: float
    area: float
    velocity: Optional[Tuple[float, float]] = None  # 运动速度 (vx, vy)

class OptimizedRedBallDetector:
    """优化版红色球体检测器"""
    
    def __init__(self, config: DetectorConfig = None):
        """
        初始化检测器
        
        Args:
            config: 配置对象
        """
        self.config = config if config else DetectorConfig()
        
        # 从配置初始化参数
        self._load_config()
        
        # 摄像头
        self.cap = None
        
        # 性能统计
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        self.total_detections = 0
        
        # 运动追踪
        self.previous_balls = []
        self.tracking_enabled = True
        
        # 自适应参数
        self.adaptive_threshold = True
        self.light_adaptation_frames = 0
        
        # 多尺度检测
        self.scales = [0.5, 0.75, 1.0] if hasattr(self.config, 'MULTI_SCALE') else [self.resize_factor]
        
        print(f"检测器初始化完成 - 目标FPS: {self.target_fps}, 分辨率: {self.frame_width}x{self.frame_height}")
    
    def _load_config(self):
        """从配置对象加载参数"""
        # 摄像头参数
        self.camera_id = self.config.CAMERA_ID
        self.target_fps = self.config.TARGET_FPS
        self.frame_width = self.config.FRAME_WIDTH
        self.frame_height = self.config.FRAME_HEIGHT
        self.frame_time = 1.0 / self.target_fps
        
        # 颜色范围
        self.lower_red1 = np.array(self.config.LOWER_RED1)
        self.upper_red1 = np.array(self.config.UPPER_RED1)
        self.lower_red2 = np.array(self.config.LOWER_RED2)
        self.upper_red2 = np.array(self.config.UPPER_RED2)
        
        # 检测参数
        self.min_radius = self.config.MIN_RADIUS
        self.max_radius = self.config.MAX_RADIUS
        self.min_area = self.config.MIN_AREA
        
        # 性能参数
        self.resize_factor = self.config.RESIZE_FACTOR
        self.blur_kernel = self.config.BLUR_KERNEL
        self.morph_kernel = np.ones((self.config.MORPH_KERNEL_SIZE, self.config.MORPH_KERNEL_SIZE), np.uint8)
        
        # Hough参数
        self.hough_dp = self.config.HOUGH_DP
        self.hough_min_dist = self.config.HOUGH_MIN_DIST
        self.hough_param1 = self.config.HOUGH_PARAM1
        self.hough_param2 = self.config.HOUGH_PARAM2
        
        # 显示参数
        self.show_window = self.config.SHOW_WINDOW
        self.print_results = self.config.PRINT_RESULTS
        self.print_interval = self.config.PRINT_INTERVAL
        self.min_confidence = self.config.MIN_CONFIDENCE
    
    def initialize_camera(self) -> bool:
        """初始化摄像头"""
        try:
            self.cap = cv2.VideoCapture(self.camera_id)
            if not self.cap.isOpened():
                print(f"错误: 无法打开摄像头 {self.camera_id}")
                return False
            
            # 优化摄像头设置
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
            self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # 设置摄像头优化参数
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # 自动曝光
            self.cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)        # 自动对焦
            
            # 验证设置
            actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.cap.get(cv2.CAP_PROP_FPS)
            
            print(f"摄像头设置: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
            return True
            
        except Exception as e:
            print(f"摄像头初始化失败: {e}")
            return False
    
    def adaptive_color_adjustment(self, hsv: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """自适应颜色范围调整"""
        if not self.adaptive_threshold:
            return self.lower_red1, self.upper_red1
        
        # 计算图像亮度
        brightness = np.mean(hsv[:, :, 2])
        
        # 根据亮度调整颜色范围
        if brightness < 80:  # 暗环境
            lower_red1 = self.lower_red1.copy()
            upper_red1 = self.upper_red1.copy()
            lower_red1[1] = max(30, lower_red1[1] - 20)  # 降低饱和度要求
            lower_red1[2] = max(30, lower_red1[2] - 20)  # 降低亮度要求
        elif brightness > 200:  # 亮环境
            lower_red1 = self.lower_red1.copy()
            upper_red1 = self.upper_red1.copy()
            lower_red1[1] = min(100, lower_red1[1] + 20)  # 提高饱和度要求
            lower_red1[2] = min(100, lower_red1[2] + 20)  # 提高亮度要求
        else:
            lower_red1, upper_red1 = self.lower_red1, self.upper_red1
        
        return lower_red1, upper_red1
    
    def multi_scale_detection(self, frame: np.ndarray) -> List[DetectedBall]:
        """多尺度检测"""
        all_balls = []
        
        for scale in self.scales:
            if scale != 1.0:
                scaled_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            else:
                scaled_frame = frame
            
            # 处理缩放后的帧
            mask, _ = self.preprocess_frame(scaled_frame)
            balls = self.detect_circles_hough(mask, scale)
            
            # 转换坐标回原尺度
            for ball in balls:
                if scale != 1.0:
                    ball.x = int(ball.x / scale)
                    ball.y = int(ball.y / scale)
                    ball.radius = int(ball.radius / scale)
                    ball.area = ball.area / (scale * scale)
                
                all_balls.append(ball)
        
        # 去重和合并相近的检测结果
        return self.merge_detections(all_balls)
    
    def merge_detections(self, balls: List[DetectedBall]) -> List[DetectedBall]:
        """合并相近的检测结果"""
        if len(balls) <= 1:
            return balls
        
        merged_balls = []
        used = set()
        
        for i, ball1 in enumerate(balls):
            if i in used:
                continue
            
            # 找到相近的球体
            similar_balls = [ball1]
            for j, ball2 in enumerate(balls[i+1:], i+1):
                if j in used:
                    continue
                
                # 计算距离
                dist = math.sqrt((ball1.x - ball2.x)**2 + (ball1.y - ball2.y)**2)
                overlap_threshold = max(ball1.radius, ball2.radius) * 0.7
                
                if dist < overlap_threshold:
                    similar_balls.append(ball2)
                    used.add(j)
            
            # 合并相似的球体
            if len(similar_balls) > 1:
                # 使用加权平均
                total_conf = sum(b.confidence for b in similar_balls)
                if total_conf > 0:
                    merged_x = sum(b.x * b.confidence for b in similar_balls) / total_conf
                    merged_y = sum(b.y * b.confidence for b in similar_balls) / total_conf
                    merged_r = sum(b.radius * b.confidence for b in similar_balls) / total_conf
                    merged_conf = max(b.confidence for b in similar_balls)
                    merged_area = sum(b.area for b in similar_balls) / len(similar_balls)
                    
                    merged_ball = DetectedBall(
                        x=int(merged_x),
                        y=int(merged_y),
                        radius=int(merged_r),
                        confidence=merged_conf,
                        area=merged_area
                    )
                    merged_balls.append(merged_ball)
            else:
                merged_balls.append(ball1)
        
        return merged_balls
    
    def track_motion(self, current_balls: List[DetectedBall]) -> List[DetectedBall]:
        """运动追踪"""
        if not self.tracking_enabled or not self.previous_balls:
            self.previous_balls = current_balls
            return current_balls
        
        # 简单的最近邻匹配
        for current_ball in current_balls:
            min_dist = float('inf')
            closest_prev = None
            
            for prev_ball in self.previous_balls:
                dist = math.sqrt((current_ball.x - prev_ball.x)**2 + 
                               (current_ball.y - prev_ball.y)**2)
                if dist < min_dist and dist < 50:  # 最大移动距离阈值
                    min_dist = dist
                    closest_prev = prev_ball
            
            # 计算速度
            if closest_prev:
                dt = 1.0 / max(self.current_fps, 1)  # 时间间隔
                vx = (current_ball.x - closest_prev.x) / dt
                vy = (current_ball.y - closest_prev.y) / dt
                current_ball.velocity = (vx, vy)
        
        self.previous_balls = current_balls
        return current_balls
    
    def preprocess_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, float]:
        """预处理帧图像 - 简化版本"""
        # 缩放图像
        if self.resize_factor != 1.0:
            frame = cv2.resize(frame, None, fx=self.resize_factor, fy=self.resize_factor)
            scale_factor = self.resize_factor
        else:
            scale_factor = 1.0
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(frame, (self.blur_kernel, self.blur_kernel), 0)
        
        # 转换到HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 自适应颜色调整
        lower_red1, upper_red1 = self.adaptive_color_adjustment(hsv)
        
        # 创建红色掩码
        mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
        mask = cv2.bitwise_or(mask1, mask2)
        
        # 简化的形态学操作
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask, scale_factor
    
    def detect_circles_hough(self, mask: np.ndarray, scale_factor: float) -> List[DetectedBall]:
        """霍夫圆检测"""
        balls = []
        
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
                # 转换坐标
                orig_x = int(x / scale_factor)
                orig_y = int(y / scale_factor)
                orig_r = int(r / scale_factor)
                
                area = math.pi * orig_r * orig_r
                
                if area >= self.min_area:
                    # 验证圆形区域的红色纯度 - 降低验证严格程度
                    circularity_score = self.verify_red_circle(mask, x, y, r)
                    
                    if circularity_score > 0.3:  # 降低阈值，从0.6改为0.3
                        confidence = circularity_score * min(1.0, area / (math.pi * self.max_radius * self.max_radius))
                        
                        if confidence >= self.min_confidence:
                            ball = DetectedBall(
                                x=orig_x,
                                y=orig_y,
                                radius=orig_r,
                                confidence=confidence,
                                area=area
                            )
                            balls.append(ball)
        
        return balls
    
    def verify_red_circle(self, mask: np.ndarray, x: int, y: int, radius: int) -> float:
        """验证检测到的圆形是否真的是红色球体 - 宽松版本"""
        # 创建圆形ROI
        roi_mask = np.zeros_like(mask)
        cv2.circle(roi_mask, (x, y), radius, 255, -1)
        
        # 计算圆形区域内的红色像素比例
        circle_area = np.sum(roi_mask > 0)
        red_pixels = np.sum((mask > 0) & (roi_mask > 0))
        
        if circle_area == 0:
            return 0.0
        
        fill_ratio = red_pixels / circle_area
        
        # 降低填充比例要求 - 只要有30%的区域是红色就接受
        if fill_ratio < 0.3:
            return 0.0
        
        # 检查圆形的完整性 - 但要求更宽松
        edge_completeness = self.check_circle_edge_completeness(mask, x, y, radius)
        
        # 更宽松的综合评分 - 更重视填充比例
        score = fill_ratio * 0.8 + edge_completeness * 0.2
        return min(1.0, score * 1.2)  # 稍微提升分数
    
    def check_circle_edge_completeness(self, mask: np.ndarray, x: int, y: int, radius: int) -> float:
        """检查圆形边缘的完整性"""
        edge_points = []
        num_points = 16  # 检查16个点
        
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            edge_x = int(x + radius * math.cos(angle))
            edge_y = int(y + radius * math.sin(angle))
            
            # 检查边缘点是否在图像范围内
            if 0 <= edge_x < mask.shape[1] and 0 <= edge_y < mask.shape[0]:
                edge_points.append(mask[edge_y, edge_x] > 0)
        
        if not edge_points:
            return 0.0
        
        # 计算边缘连续性
        continuity = sum(edge_points) / len(edge_points)
        return continuity
    
    def process_frame(self, frame: np.ndarray) -> List[DetectedBall]:
        """处理单帧"""
        # 预处理
        mask, scale_factor = self.preprocess_frame(frame)
        
        # 检测圆形
        balls = self.detect_circles_hough(mask, scale_factor)
        
        # 运动追踪
        balls = self.track_motion(balls)
        
        # 按置信度排序
        balls.sort(key=lambda b: b.confidence, reverse=True)
        
        # 更新统计
        self.total_detections += len(balls)
        
        return balls
    
    def update_fps(self):
        """更新FPS"""
        self.fps_counter += 1
        current_time = time.time()
        
        if current_time - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter / (current_time - self.fps_start_time)
            self.fps_counter = 0
            self.fps_start_time = current_time
    
    def draw_results(self, frame: np.ndarray, balls: List[DetectedBall]) -> np.ndarray:
        """绘制检测结果"""
        result_frame = frame.copy()
        
        for i, ball in enumerate(balls):
            # 绘制圆形
            color = (0, 255, 0) if ball.confidence > 0.7 else (0, 255, 255)
            cv2.circle(result_frame, (ball.x, ball.y), ball.radius, color, 2)
            cv2.circle(result_frame, (ball.x, ball.y), 2, (0, 0, 255), 3)
            
            # 绘制信息
            info_text = f"#{i+1}: ({ball.x},{ball.y}) R={ball.radius}"
            conf_text = f"Conf: {ball.confidence:.2f}"
            
            cv2.putText(result_frame, info_text, (ball.x - 50, ball.y - ball.radius - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            cv2.putText(result_frame, conf_text, (ball.x - 50, ball.y - ball.radius - 15),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            # 绘制运动轨迹
            if ball.velocity:
                vx, vy = ball.velocity
                speed = math.sqrt(vx*vx + vy*vy)
                if speed > 5:  # 只显示有明显运动的球体
                    end_x = int(ball.x + vx * 0.1)
                    end_y = int(ball.y + vy * 0.1)
                    cv2.arrowedLine(result_frame, (ball.x, ball.y), (end_x, end_y), (255, 0, 255), 2)
                    speed_text = f"Speed: {speed:.1f}"
                    cv2.putText(result_frame, speed_text, (ball.x - 50, ball.y - ball.radius - 45),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # 状态信息
        stats_text = f"Count: {len(balls)} | FPS: {self.current_fps:.1f} | Total: {self.total_detections}"
        cv2.putText(result_frame, stats_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return result_frame
    
    def print_detection_results(self, balls: List[DetectedBall]):
        """打印检测结果"""
        if balls:
            print(f"\n=== 检测结果 (FPS: {self.current_fps:.1f}) ===")
            print(f"检测到 {len(balls)} 个红色球体:")
            
            for i, ball in enumerate(balls, 1):
                print(f"球体 {i}:")
                print(f"  位置: ({ball.x}, {ball.y})")
                print(f"  半径: {ball.radius} 像素")
                print(f"  面积: {ball.area:.1f} 像素²")
                print(f"  置信度: {ball.confidence:.3f}")
                
                if ball.velocity:
                    vx, vy = ball.velocity
                    speed = math.sqrt(vx*vx + vy*vy)
                    print(f"  速度: ({vx:.1f}, {vy:.1f}) 像素/秒, 速率: {speed:.1f}")
                
                # 计算相对于图像中心的位置
                center_x, center_y = self.frame_width // 2, self.frame_height // 2
                rel_x = ball.x - center_x
                rel_y = ball.y - center_y
                distance = math.sqrt(rel_x*rel_x + rel_y*rel_y)
                print(f"  相对中心: ({rel_x:+d}, {rel_y:+d}), 距离: {distance:.1f}")
        else:
            print(f"未检测到红色球体 (FPS: {self.current_fps:.1f})")
    
    def run(self):
        """运行检测器"""
        if not self.initialize_camera():
            return
        
        print("优化版红色球体检测器启动...")
        print("按 'q' 键退出, 按 's' 键保存截图, 按 'r' 键重置统计")
        
        frame_count = 0
        
        try:
            while True:
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
                if self.print_results and frame_count % self.print_interval == 0:
                    self.print_detection_results(balls)
                
                # 显示结果
                if self.show_window:
                    result_frame = self.draw_results(frame, balls)
                    cv2.imshow('Optimized Red Ball Detector', result_frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('s'):
                        timestamp = int(time.time())
                        filename = f"detection_{timestamp}.jpg"
                        cv2.imwrite(filename, result_frame)
                        print(f"截图已保存: {filename}")
                    elif key == ord('r'):
                        self.total_detections = 0
                        print("统计已重置")
                
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
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        print("检测器已停止")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='优化版红色球体检测器')
    parser.add_argument('--config', choices=['default', 'embedded', 'high_perf', 'outdoor', 'indoor', 'strict', 'balanced'],
                       default='balanced', help='选择配置预设')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    
    args = parser.parse_args()
    
    # 选择配置
    config_map = {
        'default': DetectorConfig(),
        'embedded': EmbeddedConfig(),
        'high_perf': HighPerformanceConfig(),
        'outdoor': OutdoorConfig(),
        'indoor': IndoorConfig(),
        'strict': StrictRedBallConfig(),
        'balanced': BalancedConfig()
    }
    
    config = config_map[args.config]
    if args.camera != 0:
        config.CAMERA_ID = args.camera
    
    print(f"使用配置: {args.config}")
    
    # 创建并运行检测器
    detector = OptimizedRedBallDetector(config)
    detector.run()

if __name__ == "__main__":
    main()
