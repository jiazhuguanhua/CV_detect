#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
红色球体检测器颜色校准工具
帮助用户调整HSV颜色范围参数
"""

import cv2
import numpy as np
import sys
import os

# 添加当前目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

class ColorCalibrator:
    """颜色校准工具"""
    
    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        self.cap = None
        
        # 初始HSV范围
        self.lower_h1 = 0
        self.upper_h1 = 10
        self.lower_s1 = 120
        self.upper_s1 = 255
        self.lower_v1 = 100
        self.upper_v1 = 255
        
        self.lower_h2 = 170
        self.upper_h2 = 180
        self.lower_s2 = 120
        self.upper_s2 = 255
        self.lower_v2 = 100
        self.upper_v2 = 255
        
    def initialize_camera(self):
        """初始化摄像头"""
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            print(f"无法打开摄像头 {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        return True
    
    def create_trackbars(self):
        """创建调节滑块"""
        cv2.namedWindow('Color Calibrator')
        cv2.namedWindow('Original')
        cv2.namedWindow('HSV Mask')
        cv2.namedWindow('Combined Mask')
        
        # 红色范围1的滑块
        cv2.createTrackbar('Lower H1', 'Color Calibrator', self.lower_h1, 179, lambda x: None)
        cv2.createTrackbar('Upper H1', 'Color Calibrator', self.upper_h1, 179, lambda x: None)
        cv2.createTrackbar('Lower S1', 'Color Calibrator', self.lower_s1, 255, lambda x: None)
        cv2.createTrackbar('Upper S1', 'Color Calibrator', self.upper_s1, 255, lambda x: None)
        cv2.createTrackbar('Lower V1', 'Color Calibrator', self.lower_v1, 255, lambda x: None)
        cv2.createTrackbar('Upper V1', 'Color Calibrator', self.upper_v1, 255, lambda x: None)
        
        # 红色范围2的滑块
        cv2.createTrackbar('Lower H2', 'Color Calibrator', self.lower_h2, 179, lambda x: None)
        cv2.createTrackbar('Upper H2', 'Color Calibrator', self.upper_h2, 179, lambda x: None)
        cv2.createTrackbar('Lower S2', 'Color Calibrator', self.lower_s2, 255, lambda x: None)
        cv2.createTrackbar('Upper S2', 'Color Calibrator', self.upper_s2, 255, lambda x: None)
        cv2.createTrackbar('Lower V2', 'Color Calibrator', self.lower_v2, 255, lambda x: None)
        cv2.createTrackbar('Upper V2', 'Color Calibrator', self.upper_v2, 255, lambda x: None)
        
        # 形态学操作参数
        cv2.createTrackbar('Morph Open', 'Color Calibrator', 3, 15, lambda x: None)
        cv2.createTrackbar('Morph Close', 'Color Calibrator', 3, 15, lambda x: None)
    
    def get_trackbar_values(self):
        """获取滑块值"""
        self.lower_h1 = cv2.getTrackbarPos('Lower H1', 'Color Calibrator')
        self.upper_h1 = cv2.getTrackbarPos('Upper H1', 'Color Calibrator')
        self.lower_s1 = cv2.getTrackbarPos('Lower S1', 'Color Calibrator')
        self.upper_s1 = cv2.getTrackbarPos('Upper S1', 'Color Calibrator')
        self.lower_v1 = cv2.getTrackbarPos('Lower V1', 'Color Calibrator')
        self.upper_v1 = cv2.getTrackbarPos('Upper V1', 'Color Calibrator')
        
        self.lower_h2 = cv2.getTrackbarPos('Lower H2', 'Color Calibrator')
        self.upper_h2 = cv2.getTrackbarPos('Upper H2', 'Color Calibrator')
        self.lower_s2 = cv2.getTrackbarPos('Lower S2', 'Color Calibrator')
        self.upper_s2 = cv2.getTrackbarPos('Upper S2', 'Color Calibrator')
        self.lower_v2 = cv2.getTrackbarPos('Lower V2', 'Color Calibrator')
        self.upper_v2 = cv2.getTrackbarPos('Upper V2', 'Color Calibrator')
        
        morph_open = cv2.getTrackbarPos('Morph Open', 'Color Calibrator')
        morph_close = cv2.getTrackbarPos('Morph Close', 'Color Calibrator')
        
        return morph_open, morph_close
    
    def process_frame(self, frame):
        """处理帧并显示结果"""
        morph_open, morph_close = self.get_trackbar_values()
        
        # 高斯模糊
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        
        # 转换到HSV
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        
        # 创建红色掩码
        lower1 = np.array([self.lower_h1, self.lower_s1, self.lower_v1])
        upper1 = np.array([self.upper_h1, self.upper_s1, self.upper_v1])
        mask1 = cv2.inRange(hsv, lower1, upper1)
        
        lower2 = np.array([self.lower_h2, self.lower_s2, self.lower_v2])
        upper2 = np.array([self.upper_h2, self.upper_s2, self.upper_v2])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        
        hsv_mask = cv2.bitwise_or(mask1, mask2)
        
        # BGR方法（额外验证）
        b, g, r = cv2.split(frame)
        red_dominant = (r > g) & (r > b) & (r > 100)
        not_white = ~((r > 200) & (g > 200) & (b > 200))
        not_black = (r > 50) | (g > 50) | (b > 50)
        bgr_mask = red_dominant & not_white & not_black
        bgr_mask = bgr_mask.astype(np.uint8) * 255
        
        # 结合两种方法
        combined_mask = cv2.bitwise_and(hsv_mask, bgr_mask)
        
        # 形态学操作
        if morph_open > 0:
            kernel_open = np.ones((morph_open, morph_open), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_open)
        
        if morph_close > 0:
            kernel_close = np.ones((morph_close, morph_close), np.uint8)
            combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_close)
        
        # 显示结果
        cv2.imshow('Original', frame)
        cv2.imshow('HSV Mask', hsv_mask)
        cv2.imshow('Combined Mask', combined_mask)
        
        # 在原图上叠加掩码
        result = frame.copy()
        result[combined_mask > 0] = [0, 255, 0]  # 绿色叠加
        cv2.imshow('Color Calibrator', result)
        
        return combined_mask
    
    def save_config(self):
        """保存配置到文件"""
        config_text = f"""
# 校准后的颜色配置
# 复制以下参数到你的config.py文件中

# 红色范围1
LOWER_RED1 = [{self.lower_h1}, {self.lower_s1}, {self.lower_v1}]
UPPER_RED1 = [{self.upper_h1}, {self.upper_s1}, {self.upper_v1}]

# 红色范围2  
LOWER_RED2 = [{self.lower_h2}, {self.lower_s2}, {self.lower_v2}]
UPPER_RED2 = [{self.upper_h2}, {self.upper_s2}, {self.upper_v2}]
"""
        
        with open('calibrated_config.txt', 'w', encoding='utf-8') as f:
            f.write(config_text)
        
        print("配置已保存到 calibrated_config.txt")
        print(config_text)
    
    def run(self):
        """运行校准工具"""
        if not self.initialize_camera():
            return
        
        self.create_trackbars()
        
        print("颜色校准工具")
        print("=" * 40)
        print("使用滑块调整HSV范围参数")
        print("将红色球体放在摄像头前进行调试")
        print("按键说明:")
        print("  's' - 保存当前配置")
        print("  'r' - 重置为默认值")
        print("  'q' - 退出")
        print("=" * 40)
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("无法读取摄像头")
                break
            
            self.process_frame(frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.save_config()
            elif key == ord('r'):
                # 重置滑块
                cv2.setTrackbarPos('Lower H1', 'Color Calibrator', 0)
                cv2.setTrackbarPos('Upper H1', 'Color Calibrator', 10)
                cv2.setTrackbarPos('Lower S1', 'Color Calibrator', 120)
                cv2.setTrackbarPos('Upper S1', 'Color Calibrator', 255)
                cv2.setTrackbarPos('Lower V1', 'Color Calibrator', 100)
                cv2.setTrackbarPos('Upper V1', 'Color Calibrator', 255)
                
                cv2.setTrackbarPos('Lower H2', 'Color Calibrator', 170)
                cv2.setTrackbarPos('Upper H2', 'Color Calibrator', 180)
                cv2.setTrackbarPos('Lower S2', 'Color Calibrator', 120)
                cv2.setTrackbarPos('Upper S2', 'Color Calibrator', 255)
                cv2.setTrackbarPos('Lower V2', 'Color Calibrator', 100)
                cv2.setTrackbarPos('Upper V2', 'Color Calibrator', 255)
                
                print("已重置为默认值")
        
        self.cleanup()
    
    def cleanup(self):
        """清理资源"""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='红色球体检测器颜色校准工具')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    
    args = parser.parse_args()
    
    calibrator = ColorCalibrator(args.camera)
    calibrator.run()

if __name__ == "__main__":
    main()
