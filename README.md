# 高性能红色球体检测器

这是一个针对嵌入式设备优化的高性能红色球体检测程序，使用OpenCV和Python实现。

## 功能特点

- **高帧率检测**: 针对高FPS进行优化，可在嵌入式设备上稳定运行
- **多种配置预设**: 提供嵌入式、高性能、室内外等多种环境配置
- **智能检测算法**: 结合HSV颜色空间、霍夫圆检测和轮廓分析
- **运动追踪**: 实时追踪球体运动轨迹和速度
- **自适应调节**: 根据光照条件自动调整检测参数
- **性能监控**: 实时显示FPS和检测统计信息

## 系统要求

- Python 3.7+
- OpenCV 4.5+
- NumPy 1.20+
- USB摄像头或网络摄像头

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 基础使用

```bash
# 使用默认配置
python red_ball_detector.py

# 使用优化版检测器
python optimized_detector.py
```

### 配置选择

```bash
# 嵌入式设备优化配置（推荐用于树莓派等）
python optimized_detector.py --config embedded

# 高性能配置（用于性能较强的设备）
python optimized_detector.py --config high_perf

# 户外环境配置
python optimized_detector.py --config outdoor

# 室内环境配置
python optimized_detector.py --config indoor

# 指定摄像头
python optimized_detector.py --camera 1
```

## 配置说明

### 嵌入式配置 (EmbeddedConfig)
- 分辨率: 320x240
- 目标FPS: 20
- 关闭显示窗口节省资源
- 激进的图像缩放和简化处理

### 高性能配置 (HighPerformanceConfig)
- 分辨率: 1280x720
- 目标FPS: 60
- 保持原始图像尺寸
- 更精细的检测参数

### 户外配置 (OutdoorConfig)
- 调整颜色范围适应强光
- 增大模糊核处理光照变化
- 优化阳光环境下的检测

### 室内配置 (IndoorConfig)
- 适应室内较暗环境
- 降低颜色阈值
- 更敏感的检测参数

## 检测结果输出

程序会输出以下信息：

```
=== 检测结果 (FPS: 28.5) ===
检测到 2 个红色球体:
球体 1:
  位置: (320, 240)
  半径: 25 像素
  面积: 1963.5 像素²
  置信度: 0.856
  速度: (+12.3, -5.7) 像素/秒, 速率: 13.6
  相对中心: (+0, +0), 距离: 0.0

球体 2:
  位置: (150, 180)
  半径: 18 像素
  面积: 1017.9 像素²
  置信度: 0.742
  相对中心: (-170, -60), 距离: 180.4
```

## 性能优化技术

### 1. 图像预处理优化
- 动态图像缩放减少计算量
- 高斯模糊去噪
- HSV颜色空间提高颜色稳定性

### 2. 检测算法优化
- 霍夫圆检测参数调优
- 多尺度检测提高精度
- 轮廓分析作为备用方案

### 3. 性能提升技术
- 自适应颜色范围调整
- ROI（感兴趣区域）优化
- 运动预测减少搜索范围

### 4. 内存和CPU优化
- 最小化内存分配
- 向量化操作
- 避免不必要的图像拷贝

## 参数调节

主要参数在 `config.py` 中定义：

```python
# 颜色检测参数
LOWER_RED1 = [0, 50, 50]    # 红色范围1下限
UPPER_RED1 = [10, 255, 255]  # 红色范围1上限

# 球体大小参数
MIN_RADIUS = 10      # 最小半径
MAX_RADIUS = 100     # 最大半径
MIN_AREA = 200       # 最小面积

# 性能参数
RESIZE_FACTOR = 0.75  # 图像缩放因子
TARGET_FPS = 30       # 目标帧率
```

## 嵌入式部署建议

### 硬件要求
- ARM Cortex-A7 1GHz+ （如树莓派3B+）
- 512MB+ RAM
- USB2.0+ 摄像头

### 系统优化
1. 关闭不必要的系统服务
2. 使用 `embedded` 配置
3. 降低系统UI负载
4. 考虑使用GPU加速（如果可用）

### 部署脚本
```bash
# 创建systemd服务
sudo cp red_ball_detector.service /etc/systemd/system/
sudo systemctl enable red_ball_detector
sudo systemctl start red_ball_detector
```

## 故障排除

### 常见问题

1. **检测到非红色球体对象（如人脸、手等）**
   ```bash
   # 使用严格配置
   python optimized_detector.py --config strict
   
   # 或者进行颜色校准
   python color_calibrator.py
   ```
   
   **解决方案：**
   - 使用`strict`配置，它有更严格的颜色和形状验证
   - 调整光照条件，避免强光直射
   - 确保红色球体颜色纯正（避免粉红色、橙红色）
   - 使用颜色校准工具微调HSV参数

2. **摄像头无法打开**
   - 检查摄像头连接
   - 确认摄像头权限
   - 尝试不同的camera_id

3. **FPS过低**
   - 使用embedded配置
   - 降低分辨率
   - 增大RESIZE_FACTOR

4. **检测不准确**
   - 调整颜色范围参数
   - 改善照明条件
   - 调整最小/最大半径

5. **内存不足**
   - 降低分辨率
   - 使用更小的缓冲区
   - 关闭显示窗口

### 颜色校准工具

如果检测效果不理想，请使用颜色校准工具：

```bash
python color_calibrator.py
```

校准步骤：
1. 将红色球体放在摄像头前
2. 调整HSV滑块直到只有球体被检测到
3. 按's'保存配置
4. 将生成的参数复制到config.py中

### 配置选择指南

| 场景 | 推荐配置 | 说明 |
|------|----------|------|
| 误检严重 | `strict` | 最严格的检测，减少误检 |
| 嵌入式设备 | `embedded` | 性能优化，适合树莓派 |
| 室内环境 | `indoor` | 适应室内光照 |
| 户外环境 | `outdoor` | 适应强光环境 |
| 高性能需求 | `high_perf` | 最高精度和帧率 |

### 调试模式

使用以下代码可以显示中间处理结果：

```python
# 在 optimized_detector.py 中启用调试
detector.show_debug = True
```

## 扩展功能

程序支持以下扩展：

1. **多目标检测**: 修改颜色范围检测其他颜色球体
2. **距离估算**: 基于球体大小估算距离
3. **轨迹预测**: 基于运动历史预测球体轨迹
4. **网络传输**: 通过TCP/UDP发送检测结果

## 许可证

MIT License

## 贡献

欢迎提交问题和改进建议。
