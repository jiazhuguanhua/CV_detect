# C++版本编译说明

## 依赖安装

### Windows (使用vcpkg)
```cmd
vcpkg install opencv4[core,imgproc,imgcodecs,videoio,highgui]:x64-windows
```

### Linux (Ubuntu/Debian)
```bash
sudo apt-get update
sudo apt-get install libopencv-dev
```

### macOS (使用Homebrew)
```bash
brew install opencv
```

## 编译命令

### Windows (Visual Studio)
```cmd
cl /EHsc red_ball_detector.cpp /I"C:\vcpkg\installed\x64-windows\include" /link /LIBPATH:"C:\vcpkg\installed\x64-windows\lib" opencv_core4.lib opencv_imgproc4.lib opencv_imgcodecs4.lib opencv_videoio4.lib opencv_highgui4.lib
```

### Linux/macOS (GCC/Clang)
```bash
g++ -std=c++11 -O3 red_ball_detector.cpp -o red_ball_detector `pkg-config --cflags --libs opencv4`
```

### CMake (推荐)
创建 CMakeLists.txt:
```cmake
cmake_minimum_required(VERSION 3.10)
project(RedBallDetector)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(red_ball_detector red_ball_detector.cpp)
target_link_libraries(red_ball_detector ${OpenCV_LIBS})
```

编译:
```bash
mkdir build && cd build
cmake ..
make -j4
```

## 运行
```bash
# 使用默认摄像头
./red_ball_detector

# 指定摄像头ID
./red_ball_detector 1
```

## 性能对比

| 实现 | 语言 | 典型FPS | 内存占用 | 部署难度 |
|------|------|---------|----------|----------|
| red_ball_detector.py | Python | 20-30 | 50-80MB | 简单 |
| optimized_detector.py | Python | 25-35 | 45-70MB | 简单 |
| red_ball_detector.cpp | C++ | 40-60 | 20-40MB | 中等 |

## 推荐配置

- **快速原型**: 使用 Python 版本
- **嵌入式部署**: 使用 C++ 版本或 Python embedded 配置
- **高性能需求**: 使用 C++ 版本
