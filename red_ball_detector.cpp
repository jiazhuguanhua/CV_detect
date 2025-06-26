#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <chrono>
#include <cmath>

struct DetectedBall {
    cv::Point2f center;
    float radius;
    float confidence;
    float area;
};

class RedBallDetectorCpp {
private:
    cv::VideoCapture cap;
    
    // 检测参数
    int minRadius = 10;
    int maxRadius = 100;
    float minArea = 200.0f;
    float resizeFactor = 0.75f;
    
    // HSV颜色范围
    cv::Scalar lowerRed1 = cv::Scalar(0, 50, 50);
    cv::Scalar upperRed1 = cv::Scalar(10, 255, 255);
    cv::Scalar lowerRed2 = cv::Scalar(170, 50, 50);
    cv::Scalar upperRed2 = cv::Scalar(180, 255, 255);
    
    // Hough参数
    int houghDp = 2;
    int houghMinDist = 30;
    int houghParam1 = 50;
    int houghParam2 = 30;
    
    // 性能统计
    int frameCount = 0;
    std::chrono::steady_clock::time_point startTime;
    float currentFps = 0.0f;
    
public:
    RedBallDetectorCpp(int cameraId = 0) {
        cap.open(cameraId);
        if (!cap.isOpened()) {
            throw std::runtime_error("无法打开摄像头");
        }
        
        // 设置摄像头参数
        cap.set(cv::CAP_PROP_FRAME_WIDTH, 640);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
        cap.set(cv::CAP_PROP_FPS, 30);
        cap.set(cv::CAP_PROP_BUFFERSIZE, 1);
        
        startTime = std::chrono::steady_clock::now();
        
        std::cout << "C++ 红色球体检测器初始化完成" << std::endl;
    }
    
    cv::Mat preprocessFrame(const cv::Mat& frame, float& scaleFactor) {
        cv::Mat processedFrame;
        
        // 缩放图像
        if (resizeFactor != 1.0f) {
            cv::resize(frame, processedFrame, cv::Size(), resizeFactor, resizeFactor);
            scaleFactor = resizeFactor;
        } else {
            processedFrame = frame.clone();
            scaleFactor = 1.0f;
        }
        
        // 高斯模糊
        cv::GaussianBlur(processedFrame, processedFrame, cv::Size(5, 5), 0);
        
        // 转换到HSV
        cv::Mat hsv;
        cv::cvtColor(processedFrame, hsv, cv::COLOR_BGR2HSV);
        
        // 创建红色掩码
        cv::Mat mask1, mask2, mask;
        cv::inRange(hsv, lowerRed1, upperRed1, mask1);
        cv::inRange(hsv, lowerRed2, upperRed2, mask2);
        cv::bitwise_or(mask1, mask2, mask);
        
        // 形态学操作
        cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
        cv::morphologyEx(mask, mask, cv::MORPH_OPEN, kernel);
        cv::morphologyEx(mask, mask, cv::MORPH_CLOSE, kernel);
        
        return mask;
    }
    
    std::vector<DetectedBall> detectCircles(const cv::Mat& mask, float scaleFactor) {
        std::vector<DetectedBall> balls;
        std::vector<cv::Vec3f> circles;
        
        // Hough圆检测
        cv::HoughCircles(mask, circles, cv::HOUGH_GRADIENT, houghDp, houghMinDist,
                        houghParam1, houghParam2, 
                        static_cast<int>(minRadius * scaleFactor),
                        static_cast<int>(maxRadius * scaleFactor));
        
        for (const auto& circle : circles) {
            // 转换坐标回原尺度
            float origX = circle[0] / scaleFactor;
            float origY = circle[1] / scaleFactor;
            float origR = circle[2] / scaleFactor;
            
            float area = M_PI * origR * origR;
            
            if (area >= minArea) {
                float confidence = std::min(1.0f, area / (M_PI * maxRadius * maxRadius));
                
                DetectedBall ball;
                ball.center = cv::Point2f(origX, origY);
                ball.radius = origR;
                ball.confidence = confidence;
                ball.area = area;
                
                balls.push_back(ball);
            }
        }
        
        // 按置信度排序
        std::sort(balls.begin(), balls.end(), 
                 [](const DetectedBall& a, const DetectedBall& b) {
                     return a.confidence > b.confidence;
                 });
        
        return balls;
    }
    
    std::vector<DetectedBall> processFrame(const cv::Mat& frame) {
        float scaleFactor;
        cv::Mat mask = preprocessFrame(frame, scaleFactor);
        return detectCircles(mask, scaleFactor);
    }
    
    void updateFps() {
        frameCount++;
        auto currentTime = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(currentTime - startTime);
        
        if (elapsed.count() >= 1000) {  // 每秒更新一次
            currentFps = static_cast<float>(frameCount) / (elapsed.count() / 1000.0f);
            frameCount = 0;
            startTime = currentTime;
        }
    }
    
    cv::Mat drawResults(const cv::Mat& frame, const std::vector<DetectedBall>& balls) {
        cv::Mat result = frame.clone();
        
        for (size_t i = 0; i < balls.size(); ++i) {
            const auto& ball = balls[i];
            
            // 绘制圆形
            cv::Scalar color = ball.confidence > 0.7 ? cv::Scalar(0, 255, 0) : cv::Scalar(0, 255, 255);
            cv::circle(result, ball.center, static_cast<int>(ball.radius), color, 2);
            cv::circle(result, ball.center, 2, cv::Scalar(0, 0, 255), 3);
            
            // 绘制信息
            std::string info = "Ball " + std::to_string(i + 1) + ": (" + 
                              std::to_string(static_cast<int>(ball.center.x)) + "," + 
                              std::to_string(static_cast<int>(ball.center.y)) + ") R=" + 
                              std::to_string(static_cast<int>(ball.radius));
            
            std::string confInfo = "Conf: " + std::to_string(ball.confidence).substr(0, 4);
            
            cv::Point textPos(static_cast<int>(ball.center.x - 50), 
                             static_cast<int>(ball.center.y - ball.radius - 30));
            cv::Point confPos(static_cast<int>(ball.center.x - 50), 
                             static_cast<int>(ball.center.y - ball.radius - 15));
            
            cv::putText(result, info, textPos, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
            cv::putText(result, confInfo, confPos, cv::FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(255, 255, 255), 1);
        }
        
        // 状态信息
        std::string stats = "Count: " + std::to_string(balls.size()) + " | FPS: " + 
                           std::to_string(currentFps).substr(0, 4);
        cv::putText(result, stats, cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 255, 255), 2);
        
        return result;
    }
    
    void printResults(const std::vector<DetectedBall>& balls) {
        if (!balls.empty()) {
            std::cout << "\n=== 检测结果 (FPS: " << currentFps << ") ===" << std::endl;
            std::cout << "检测到 " << balls.size() << " 个红色球体:" << std::endl;
            
            for (size_t i = 0; i < balls.size(); ++i) {
                const auto& ball = balls[i];
                std::cout << "球体 " << (i + 1) << ":" << std::endl;
                std::cout << "  位置: (" << static_cast<int>(ball.center.x) 
                         << ", " << static_cast<int>(ball.center.y) << ")" << std::endl;
                std::cout << "  半径: " << static_cast<int>(ball.radius) << " 像素" << std::endl;
                std::cout << "  面积: " << ball.area << " 像素²" << std::endl;
                std::cout << "  置信度: " << ball.confidence << std::endl;
            }
        } else {
            std::cout << "未检测到红色球体 (FPS: " << currentFps << ")" << std::endl;
        }
    }
    
    void run() {
        std::cout << "C++ 红色球体检测器启动..." << std::endl;
        std::cout << "按 'q' 键退出, 按 's' 键保存截图" << std::endl;
        
        cv::Mat frame;
        int printCounter = 0;
        
        while (true) {
            auto frameStart = std::chrono::steady_clock::now();
            
            // 读取帧
            if (!cap.read(frame)) {
                std::cerr << "警告: 无法读取摄像头帧" << std::endl;
                continue;
            }
            
            // 处理帧
            auto balls = processFrame(frame);
            
            // 更新FPS
            updateFps();
            
            // 打印结果 (每30帧一次)
            if (++printCounter % 30 == 0) {
                printResults(balls);
            }
            
            // 显示结果
            cv::Mat result = drawResults(frame, balls);
            cv::imshow("C++ Red Ball Detector", result);
            
            // 处理按键
            int key = cv::waitKey(1) & 0xFF;
            if (key == 'q') {
                break;
            } else if (key == 's') {
                std::string filename = "cpp_detection_" + std::to_string(time(nullptr)) + ".jpg";
                cv::imwrite(filename, result);
                std::cout << "截图已保存: " << filename << std::endl;
            }
            
            // 帧率控制
            auto frameEnd = std::chrono::steady_clock::now();
            auto frameTime = std::chrono::duration_cast<std::chrono::milliseconds>(frameEnd - frameStart);
            
            if (frameTime.count() < 33) {  // 目标30fps
                std::this_thread::sleep_for(std::chrono::milliseconds(33 - frameTime.count()));
            }
        }
        
        cleanup();
    }
    
    void cleanup() {
        cap.release();
        cv::destroyAllWindows();
        std::cout << "C++ 检测器已停止" << std::endl;
    }
};

int main(int argc, char* argv[]) {
    try {
        int cameraId = 0;
        if (argc > 1) {
            cameraId = std::atoi(argv[1]);
        }
        
        RedBallDetectorCpp detector(cameraId);
        detector.run();
        
    } catch (const std::exception& e) {
        std::cerr << "错误: " << e.what() << std::endl;
        return -1;
    }
    
    return 0;
}
