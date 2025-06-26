@echo off
echo 红色球体检测器 - 启动菜单
echo ================================
echo.
echo 请选择运行模式:
echo 1. 严格检测模式 (推荐，减少误检)
echo 2. 嵌入式模式 (性能优化)
echo 3. 颜色校准工具 (调试用)
echo 4. 默认模式
echo 5. 退出
echo.
set /p choice="请输入选择 (1-5): "

if "%choice%"=="1" (
    echo 启动严格检测模式...
    python optimized_detector.py --config strict
) else if "%choice%"=="2" (
    echo 启动嵌入式模式...
    python optimized_detector.py --config embedded
) else if "%choice%"=="3" (
    echo 启动颜色校准工具...
    python color_calibrator.py
) else if "%choice%"=="4" (
    echo 启动默认模式...
    python optimized_detector.py
) else if "%choice%"=="5" (
    echo 退出
    exit
) else (
    echo 无效选择，启动严格模式...
    python optimized_detector.py --config strict
)

pause
