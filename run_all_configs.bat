@echo off
setlocal enabledelayedexpansion

:: 设置编码为UTF-8，确保中文正常显示
chcp 65001 > nul

:: 创建日志文件
set LOG_FILE=damm_ablation_training_log.txt
echo DAMM分支消融实验批量训练开始于: %date% %time% > %LOG_FILE%

:: 记录系统和环境信息，帮助追踪可能的变量
echo. >> %LOG_FILE%
echo ==================== 系统信息 ==================== >> %LOG_FILE%
echo 操作系统: %OS% >> %LOG_FILE%
echo 计算机名: %COMPUTERNAME% >> %LOG_FILE%
echo 用户名: %USERNAME% >> %LOG_FILE%
echo 处理器架构: %PROCESSOR_ARCHITECTURE% >> %LOG_FILE%
echo. >> %LOG_FILE%

:: 记录PyTorch版本和CUDA信息
python -c "import torch; print(f'PyTorch版本: {torch.__version__}'); print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'CUDA版本: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}'); print(f'GPU数量: {torch.cuda.device_count() if torch.cuda.is_available() else 0}')" >> %LOG_FILE%
echo. >> %LOG_FILE%

:: 设置基础命令，所有实验都启用DropBlock、GCNet和SASPP模块
set BASE_CMD=--use_dropblock --use_gcnet --use_saspp --use_damm

:: 输出基本信息
echo ==================== 训练配置 ==================== >> %LOG_FILE%
echo 基础命令: %BASE_CMD% >> %LOG_FILE%
echo. >> %LOG_FILE%

:: 记录开始时间
set START_TIME=%TIME%
echo 开始DAMM分支消融实验批量训练...
echo. >> %LOG_FILE%

:: 清理CUDA缓存工具函数
echo 创建CUDA缓存清理脚本...
echo import torch > clean_cuda.py
echo if torch.cuda.is_available(): >> clean_cuda.py
echo     torch.cuda.empty_cache() >> clean_cuda.py
echo     print("已清理CUDA缓存") >> clean_cuda.py
echo else: >> clean_cuda.py
echo     print("无CUDA可用") >> clean_cuda.py

:: 总配置数
set TOTAL_CONFIGS=17
set CURRENT_CONFIG=0
set START_TIME_SECONDS=0
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
   set /a "START_TIME_SECONDS=(((%%a*60)+1%%b %% 100)*60+1%%c %% 100)*100+1%%d %% 100"
)

:: ==================== 开始运行配置 ====================

:: 配置1: 完整DAMM (SPA+CA+PA+DIR 全部开启)
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - 完整DAMM (SPA+CA+PA+DIR 全部开启)
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - 完整DAMM (SPA+CA+PA+DIR 全部开启) >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置2: DAMM不使用SPA (CA+PA+DIR)
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用SPA (CA+PA+DIR)
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用SPA (CA+PA+DIR) >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_ca --use_damm_pa --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置3: DAMM不使用CA (SPA+PA+DIR)
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用CA (SPA+PA+DIR)
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用CA (SPA+PA+DIR) >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa --use_damm_pa --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置4: DAMM不使用PA (SPA+CA+DIR)
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用PA (SPA+CA+DIR)
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用PA (SPA+CA+DIR) >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa --use_damm_ca --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置5: DAMM不使用DIR (SPA+CA+PA)
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用DIR (SPA+CA+PA)
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM不使用DIR (SPA+CA+PA) >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa --use_damm_ca --use_damm_pa
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置6: DAMM只使用SPA+CA
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA+CA
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA+CA >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa --use_damm_ca
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置7: DAMM只使用SPA+PA
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA+PA
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA+PA >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa --use_damm_pa
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置8: DAMM只使用SPA+DIR
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA+DIR
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA+DIR >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置9: DAMM只使用CA+PA
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用CA+PA
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用CA+PA >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_ca --use_damm_pa
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置10: DAMM只使用CA+DIR
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用CA+DIR
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用CA+DIR >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_ca --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置11: DAMM只使用PA+DIR
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用PA+DIR
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用PA+DIR >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_pa --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置12: DAMM只使用SPA
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用SPA >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_spa
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置13: DAMM只使用CA
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用CA
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用CA >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_ca
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置14: DAMM只使用PA
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用PA
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用PA >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_pa
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置15: DAMM只使用DIR
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用DIR
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM只使用DIR >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD% --use_damm_dir
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置16: DAMM空壳 (所有分支都关闭)
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM空壳 (所有分支都关闭)
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - DAMM空壳 (所有分支都关闭) >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py %BASE_CMD%
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 计算已用时间和预计剩余时间
call :calculate_time

:: 配置17: 不使用DAMM (基准模型)
set /a CURRENT_CONFIG+=1
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS%
echo 开始运行配置: 配置!CURRENT_CONFIG! - 不使用DAMM (基准模型)
echo ================================================================================
echo 进度: !CURRENT_CONFIG!/%TOTAL_CONFIGS% >> %LOG_FILE%
echo 开始运行配置: 配置!CURRENT_CONFIG! - 不使用DAMM (基准模型) >> %LOG_FILE%
echo 执行时间: %date% %time% >> %LOG_FILE%

:: 清理CUDA缓存
python clean_cuda.py >> %LOG_FILE%

python train_chase.py --use_dropblock --use_gcnet --use_saspp
if %ERRORLEVEL% NEQ 0 (
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL% >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练失败，返回码: %ERRORLEVEL%
    choice /c yn /m "是否继续下一个配置 (Y/N)?"
    if !ERRORLEVEL! EQU 2 goto :end
) else (
    echo 配置!CURRENT_CONFIG!训练成功完成! >> %LOG_FILE%
    echo 配置!CURRENT_CONFIG!训练成功完成!
)

:: 训练全部完成
:end
echo ================================================================================
echo 所有DAMM分支消融实验配置训练完成!
echo ================================================================================
echo 所有DAMM分支消融实验配置训练完成! >> %LOG_FILE%

:: 计算总运行时间
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
   set /a "END_TIME_SECONDS=(((%%a*60)+1%%b %% 100)*60+1%%c %% 100)*100+1%%d %% 100"
)
set /a ELAPSED_SECONDS=%END_TIME_SECONDS%-%START_TIME_SECONDS%
if %ELAPSED_SECONDS% LSS 0 set /a ELAPSED_SECONDS+=24*3600*100

:: 转换为小时:分:秒格式
set /a ELAPSED_HOURS=%ELAPSED_SECONDS% / 360000
set /a ELAPSED_MINUTES=(%ELAPSED_SECONDS% - %ELAPSED_HOURS%*360000) / 6000
set /a ELAPSED_SECONDS=(%ELAPSED_SECONDS% - %ELAPSED_HOURS%*360000 - %ELAPSED_MINUTES%*6000) / 100

echo 批量训练开始于: %DATE% %START_TIME% >> %LOG_FILE%
echo 批量训练结束于: %DATE% %TIME% >> %LOG_FILE%
echo 总运行时间: %ELAPSED_HOURS%小时 %ELAPSED_MINUTES%分钟 %ELAPSED_SECONDS%秒 >> %LOG_FILE%

echo 批量训练开始于: %DATE% %START_TIME%
echo 批量训练结束于: %DATE% %TIME%
echo 总运行时间: %ELAPSED_HOURS%小时 %ELAPSED_MINUTES%分钟 %ELAPSED_SECONDS%秒

:: 删除临时文件
del clean_cuda.py

:: 自动启动评估，不询问用户
echo.
echo 开始评估所有模型...
echo 开始评估所有模型... >> %LOG_FILE%
python run_damm_ablation_eval.py

echo.
echo 日志已保存到: %LOG_FILE%
exit /b 0

:: 辅助函数 - 计算时间和预计剩余时间
:calculate_time
for /f "tokens=1-4 delims=:.," %%a in ("%time%") do (
  set /a "CURRENT_TIME_SECONDS=(((%%a*60)+1%%b %% 100)*60+1%%c %% 100)*100+1%%d %% 100"
)
set /a ELAPSED_SECONDS=%CURRENT_TIME_SECONDS%-%START_TIME_SECONDS%
if %ELAPSED_SECONDS% LSS 0 set /a ELAPSED_SECONDS+=24*3600*100

:: 计算每个配置平均时间
set /a AVG_TIME_PER_CONFIG=%ELAPSED_SECONDS% / %CURRENT_CONFIG%

:: 计算剩余时间
set /a REMAINING_CONFIGS=%TOTAL_CONFIGS%-%CURRENT_CONFIG%
set /a ESTIMATED_REMAINING_SECONDS=%AVG_TIME_PER_CONFIG% * %REMAINING_CONFIGS%

:: 转换为小时:分:秒格式
set /a ELAPSED_HOURS=%ELAPSED_SECONDS% / 360000
set /a ELAPSED_MINUTES=(%ELAPSED_SECONDS% - %ELAPSED_HOURS%*360000) / 6000
set /a ELAPSED_SECONDS=(%ELAPSED_SECONDS% - %ELAPSED_HOURS%*360000 - %ELAPSED_MINUTES%*6000) / 100

set /a REMAINING_HOURS=%ESTIMATED_REMAINING_SECONDS% / 360000
set /a REMAINING_MINUTES=(%ESTIMATED_REMAINING_SECONDS% - %REMAINING_HOURS%*360000) / 6000
set /a REMAINING_SECONDS=(%ESTIMATED_REMAINING_SECONDS% - %REMAINING_HOURS%*360000 - %REMAINING_MINUTES%*6000) / 100

echo 配置 %CURRENT_CONFIG% 运行时间: %ELAPSED_HOURS%小时 %ELAPSED_MINUTES%分钟 %ELAPSED_SECONDS%秒 >> %LOG_FILE%
echo 预计剩余时间: %REMAINING_HOURS%小时 %REMAINING_MINUTES%分钟 %REMAINING_SECONDS%秒 >> %LOG_FILE%

echo 配置 %CURRENT_CONFIG% 运行时间: %ELAPSED_HOURS%小时 %ELAPSED_MINUTES%分钟 %ELAPSED_SECONDS%秒
echo 预计剩余时间: %REMAINING_HOURS%小时 %REMAINING_MINUTES%分钟 %REMAINING_SECONDS%秒

:: 短暂休息以稳定系统状态
timeout /t 5 /nobreak > nul
goto :eof