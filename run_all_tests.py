import os
import subprocess
import time
import datetime

def run_test(cmd, index):
    """执行测试命令"""
    print(f"\n运行配置{index}: {cmd}")
    
    # 执行命令
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    execution_time = time.time() - start_time
    
    # 检查执行结果
    if result.returncode == 0:
        print(f"配置{index}评估完成! 耗时: {execution_time:.2f}秒")
    else:
        print(f"配置{index}评估失败! 耗时: {execution_time:.2f}秒")
        print(f"错误信息: {result.stderr}")
    
    return {
        "index": index,
        "command": cmd,
        "success": result.returncode == 0,
        "execution_time": execution_time
    }

def main():
    """主函数"""
    print("开始RSANet模型评估批处理...")
    
    # 创建日志目录
    log_dir = "eval_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建日志文件
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"eval_run_{timestamp}.log")
    
    # 打开日志文件
    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"RSANet模型评估批处理开始时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    
    # 测试命令列表
    cmds = [
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_base.pth"',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca-pa-dir.pth" --use_saspp --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm-spa-ca-pa-dir_dropblock.pth" --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir --use_dropblock',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm-spa-ca-pa-dir_gcnet.pth" --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir --use_gcnet',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_dropblock_gcnet.pth" --use_dropblock --use_gcnet',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_dropblock.pth" --use_saspp --use_dropblock',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_gcnet.pth" --use_saspp --use_gcnet',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm-spa-ca-pa-dir.pth" --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp.pth" --use_saspp',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_gcnet.pth" --use_gcnet',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_dropblock.pth" --use_dropblock',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_damm-spa-ca-pa-dir_dropblock_gcnet.pth" --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir --use_dropblock --use_gcnet',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca-pa-dir_dropblock.pth" --use_saspp --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir --use_dropblock',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca-pa-dir_gcnet.pth" --use_saspp --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir --use_gcnet',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_dropblock_gcnet.pth" --use_saspp --use_dropblock --use_gcnet',
        'python eval_chase.py --weight_path "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca-pa-dir_dropblock_gcnet.pth" --use_saspp --use_damm --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir --use_dropblock --use_gcnet'
    ]
    
    # 执行所有测试命令
    results = []
    for i, cmd in enumerate(cmds, 1):
        result = run_test(cmd, i)
        results.append(result)
        
        # 将结果写入日志
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"配置{i} - {cmd}\n")
            f.write(f"{'='*80}\n")
            f.write(f"开始时间: {datetime.datetime.fromtimestamp(time.time() - result['execution_time']).strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"执行时间: {result['execution_time']:.2f}秒\n")
            f.write(f"执行结果: {'成功' if result['success'] else '失败'}\n\n")
    
    # 记录总结信息
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\n所有配置评估完成! 结束时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print("\n所有配置评估完成!")
    print(f"详细日志已保存到: {log_file}")

if __name__ == "__main__":
    main()