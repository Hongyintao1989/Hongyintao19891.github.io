import os
import time
import sys

# 创建日志文件并设置同时输出到终端和文件
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        # 确保内容立即写入文件和终端
        self.terminal.flush()
        self.file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def run_command(command):
    """运行命令并实时显示输出"""
    # 如果命令包含--use_damm参数，自动添加所有DAMM分支参数
    if "--use_damm" in command:
        command += " --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir"
    
    # 直接使用os.system来执行命令，这样输出会直接显示在终端上
    return os.system(command)

def run_config(config_name, command):
    """运行给定的配置并显示进度"""
    print("="*60)
    print(f"开始运行配置: {config_name}")
    print("-"*40)
    
    # 直接运行命令，输出会显示在终端上
    return_code = run_command(command)
    
    if return_code == 0:
        print(f"\n配置 {config_name} 训练成功完成!")
    else:
        print(f"\n配置 {config_name} 训练失败，返回码: {return_code}")
    
    print("="*60)
    print()  # 添加一个空行
    
    return return_code

def main():
    # 设置同时输出到终端和文件
    log_file = "training_log.txt"
    tee = Tee(log_file)
    sys.stdout = tee
    
    # 记录开始时间和日期
    start_time = time.time()
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    print(f"\n批量训练开始于: {start_datetime}")
    print("\n开始RSANet模型训练批处理...\n")
    
    # 所有配置及其命令 - 保持原有命令不变
    configs = [
        ("配置1: DAMM=true, GCNet=false, SASPP=false, DropBlock=false", 
         "python train_chase.py --use_damm"),
        
        ("配置2: DAMM=false, GCNet=false, SASPP=true, DropBlock=false", 
         "python train_chase.py --use_saspp"),
        
        ("配置3: DAMM=false, GCNet=true, SASPP=false, DropBlock=false", 
         "python train_chase.py --use_gcnet"),
        
        ("配置4: DAMM=true, GCNet=true, SASPP=false, DropBlock=false", 
         "python train_chase.py --use_damm --use_gcnet"),
        
        ("配置5: DAMM=true, GCNet=false, SASPP=true, DropBlock=false", 
         "python train_chase.py --use_damm --use_saspp"),
        
        ("配置6: DAMM=false, GCNet=true, SASPP=true, DropBlock=false", 
         "python train_chase.py --use_gcnet --use_saspp"),
        
        ("配置7: DAMM=false, GCNet=false, SASPP=false, DropBlock=true", 
         "python train_chase.py --use_dropblock"),
        
        ("配置8: DAMM=true, GCNet=false, SASPP=false, DropBlock=true", 
         "python train_chase.py --use_damm --use_dropblock"),
        
        ("配置9: DAMM=false, GCNet=true, SASPP=false, DropBlock=true", 
         "python train_chase.py --use_gcnet --use_dropblock"),
        
        ("配置10: DAMM=false, GCNet=false, SASPP=true, DropBlock=true", 
         "python train_chase.py --use_saspp --use_dropblock"),
        
        ("配置11: DAMM=true, GCNet=true, SASPP=true, DropBlock=false", 
         "python train_chase.py --use_damm --use_gcnet --use_saspp"),
        
        ("配置12: DAMM=true, GCNet=true, SASPP=false, DropBlock=true", 
         "python train_chase.py --use_damm --use_gcnet --use_dropblock"),
        
        ("配置13: DAMM=true, GCNet=false, SASPP=true, DropBlock=true", 
         "python train_chase.py --use_damm --use_saspp --use_dropblock"),
        
        ("配置14: DAMM=false, GCNet=true, SASPP=true, DropBlock=true", 
         "python train_chase.py --use_gcnet --use_saspp --use_dropblock"),
        
        ("配置15: DAMM=true, GCNet=true, SASPP=true, DropBlock=true (全部模块)", 
         "python train_chase.py --use_damm --use_gcnet --use_saspp --use_dropblock"),
        
        ("配置16: DAMM=false, GCNet=false, SASPP=false, DropBlock=false (基础模型)", 
         "python train_chase.py")
    ]
    
    # 运行所有配置
    for i, (config_name, command) in enumerate(configs):
        config_start_time = time.time()
        print(f"进度: {i+1}/{len(configs)}")
        
        # 运行配置并检查返回值
        return_code = run_config(config_name, command)
        
        # 如果运行失败，询问是否继续
        if return_code != 0:
            print(f"配置 {i+1} 运行失败。返回码: {return_code}")
            cont = input("是否继续下一个配置？(y/n): ")
            if cont.lower() != 'y':
                print("用户取消了批处理，退出程序。")
                break
        
        # 计算并显示此配置花费的时间
        config_time = time.time() - config_start_time
        hours, remainder = divmod(config_time, 3600)
        minutes, seconds = divmod(remainder, 60)
        print(f"配置 {i+1} 运行时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 计算并显示总运行时间
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # 获取结束时间
    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    print(f"所有配置训练完成!")
    print(f"批量训练开始于: {start_datetime}")
    print(f"批量训练结束于: {end_datetime}")
    print(f"总运行时间: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")
    
    # 关闭日志文件
    sys.stdout = sys.stdout.terminal
    tee.close()
    print(f"日志已保存到: {log_file}")

if __name__ == "__main__":
    main()