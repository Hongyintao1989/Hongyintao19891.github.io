import os
import sys
import random
# 确保当前文件夹在模块搜索路径中，能够正确导入模块
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
import cv2
from model import RSANet
import matplotlib.pyplot as plt
from loss_search import CombinedLoss, LossSearchAgent, get_default_search_space
import json
from datetime import datetime
from rl_loss_search import ReinforcementLossSearch

# ----------------- 超参数配置 -----------------
batch_size    = 1
epochs        = 15  # 适当减少用于搜索的epoch数
learning_rate = 5e-4
weight_decay  = 1e-4
start_neurons = 16
keep_prob     = 0.87
block_size    = 7
desired_size  = 1008
# ----------------- 损失函数搜索配置 -----------------
search_generations = 150  # 搜索的代数
population_size = 10     # 种群大小
validation_epochs = 15   # 用于评估每个损失函数的epoch数
# -----------------------------------------------------------

# 设置随机种子以确保结果可复现
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ----------------- 命令行参数解析 -----------------
parser = argparse.ArgumentParser(description='RSANet Training with Loss Function Search')
parser.add_argument('--use_saspp', action='store_true', help='启用SASPP模块')
parser.add_argument('--use_damm', action='store_true', help='启用DAMM模块')
parser.add_argument('--use_dropblock', action='store_true', help='启用DropBlock模块')
parser.add_argument('--use_gcnet', action='store_true', help='启用GCNet模块')
parser.add_argument('--search_loss', action='store_true', help='启用损失函数自动搜索')
parser.add_argument('--load_loss', type=str, default='', help='加载预定义的损失函数配置')
parser.add_argument('--method', type=str, default='ga', choices=['ga', 'rl'],help='搜索方法: ga (遗传算法) 或 rl (强化学习)')
args = parser.parse_args()

# 确定模型名称
modules = []
if args.use_saspp:
    modules.append('saspp')
if args.use_damm:
    modules.append('damm')
if args.use_dropblock:
    modules.append('dropblock')
if args.use_gcnet:
    modules.append('gcnet')

module_name = '_'.join(modules) if modules else 'base'
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
experiment_name = f"{module_name}_{timestamp}"

# 打印使用的模块
module_info = []
if args.use_saspp:
    module_info.append('SASPP')
if args.use_damm:
    module_info.append('DAMM')
if args.use_dropblock:
    module_info.append('DropBlock')
if args.use_gcnet:
    module_info.append('GCNet')

print(f"模型使用模块: {' + '.join(module_info) if module_info else 'RSAN(无增强模块)'}")

# 数据目录设置
data_location        = ''
training_images_loc  = os.path.join(data_location, 'Chase/train/image/')
training_label_loc   = os.path.join(data_location, 'Chase/train/label/')
validate_images_loc  = os.path.join(data_location, 'Chase/validate/images/')
validate_label_loc   = os.path.join(data_location, 'Chase/validate/labels/')

# 自定义数据集定义
class ChaseDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels
    def __len__(self): return len(self.images)
    def __getitem__(self, idx): return self.images[idx], self.labels[idx]

# 加载数据函数
def load_data(files, img_dir, lbl_dir):
    data, labels = [], []
    for fn in files:
        im = cv2.imread(os.path.join(img_dir, fn), cv2.IMREAD_GRAYSCALE)
        lab = cv2.imread(os.path.join(lbl_dir, f"Image_{fn.split('_')[1].split('.')[0]}_1stHO.png"), cv2.IMREAD_GRAYSCALE)
        h, w = im.shape
        th, tw = desired_size, desired_size
        top = (th - h)//2; bottom = th - h - top
        left = (tw - w)//2; right = tw - w - left
        im_pad = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        lbl_pad = cv2.copyMakeBorder(lab, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        im_rz = cv2.resize(im_pad, (tw, th))
        _, lbl_rz = cv2.threshold(cv2.resize(lbl_pad, (tw, th)), 127, 255, cv2.THRESH_BINARY)
        data.append(im_rz.astype('float32')/255.)
        labels.append(lbl_rz.astype('float32')/255.)
    data = np.array(data).reshape(-1,1,desired_size,desired_size)
    labels = np.array(labels).reshape(-1,1,desired_size,desired_size)
    return torch.from_numpy(data), torch.from_numpy(labels)

# 创建保存路径
results_dir = f'search_results/{experiment_name}'
ckpt_dir = f'{results_dir}/checkpoint'
loss_dir = f'{results_dir}/loss'
log_dir = f'{results_dir}/logs'
search_dir = f'{results_dir}/search'

os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs(loss_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)
os.makedirs(search_dir, exist_ok=True)

# 保存损失函数配置
def save_loss_config(loss_config, fitness, generation=None):
    """保存损失函数配置到JSON文件"""
    config_data = {
        'loss_config': loss_config,
        'fitness': fitness,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    if generation is not None:
        filename = f"{search_dir}/loss_config_gen{generation}.json"
    else:
        filename = f"{search_dir}/best_loss_config.json"
    
    with open(filename, 'w') as f:
        json.dump(config_data, f, indent=4)
    
    print(f"损失函数配置已保存到 {filename}")

# 加载损失函数配置
def load_loss_config(filename):
    """从JSON文件加载损失函数配置"""
    with open(filename, 'r') as f:
        config_data = json.load(f)
    
    return config_data['loss_config']

# 训练单个epoch
def train_epoch(model, loader, optimizer, loss_fn, device):
    """训练单个epoch并返回平均损失"""
    model.train()
    total_loss = 0.0
    
    for imgs, lbls in loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        outs = model(imgs)
        loss, _ = loss_fn(outs, lbls)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    return total_loss / len(loader)

# 验证函数
def validate(model, loader, loss_fn, device):
    """在验证集上评估模型并返回平均损失"""
    model.eval()
    total_loss = 0.0
    
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            outs = model(imgs)
            loss, _ = loss_fn(outs, lbls)
            total_loss += loss.item()
    
    return total_loss / len(loader)

# 训练模型使用特定损失函数
def train_with_loss_config(model, loss_config, train_loader, val_loader, device, num_epochs=validation_epochs):
    """使用特定损失函数配置训练模型"""
    # 创建损失函数
    loss_fn = CombinedLoss(loss_config)
    print(f"使用损失函数: {loss_fn}")
    
    # 创建优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # 训练和验证
    train_losses = []
    val_losses = []
    
    for epoch in range(1, num_epochs + 1):
        # 训练一个epoch
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
        train_losses.append(train_loss)
        
        # 验证
        val_loss = validate(model, val_loader, loss_fn, device)
        val_losses.append(val_loss)
        
        print(f"Epoch {epoch}/{num_epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
    
    # 返回最终的验证损失作为适应度值
    return val_losses[-1]

def search_loss_function_ga(model, train_loader, val_loader, device):
    """搜索最佳损失函数组合"""
    print("开始损失函数自动搜索...")
    
    # 获取搜索空间
    search_space = get_default_search_space()
    
    # 创建搜索代理
    agent = LossSearchAgent(
        search_space=search_space,
        population_size=population_size,
        mutation_rate=0.3,
        crossover_rate=0.5,
        elite_size=1
    )
    
    best_fitness = float('inf')
    best_config = None
    
    # 记录搜索过程
    search_history = []
    
    # 逐代进化
    for generation in range(search_generations):
        print(f"\n=== 第 {generation+1}/{search_generations} 代 ===")
        
        # 获取当前种群
        population = agent.population
        
        # 评估每个个体的适应度
        fitness_values = []
        
        for i, loss_config in enumerate(population):
            print(f"\n评估个体 {i+1}/{len(population)}")
            
            # 重置模型权重
            model_copy = create_model(device)
            
            # 使用当前损失函数配置训练模型
            fitness = train_with_loss_config(
                model=model_copy,
                loss_config=loss_config,
                train_loader=train_loader,
                val_loader=val_loader,
                device=device
            )
            
            fitness_values.append(fitness)
            
            # 更新最佳个体
            if fitness < best_fitness:
                best_fitness = fitness
                best_config = loss_config.copy()
                
                # 保存最佳损失函数配置
                save_loss_config(best_config, best_fitness)
                
                print(f"\n发现新的最佳损失函数! 验证损失: {best_fitness:.4f}")
                print(f"配置: {CombinedLoss(best_config)}")
            
            # 记录当前个体的结果
            individual_record = {
                'generation': generation,
                'individual': i,
                'loss_config': loss_config,
                'fitness': fitness
            }
            search_history.append(individual_record)
        
        # 保存当前代的最佳个体
        gen_best_idx = fitness_values.index(min(fitness_values))
        save_loss_config(population[gen_best_idx], min(fitness_values), generation)
        
        # 进化到下一代
        if generation < search_generations - 1:
            agent.evolve(fitness_values)
        
        # 打印当前代的结果
        print(f"\n第 {generation+1} 代结果:")
        print(f"最佳适应度: {min(fitness_values):.4f}")
        print(f"平均适应度: {sum(fitness_values)/len(fitness_values):.4f}")
        print(f"最佳损失函数: {CombinedLoss(population[gen_best_idx])}")
    
    # 保存搜索历史
    with open(f"{search_dir}/search_history.json", 'w') as f:
        json.dump(search_history, f, indent=4)
    
    # 绘制搜索过程
    plot_search_history(agent.get_fitness_history())
    
    return best_config, best_fitness
def search_loss_function_rl(train_loader, val_loader, device):
    """使用强化学习搜索最佳损失函数"""
    print("开始使用强化学习进行损失函数搜索...")
    
    search_space = get_default_search_space()
    
    rl_agent = ReinforcementLossSearch(
        search_space=search_space,
        max_episodes=search_generations,  # 使用20代
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=0.9,
        learning_rate=0.001,
        batch_size=4,
        memory_size=200,
        update_target_freq=2
    )
    
    def train_fn(model, loss_config, train_loader, val_loader, device):
        return train_with_loss_config(
            model=model,
            loss_config=loss_config,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=validation_epochs
        )
    
    best_config, best_fitness = rl_agent.train(
        train_fn=train_fn,
        create_model_fn=create_model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device
    )
    
    return best_config, best_fitness
def search_loss_function(model, train_loader, val_loader, device):
    """根据选择的方法进行损失函数搜索"""
    print(f"选择的搜索方法: {args.method.upper()}")
    
    if args.method == 'rl':
        print("=== 启动强化学习搜索 ===")
        return search_loss_function_rl(train_loader, val_loader, device)
    else:
        print("=== 启动遗传算法搜索 ===")
        return search_loss_function_ga(model, train_loader, val_loader, device)
    # 定义绘制搜索历史的函数
def plot_search_history(fitness_history):
    """绘制搜索历史曲线"""
    generations = list(range(1, len(fitness_history) + 1))
    best_fitness = [h[0] for h in fitness_history]
    avg_fitness = [h[1] for h in fitness_history]
    
    plt.figure(figsize=(10, 6))
    plt.plot(generations, best_fitness, 'b-', label='最佳适应度')
    plt.plot(generations, avg_fitness, 'r-', label='平均适应度')
    plt.xlabel('代数')
    plt.ylabel('适应度 (验证损失)')
    plt.title('损失函数搜索历史')
    plt.legend()
    plt.grid(True)
    
    # 保存图像
    plt.savefig(f"{search_dir}/search_history.png")
    plt.close()

# 创建模型函数
def create_model(device):
    """创建并初始化模型"""
    model = RSANet(
        input_channels=1,
        start_neurons=start_neurons,
        keep_prob=keep_prob,
        block_size=block_size,
        use_saspp=args.use_saspp,
        use_damm=args.use_damm,
        use_dropblock=args.use_dropblock,
        use_gcnet=args.use_gcnet
    )
    model.to(device)
    return model

# 主训练流程
def main():
    print(f"调试信息: args.method = {args.method}")  # 添加这行
    print(f"调试信息: args.search_loss = {args.search_loss}")  # 添加这行
    # 加载数据
    train_files = os.listdir(training_images_loc)
    val_files = os.listdir(validate_images_loc)
    x_train, y_train = load_data(train_files, training_images_loc, training_label_loc)
    x_val, y_val = load_data(val_files, validate_images_loc, validate_label_loc)
    print(f"训练集: {x_train.shape}, 验证集: {x_val.shape}")
    
    # 创建数据加载器
    train_loader = DataLoader(ChaseDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(ChaseDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建模型
    model = create_model(device)
    
    # 损失函数和优化器
    if args.search_loss:
        # 开始损失函数搜索
        best_loss_config, best_fitness = search_loss_function(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device
        )
        
        print("\n自动损失函数搜索完成!")
        loss_fn = CombinedLoss(best_loss_config)
        print(f"最佳损失函数: {loss_fn}")
        print(f"验证损失: {best_fitness:.4f}")
        
        # 使用最佳损失函数训练最终模型
        print("\n使用最佳损失函数训练完整模型...")
        
    elif args.load_loss:
        # 加载预定义的损失函数配置
        print(f"加载损失函数配置: {args.load_loss}")
        loss_config = load_loss_config(args.load_loss)
        loss_fn = CombinedLoss(loss_config)
        print(f"使用损失函数: {loss_fn}")
        
    else:
        # 使用默认损失函数
        loss_fn = nn.BCELoss()
        print("使用默认损失函数: BCELoss")
    
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    writer = SummaryWriter(log_dir=log_dir)
    
    # 开始完整训练
    train_losses, val_losses = [], []
    best_val_loss = float('inf')
    
    for epoch in range(1, epochs + 1):
        # 训练
        model.train()
        epoch_loss = 0.0
        
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            outs = model(imgs)
            
            if args.search_loss or args.load_loss:
                loss, _ = loss_fn(outs, lbls)
            else:
                loss = loss_fn(outs, lbls)
                
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)
        
        # 验证
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                outs = model(imgs)
                
                if args.search_loss or args.load_loss:
                    loss, _ = loss_fn(outs, lbls)
                else:
                    loss = loss_fn(outs, lbls)
                    
                val_loss += loss.item()
        
        val_loss = val_loss / len(val_loader)
        val_losses.append(val_loss)
        
        # 记录到TensorBoard
        writer.add_scalar('Train/Loss', train_loss, epoch)
        writer.add_scalar('Val/Loss', val_loss, epoch)
        
        # 打印进度
        print(f"Epoch {epoch}/{epochs} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            os.makedirs(ckpt_dir, exist_ok=True)
            torch.save(model.state_dict(), f"{ckpt_dir}/best_model.pth")
            print(f"保存最佳模型 (验证损失: {best_val_loss:.4f})")
    
    writer.close()
    
    # 绘制损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs + 1), train_losses, 'b-', label='训练损失')
    plt.plot(range(1, epochs + 1), val_losses, 'r-', label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练与验证损失')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{loss_dir}/loss_curve.png")
    plt.close()
    
    print("\n训练完成!")
    print(f"最佳验证损失: {best_val_loss:.4f}")
    print(f"结果保存在: {results_dir}")

if __name__ == '__main__':
    main()