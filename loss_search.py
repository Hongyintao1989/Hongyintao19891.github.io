import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from collections import OrderedDict

class DiceLoss(nn.Module):
    """Dice损失函数实现"""
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # 展平预测和目标
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # 计算交集
        intersection = (predictions * targets).sum()
        
        # 计算Dice系数
        dice = (2.0 * intersection + self.smooth) / (
            predictions.sum() + targets.sum() + self.smooth
        )
        
        # 返回Dice损失
        return 1.0 - dice

class FocalLoss(nn.Module):
    """Focal损失函数实现"""
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        
    def forward(self, predictions, targets):
        # 展平预测和目标
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # 构建BCE损失
        bce = F.binary_cross_entropy(predictions, targets, reduction='none')
        
        # 计算权重
        p_t = predictions * targets + (1 - predictions) * (1 - targets)
        alpha_factor = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        modulating_factor = (1.0 - p_t) ** self.gamma
        
        # 应用权重并计算损失
        loss = alpha_factor * modulating_factor * bce
        
        return loss.mean()

class TverskyLoss(nn.Module):
    """Tversky损失函数实现"""
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0):
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, predictions, targets):
        # 展平预测和目标
        predictions = predictions.view(-1)
        targets = targets.view(-1)
        
        # 计算真阳性、假阳性和假阴性
        tp = (predictions * targets).sum()
        fp = ((1 - targets) * predictions).sum()
        fn = (targets * (1 - predictions)).sum()
        
        # 计算Tversky指数
        tversky = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        
        # 返回Tversky损失
        return 1.0 - tversky

class BCEWithLogitsLoss(nn.Module):
    """带有正样本权重的二元交叉熵损失"""
    def __init__(self, pos_weight=1.0):
        super(BCEWithLogitsLoss, self).__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions, targets):
        # 已经是概率了，不需要用logits版本
        # 手动实现带权重的BCE
        epsilon = 1e-7  # 防止log(0)
        
        predictions = torch.clamp(predictions, epsilon, 1.0 - epsilon)
        
        pos_loss = -targets * torch.log(predictions) * self.pos_weight
        neg_loss = -(1 - targets) * torch.log(1 - predictions)
        
        loss = pos_loss + neg_loss
        return loss.mean()

class CombinedLoss(nn.Module):
    """组合多种损失函数"""
    def __init__(self, loss_configs):
        """
        初始化组合损失函数
        :param loss_configs: 字典，包含损失函数类型、权重和参数
        例如: {
            'bce': {'weight': 1.0, 'params': {'pos_weight': 2.0}},
            'dice': {'weight': 0.5, 'params': {}},
            'focal': {'weight': 0.3, 'params': {'gamma': 2.0}},
            'tversky': {'weight': 0.2, 'params': {'alpha': 0.7, 'beta': 0.3}}
        }
        """
        super(CombinedLoss, self).__init__()
        self.losses = OrderedDict()
        self.weights = {}
        
        for loss_name, config in loss_configs.items():
            weight = config['weight']
            params = config['params']
            
            if weight <= 0:
                continue  # 跳过权重为0的损失函数
                
            if loss_name == 'bce':
                self.losses[loss_name] = BCEWithLogitsLoss(**params)
            elif loss_name == 'dice':
                self.losses[loss_name] = DiceLoss(**params)
            elif loss_name == 'focal':
                self.losses[loss_name] = FocalLoss(**params)
            elif loss_name == 'tversky':
                self.losses[loss_name] = TverskyLoss(**params)
            else:
                raise ValueError(f"不支持的损失函数类型: {loss_name}")
            
            self.weights[loss_name] = weight
    
    def forward(self, predictions, targets):
        """计算组合损失"""
        total_loss = 0.0
        loss_values = {}
        
        for loss_name, loss_fn in self.losses.items():
            loss_value = loss_fn(predictions, targets)
            weighted_loss = loss_value * self.weights[loss_name]
            total_loss += weighted_loss
            loss_values[loss_name] = loss_value.item()
        
        return total_loss, loss_values
    
    def __str__(self):
        """返回损失函数的字符串表示"""
        components = []
        for loss_name, loss_fn in self.losses.items():
            weight = self.weights[loss_name]
            
            if loss_name == 'bce':
                pos_weight = loss_fn.pos_weight
                components.append(f"{weight:.1f} * BCE损失(pos_weight={pos_weight:.1f})")
            elif loss_name == 'dice':
                components.append(f"{weight:.1f} * Dice损失")
            elif loss_name == 'focal':
                gamma = loss_fn.gamma
                components.append(f"{weight:.1f} * Focal损失(gamma={gamma:.1f})")
            elif loss_name == 'tversky':
                alpha, beta = loss_fn.alpha, loss_fn.beta
                components.append(f"{weight:.1f} * Tversky损失(alpha={alpha:.1f}, beta={beta:.1f})")
        
        return "最终损失 = " + " + ".join(components)

class LossSearchAgent:
    """使用强化学习或其他策略搜索最佳损失函数组合"""
    def __init__(self, 
                 search_space,
                 population_size=5,
                 mutation_rate=0.3,
                 crossover_rate=0.5,
                 elite_size=1):
        """
        初始化搜索代理
        :param search_space: 损失函数搜索空间
        :param population_size: 种群大小
        :param mutation_rate: 变异率
        :param crossover_rate: 交叉率
        :param elite_size: 精英数量
        """
        self.search_space = search_space
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_size = elite_size
        
        self.population = []
        self.fitness_history = []
        self.best_fitness = float('inf')
        self.best_individual = None
        self.generation = 0
        
        # 初始化种群
        self._initialize_population()
    
    def _initialize_population(self):
        """初始化随机种群"""
        self.population = []
        
        for _ in range(self.population_size):
            individual = {}
            
            # 为每种损失函数随机选择权重和参数
            for loss_name, param_ranges in self.search_space.items():
                weight_range = param_ranges['weight_range']
                
                # 随机选择是否使用该损失函数
                if random.random() < 0.7:  # 70%的概率使用该损失函数
                    weight = random.uniform(*weight_range)
                else:
                    weight = 0.0  # 不使用该损失函数
                
                params = {}
                for param_name, param_range in param_ranges['params'].items():
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                            params[param_name] = random.randint(*param_range)
                        else:
                            params[param_name] = random.uniform(*param_range)
                    elif isinstance(param_range, list):
                        params[param_name] = random.choice(param_range)
                    else:
                        params[param_name] = param_range
                
                individual[loss_name] = {
                    'weight': weight,
                    'params': params
                }
            
            # 确保至少有一个损失函数被启用
            while all(config['weight'] <= 0 for config in individual.values()):
                loss_name = random.choice(list(individual.keys()))
                individual[loss_name]['weight'] = random.uniform(
                    *self.search_space[loss_name]['weight_range']
                )
            
            # 归一化权重，使它们的总和为1
            self._normalize_weights(individual)
            
            self.population.append(individual)
    
    def _normalize_weights(self, individual):
        """归一化损失函数权重"""
        total_weight = sum(config['weight'] for config in individual.values() if config['weight'] > 0)
        
        if total_weight > 0:
            for loss_name in individual:
                if individual[loss_name]['weight'] > 0:
                    individual[loss_name]['weight'] /= total_weight
    
    def _mutate(self, individual):
        """变异操作"""
        mutated = individual.copy()
        
        for loss_name, config in mutated.items():
            # 以突变率的概率变异权重
            if random.random() < self.mutation_rate:
                if config['weight'] > 0:
                    # 80%的概率调整现有权重，20%的概率置0
                    if random.random() < 0.8:
                        weight_range = self.search_space[loss_name]['weight_range']
                        delta = random.uniform(-0.2, 0.2) * (weight_range[1] - weight_range[0])
                        new_weight = max(0, config['weight'] + delta)
                        mutated[loss_name]['weight'] = new_weight
                    else:
                        mutated[loss_name]['weight'] = 0.0
                else:
                    # 如果权重为0，有30%的概率激活它
                    if random.random() < 0.3:
                        weight_range = self.search_space[loss_name]['weight_range']
                        mutated[loss_name]['weight'] = random.uniform(*weight_range)
            
            # 变异参数
            for param_name, param_value in config['params'].items():
                if random.random() < self.mutation_rate:
                    param_range = self.search_space[loss_name]['params'][param_name]
                    
                    if isinstance(param_range, tuple) and len(param_range) == 2:
                        if isinstance(param_range[0], int) and isinstance(param_range[1], int):
                            mutated[loss_name]['params'][param_name] = random.randint(*param_range)
                        else:
                            mutated[loss_name]['params'][param_name] = random.uniform(*param_range)
                    elif isinstance(param_range, list):
                        mutated[loss_name]['params'][param_name] = random.choice(param_range)
        
        # 确保至少有一个损失函数被启用
        while all(config['weight'] <= 0 for config in mutated.values()):
            loss_name = random.choice(list(mutated.keys()))
            mutated[loss_name]['weight'] = random.uniform(
                *self.search_space[loss_name]['weight_range']
            )
        
        # 归一化权重
        self._normalize_weights(mutated)
        
        return mutated
    
    def _crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        child = {}
        
        for loss_name in parent1:
            # 50%的概率从parent1继承，50%的概率从parent2继承
            if random.random() < 0.5:
                child[loss_name] = parent1[loss_name].copy()
            else:
                child[loss_name] = parent2[loss_name].copy()
        
        # 确保至少有一个损失函数被启用
        while all(config['weight'] <= 0 for config in child.values()):
            loss_name = random.choice(list(child.keys()))
            child[loss_name]['weight'] = random.uniform(
                *self.search_space[loss_name]['weight_range']
            )
        
        # 归一化权重
        self._normalize_weights(child)
        
        return child
    
    def _select_parent(self, fitness_values):
        """使用轮盘赌选择父代"""
        # 转换为最小化问题的适应度值
        max_fitness = max(fitness_values)
        adjusted_fitness = [max_fitness - fitness for fitness in fitness_values]
        total_fitness = sum(adjusted_fitness)
        
        if total_fitness <= 0:
            # 如果所有适应度都相同，则随机选择
            return random.randint(0, len(fitness_values) - 1)
        
        # 轮盘赌选择
        pick = random.uniform(0, total_fitness)
        current = 0
        
        for i, fitness in enumerate(adjusted_fitness):
            current += fitness
            if current > pick:
                return i
        
        return len(fitness_values) - 1
    
    def evolve(self, fitness_values):
        """进化到下一代"""
        if len(fitness_values) != len(self.population):
            raise ValueError("适应度值数量与种群大小不匹配")
        
        # 记录适应度历史
        avg_fitness = sum(fitness_values) / len(fitness_values)
        self.fitness_history.append((min(fitness_values), avg_fitness))
        
        # 更新最佳个体
        best_idx = fitness_values.index(min(fitness_values))
        
        if fitness_values[best_idx] < self.best_fitness:
            self.best_fitness = fitness_values[best_idx]
            self.best_individual = self.population[best_idx].copy()
        
        # 精英保留
        elites = []
        elite_indices = sorted(range(len(fitness_values)), key=lambda i: fitness_values[i])[:self.elite_size]
        
        for idx in elite_indices:
            elites.append(self.population[idx].copy())
        
        # 创建新一代
        new_population = elites.copy()
        
        while len(new_population) < self.population_size:
            # 选择父代
            parent1_idx = self._select_parent(fitness_values)
            parent2_idx = self._select_parent(fitness_values)
            
            # 避免选择同一个父代
            while parent2_idx == parent1_idx and len(self.population) > 1:
                parent2_idx = self._select_parent(fitness_values)
            
            parent1 = self.population[parent1_idx]
            parent2 = self.population[parent2_idx]
            
            # 交叉和变异
            child = self._crossover(parent1, parent2)
            child = self._mutate(child)
            
            new_population.append(child)
        
        self.population = new_population
        self.generation += 1
        
        return self.population
    
    def get_best_individual(self):
        """获取最佳个体"""
        return self.best_individual, self.best_fitness
    
    def get_fitness_history(self):
        """获取适应度历史"""
        return self.fitness_history

# 搜索空间定义示例
def get_default_search_space():
    """获取默认的损失函数搜索空间"""
    return {
        'bce': {
            'weight_range': (0.0, 1.0),
            'params': {
                'pos_weight': (1.0, 20.0)
            }
        },
        'dice': {
            'weight_range': (0.0, 1.0),
            'params': {
                'smooth': 1.0  # 固定值
            }
        },
        'focal': {
            'weight_range': (0.0, 1.0),
            'params': {
                'gamma': (1.0, 5.0),
                'alpha': (0.1, 0.9)
            }
        },
        'tversky': {
            'weight_range': (0.0, 1.0),
            'params': {
                'alpha': (0.1, 0.9),
                'beta': (0.1, 0.9),
                'smooth': 1.0  # 固定值
            }
        }
    }