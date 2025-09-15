import os
import sys
import argparse
import numpy as np
import torch
from sklearn.metrics import recall_score, roc_auc_score, accuracy_score, confusion_matrix, f1_score
import cv2
from util import crop_to_shape
from model import RSANet

# 设置环境变量以避免OpenMP错误
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# 命令行参数解析
parser = argparse.ArgumentParser(description='评估RSANet模型')
parser.add_argument('--use_saspp', action='store_true', help='启用SASPP模块')
parser.add_argument('--use_damm', action='store_true', help='启用DAMM(方向感知混合注意力)模块') 
parser.add_argument('--use_dropblock', action='store_true', help='启用DropBlock模块')
parser.add_argument('--weight_path', type=str, default="Chase/Model/RSAN.pth", help='模型权重路径')
parser.add_argument('--use_gcnet', action='store_true', help='启用GCNet模块')
parser.add_argument('--use_damm_spa', action='store_true', help='启用DAMM的简单像素注意力分支')
parser.add_argument('--use_damm_ca', action='store_true', help='启用DAMM的通道注意力分支')
parser.add_argument('--use_damm_pa', action='store_true', help='启用DAMM的像素注意力分支')
parser.add_argument('--use_damm_dir', action='store_true', help='启用DAMM的方向感知分支')
parser.add_argument('--loss_config', type=str, default=None, help='损失函数配置文件路径')
# 添加阈值搜索相关参数
parser.add_argument('--threshold_search', action='store_true', default=True, help='启用阈值搜索')
parser.add_argument('--no_threshold_search', action='store_true', help='禁用阈值搜索，使用固定阈值0.5')
parser.add_argument('--threshold_start', type=float, default=0.1, help='阈值搜索起始值')
parser.add_argument('--threshold_end', type=float, default=0.9, help='阈值搜索结束值')
parser.add_argument('--threshold_step', type=float, default=0.05, help='阈值搜索步长')
parser.add_argument('--optimize_metric', type=str, default='f1', 
                   choices=['f1', 'sensitivity', 'specificity', 'accuracy'],
                   help='用于优化的指标')
args = parser.parse_args()

# 全局配置参数
CONFIG = {
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'desired_size': 1008,
    'batch_size': 1,
    'weight_path': args.weight_path,
    'use_saspp': args.use_saspp,
    'use_damm': args.use_damm,
    'use_dropblock': args.use_dropblock,
    'use_gcnet': args.use_gcnet,
    # 阈值搜索配置
    'threshold_search': args.threshold_search and not args.no_threshold_search,
    'threshold_start': args.threshold_start,
    'threshold_end': args.threshold_end,
    'threshold_step': args.threshold_step,
    'optimize_metric': args.optimize_metric,
    'model_params': {
        'input_channels': 1,
        'start_neurons': 16,
        'keep_prob': 0.87,
        'block_size': 7
    },
    'paths': {
        'data_location': '',
        'result_dir': './Chase/test/result/result-eval',
        'visual_dir': './Chase/test/visualization/visualization-eval',
        'eval_result_file': './Chase/test/evaluation_results_eval.txt'
    }
}

# 生成模型后缀
def generate_model_suffix():
    """动态生成模型后缀，基于启用的模块"""
    modules = []
    if CONFIG['use_saspp']:
        modules.append("saspp")
    if CONFIG['use_damm']:
        modules.append("damm")
    if CONFIG['use_dropblock']:
        modules.append("dropblock")
    if CONFIG['use_gcnet']:
        modules.append("gcnet")
    
    if modules:
        return "_" + "_".join(modules)
    else:
        return ""

# 使用新的函数生成后缀
model_suffix = generate_model_suffix()

# 更新输出路径
CONFIG['paths']['result_dir'] = f"./Chase/test/result/result-eval{model_suffix}"
CONFIG['paths']['visual_dir'] = f"./Chase/test/visualization/visualization-eval{model_suffix}"
CONFIG['paths']['eval_result_file'] = f"./Chase/test/evaluation_results_eval{model_suffix}.txt"
# 设置派生路径
CONFIG['paths']['orig_images_loc'] = CONFIG['paths']['data_location'] + 'Chase/test/im/'
CONFIG['paths']['testing_images_loc'] = CONFIG['paths']['data_location'] + 'Chase/test/image/'
CONFIG['paths']['testing_label_loc'] = CONFIG['paths']['data_location'] + 'Chase/test/label/'

# 打印使用的模块信息
modules_used = []
if CONFIG['use_saspp']:
    modules_used.append("SASPP")
if CONFIG['use_damm']:
    modules_used.append("DAMM")
if args.use_dropblock:
    modules_used.append("DropBlock")
if args.use_gcnet:
    modules_used.append("GCNet")

print(f"评估模型: RSANet {'使用 ' + ' + '.join(modules_used) if modules_used else '无增强模块'}")
print(f"使用权重文件: {CONFIG['weight_path']}")
print(f"阈值搜索: {'启用' if CONFIG['threshold_search'] else '禁用'}")
if CONFIG['threshold_search']:
    print(f"阈值范围: {CONFIG['threshold_start']:.2f} - {CONFIG['threshold_end']:.2f}, 步长: {CONFIG['threshold_step']:.3f}")
    print(f"优化指标: {CONFIG['optimize_metric']}")
print(f"评估结果将保存到: {CONFIG['paths']['eval_result_file']}")
print(f"可视化结果将保存到: {CONFIG['paths']['visual_dir']}")


def setup_directories():
    """创建必要的目录"""
    os.makedirs(CONFIG['paths']['result_dir'], exist_ok=True)
    os.makedirs(CONFIG['paths']['visual_dir'], exist_ok=True)


def load_test_data():
    """加载和预处理测试数据"""
    test_files = os.listdir(CONFIG['paths']['testing_images_loc'])
    test_data = []
    test_label = []
    file_indices = {}
    
    for i, file_name in enumerate(test_files):
        print(f"处理图像 {i+1}/{len(test_files)}: {file_name}")
        file_indices[i] = file_name
        
        # 读取图像和标签
        im = cv2.imread(CONFIG['paths']['testing_images_loc'] + file_name, cv2.IMREAD_GRAYSCALE)
        label_name = f"Image_{file_name.split('_')[1].split('.')[0]}_1stHO.png"
        label = cv2.imread(CONFIG['paths']['testing_label_loc'] + label_name, cv2.IMREAD_GRAYSCALE)
        
        # 计算需要填充的尺寸
        old_size = im.shape[:2]
        delta_w = CONFIG['desired_size'] - old_size[1]
        delta_h = CONFIG['desired_size'] - old_size[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        
        # 填充图像
        new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        new_label = cv2.copyMakeBorder(label, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
        
        # 调整大小
        resized_im = cv2.resize(new_im, (CONFIG['desired_size'], CONFIG['desired_size']))
        temp = cv2.resize(new_label, (CONFIG['desired_size'], CONFIG['desired_size']))
        _, temp = cv2.threshold(temp, 127, 255, cv2.THRESH_BINARY)
        
        test_data.append(resized_im)
        test_label.append(temp)
    
    # 转换为NumPy数组
    test_data_np = np.array(test_data)
    test_label_np = np.array(test_label)
    
    # 归一化并调整数据维度
    x_test = test_data_np.astype('float32') / 255.
    y_test = test_label_np.astype('float32') / 255.
    x_test = np.reshape(x_test, (len(x_test), 1, CONFIG['desired_size'], CONFIG['desired_size']))
    y_test = np.reshape(y_test, (len(y_test), 1, CONFIG['desired_size'], CONFIG['desired_size']))
    
    return test_data_np, x_test, y_test, file_indices


def load_model():
    """加载模型"""
    model = RSANet(input_channels=CONFIG['model_params']['input_channels'],
                   start_neurons=CONFIG['model_params']['start_neurons'],
                   keep_prob=CONFIG['model_params']['keep_prob'],
                   block_size=CONFIG['model_params']['block_size'],
                   use_saspp=CONFIG['use_saspp'],
                   use_damm=CONFIG['use_damm'],
                   use_dropblock=CONFIG['use_dropblock'],
                   use_gcnet=CONFIG['use_gcnet'])
    if os.path.isfile(CONFIG['weight_path']):
        try:
            model.load_state_dict(torch.load(CONFIG['weight_path'], map_location=CONFIG['device']))
            print(f"加载模型权重: {CONFIG['weight_path']}")
        except Exception as e:
            print(f"警告: 加载权重时出错 - {e}")
            # 使用非严格模式加载可用的参数
            model.load_state_dict(torch.load(CONFIG['weight_path'], map_location=CONFIG['device']), strict=False)
            print("已使用非严格模式加载部分权重")
    else:
        print(f"警告: 找不到权重文件 {CONFIG['weight_path']}")
        alt_paths = ["Chase/Model/RSAN.pth"]
        for alt_path in alt_paths:
            if os.path.isfile(alt_path):
                model.load_state_dict(torch.load(alt_path, map_location=CONFIG['device']), strict=False)
                print(f"加载替代模型权重: {alt_path}")
                break
        else:
            print("无法找到任何模型权重文件，使用随机初始化的模型")
    
    model.to(CONFIG['device'])
    model.eval()
    return model


def predict(model, x_test):
    """执行预测"""
    x_test_tensor = torch.from_numpy(x_test).to(CONFIG['device'])
    y_pred_list = []
    
    print("\n开始模型预测...")
    with torch.no_grad():
        for i in range(0, len(x_test_tensor), CONFIG['batch_size']):
            batch_end = min(i + CONFIG['batch_size'], len(x_test_tensor))
            input_batch = x_test_tensor[i:batch_end]
            output = model(input_batch)
            y_pred_list.append(output.cpu().numpy())
            print(f"已处理 {batch_end}/{len(x_test_tensor)} 个样本", end="\r")
    
    print("\n预测完成！")
    return np.concatenate(y_pred_list, axis=0)


def calculate_metrics(y_true_flat, y_pred_thresh_flat, y_pred_prob_flat=None):
    """计算所有评估指标"""
    try:
        tn, fp, fn, tp = confusion_matrix(y_true_flat, y_pred_thresh_flat).ravel()
        
        metrics = {
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'f1': 2*tp/(2*tp+fn+fp) if (2*tp+fn+fp) > 0 else 0,
            'accuracy': (tp + tn) / (tp + tn + fp + fn),
            'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
            'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn
        }
        
        # 如果提供了概率预测，计算AUC
        if y_pred_prob_flat is not None:
            try:
                metrics['auc'] = roc_auc_score(y_true_flat, y_pred_prob_flat)
            except:
                metrics['auc'] = 0.0
        else:
            metrics['auc'] = 0.0
            
        return metrics
    except Exception as e:
        print(f"计算指标时出错: {e}")
        return None


def threshold_search(y_test_cropped, y_pred_cropped):
    """执行阈值搜索"""
    print("\n开始阈值搜索...")
    
    # 扁平化数据
    y_test_flat = y_test_cropped.flatten()
    y_pred_flat = y_pred_cropped.flatten()
    
    # 生成阈值列表
    thresholds = np.arange(CONFIG['threshold_start'], 
                          CONFIG['threshold_end'] + CONFIG['threshold_step'], 
                          CONFIG['threshold_step'])
    
    best_threshold = 0.5
    best_score = 0
    best_metrics = None
    threshold_results = []
    
    print(f"测试 {len(thresholds)} 个阈值...")
    
    for i, threshold in enumerate(thresholds):
        # 应用阈值
        y_pred_thresh = (y_pred_flat > threshold).astype(np.float32)
        
        # 计算指标
        metrics = calculate_metrics(y_test_flat, y_pred_thresh, y_pred_flat)
        
        if metrics is not None:
            # 获取优化指标的值
            current_score = metrics[CONFIG['optimize_metric']]
            threshold_results.append({
                'threshold': threshold,
                'metrics': metrics,
                'score': current_score
            })
            
            # 更新最佳阈值
            if current_score > best_score:
                best_score = current_score
                best_threshold = threshold
                best_metrics = metrics
            
            print(f"阈值 {threshold:.3f}: {CONFIG['optimize_metric']} = {current_score:.4f}", end="\r")
    
    print(f"\n阈值搜索完成！")
    print(f"最佳阈值: {best_threshold:.3f}")
    print(f"最佳 {CONFIG['optimize_metric']}: {best_score:.4f}")
    
    return best_threshold, best_metrics, threshold_results


def process_results(test_data, y_test, y_pred, file_indices):
    """处理预测结果，创建可视化并计算评估指标"""
    # 获取裁剪形状
    new_shape = (len(y_test), 960, 999, 1)
    
    # 调整数据格式并裁剪
    y_pred_numpy = y_pred.transpose(0, 2, 3, 1)
    y_test_numpy = y_test.transpose(0, 2, 3, 1)
    
    y_pred_cropped = crop_to_shape(y_pred_numpy, new_shape)
    y_test_cropped = crop_to_shape(y_test_numpy, new_shape)
    
    # 裁剪测试数据
    test_data_cropped = []
    for i in range(len(test_data)):
        img_reshaped = np.reshape(test_data[i], (1, test_data[i].shape[0], test_data[i].shape[1], 1))
        img_cropped = crop_to_shape(img_reshaped, (1, new_shape[1], new_shape[2], 1))
        test_data_cropped.append(img_cropped[0, :, :, 0])
    
    print(f"裁剪后预测结果形状: {y_pred_cropped.shape}")
    print(f"裁剪后标签形状: {y_test_cropped.shape}")
    print(f"裁剪后灰度图形状: {test_data_cropped[0].shape}")
    
    # 阈值搜索或使用固定阈值
    if CONFIG['threshold_search']:
        optimal_threshold, best_metrics, threshold_results = threshold_search(y_test_cropped, y_pred_cropped)
    else:
        optimal_threshold = 0.5
        print(f"\n使用固定阈值: {optimal_threshold}")
        
        # 计算固定阈值的指标
        y_test_flat = y_test_cropped.flatten()
        y_pred_flat = y_pred_cropped.flatten()
        y_pred_thresh = (y_pred_flat > optimal_threshold).astype(np.float32)
        best_metrics = calculate_metrics(y_test_flat, y_pred_thresh, y_pred_flat)
        threshold_results = []
    
    # 使用最佳阈值生成最终结果
    y_pred_threshold = (y_pred_cropped > optimal_threshold).astype(np.float32)
    
    # 生成可视化
    print("\n创建可视化结果...")
    create_visualizations(test_data, test_data_cropped, y_pred_cropped, y_test_cropped, 
                          y_pred_threshold, file_indices, optimal_threshold)
    
    # 保存评估结果
    save_evaluation_results(best_metrics, optimal_threshold, threshold_results)
    
    return y_test_cropped, y_pred_cropped, y_pred_threshold, optimal_threshold


def create_error_map(gt_mask, pred_mask):
    """创建误差热图"""
    h, w = gt_mask.shape
    
    # 计算TP, FP, FN (转为0-1值)
    tp = (gt_mask > 0.5) & (pred_mask > 0.5)
    fp = (gt_mask <= 0.5) & (pred_mask > 0.5)
    fn = (gt_mask > 0.5) & (pred_mask <= 0.5)
    tn = (gt_mask <= 0.5) & (pred_mask <= 0.5)
    
    # 创建误差图
    error_map = np.zeros_like(gt_mask)
    error_map[tp] = 2  # 真阳性
    error_map[fp] = 3  # 假阳性
    error_map[fn] = 1  # 假阴性
    # 背景(tn)保持为0
    
    # 给每个区域上色
    error_map_rgb = np.zeros((h, w, 3), dtype=np.uint8)
    error_map_rgb[error_map == 0] = [255, 255, 255]  # 白色背景
    error_map_rgb[error_map == 1] = [255, 0, 0]      # 蓝色假阴性(OpenCV BGR格式)
    error_map_rgb[error_map == 2] = [0, 255, 255]    # 黄色真阳性(OpenCV BGR格式)
    error_map_rgb[error_map == 3] = [0, 0, 255]      # 红色假阳性(OpenCV BGR格式)
    
    # 计算各指标数量
    n_tp = np.sum(tp)
    n_fp = np.sum(fp)
    n_fn = np.sum(fn)
    n_tn = np.sum(tn)
    
    # 计算指标
    precision = n_tp / (n_tp + n_fp) if (n_tp + n_fp) > 0 else 0
    recall = n_tp / (n_tp + n_fn) if (n_tp + n_fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (n_tp + n_tn) / (n_tp + n_tn + n_fp + n_fn)
    
    # 返回误差热图和所有指标
    return error_map_rgb, {
        'n_tp': n_tp,
        'n_fp': n_fp,
        'n_fn': n_fn,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy
    }


def create_visualizations(test_data, test_data_cropped, y_pred_cropped, y_test_cropped, 
                          y_pred_threshold, file_indices, optimal_threshold):
    """创建和保存可视化结果"""
    for i, y in enumerate(y_pred_cropped):
        current_file = file_indices[i]
        
        # 保存预测结果图像
        y_img = (y * 255).astype(np.uint8)
        result_path = f"{CONFIG['paths']['result_dir']}/{i}.png"
        cv2.imwrite(result_path, y_img)
        
        # 获取原始图像
        orig_img = get_original_image(current_file, test_data[i])
        
        # 创建可视化画布
        h, w = y_pred_cropped[i].shape[:2]
        orig_img = cv2.resize(orig_img, (w, h), interpolation=cv2.INTER_AREA)
        
        # 准备各个可视化部分
        gt_img = create_binary_image(y_test_cropped[i, :, :, 0])
        pred_img = create_binary_image(y_pred_threshold[i, :, :, 0])
        overlay_img = create_overlay_image(test_data_cropped[i], 
                                          y_test_cropped[i, :, :, 0], 
                                          y_pred_threshold[i, :, :, 0])
        
        # 创建误差热图
        error_map, metrics = create_error_map(y_test_cropped[i, :, :, 0], 
                                            y_pred_threshold[i, :, :, 0])
        
        # 组合画布
        canvas = combine_visualizations(orig_img, gt_img, pred_img, overlay_img, 
                                       error_map, metrics, optimal_threshold)
        
        # 保存可视化结果
        vis_output_path = f"{CONFIG['paths']['visual_dir']}/{i}_{current_file.split('.')[0]}_comparison.png"
        cv2.imwrite(vis_output_path, canvas)
        print(f"已保存可视化结果到: {vis_output_path}")


def get_original_image(current_file, default_img):
    """获取原始图像或使用灰度图像代替"""
    base_name = f"Image_{current_file.split('_')[1].split('.')[0]}"
    orig_img_path = os.path.join(CONFIG['paths']['orig_images_loc'], f"{base_name}.png")
    
    if not os.path.exists(orig_img_path):
        for ext in ['.jpg', '.JPG', '.png', '.PNG']:
            alt_path = os.path.join(CONFIG['paths']['orig_images_loc'], f"{base_name}{ext}")
            if os.path.exists(alt_path):
                orig_img_path = alt_path
                break
    
    if os.path.exists(orig_img_path):
        orig_img = cv2.imread(orig_img_path)
        print(f"读取原始图像: {orig_img_path}")
    else:
        print(f"未找到原始图像: {orig_img_path}，使用灰度图像代替")
        orig_img = cv2.cvtColor(default_img, cv2.COLOR_GRAY2BGR)
    
    return orig_img


def create_binary_image(mask):
    """创建二值图像"""
    h, w = mask.shape
    binary_img = np.zeros((h, w, 3), dtype=np.uint8)
    binary_img[mask > 0.5] = [255, 255, 255]
    return binary_img


def create_overlay_image(gray_img, gt_mask, pred_mask):
    """创建叠加图像"""
    # 一定要先将灰度图转为彩色图，保持其背景
    overlay_img = cv2.cvtColor(gray_img.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    
    # 在血管区域添加颜色
    overlay_img[gt_mask > 0.5, 1] = 255  # 绿色通道表示GT
    overlay_img[pred_mask > 0.5, 2] = 255  # 红色通道表示预测
    
    return overlay_img


def combine_visualizations(orig_img, gt_img, pred_img, overlay_img, error_map=None, 
                          metrics=None, threshold=0.5):
    """组合可视化结果到一个画布，包含误差热图和阈值信息"""
    h, w = orig_img.shape[:2]
    title_h = 30
    
    # 包含误差热图，创建5列布局
    metrics_h = 60  # 增加高度以显示更多信息
    total_h = h + title_h + metrics_h
    canvas = np.ones((total_h, w * 5, 3), dtype=np.uint8) * 255
    
    # 添加标题
    titles = [
        ('Original Image', w//2 - 80),
        ('Ground Truth', w + w//2 - 80),
        (f'Prediction (T={threshold:.3f})', 2*w + w//2 - 100),
        ('Overlay (GT-green / Pred-red)', 3*w + w//2 - 140),
        ('Error Map', 4*w + w//2 - 60)
    ]
    
    for title, pos_x in titles:
        cv2.putText(canvas, title, (pos_x, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
    
    # 放置图像
    canvas[title_h:title_h+h, 0:w] = orig_img
    canvas[title_h:title_h+h, w:2*w] = gt_img
    canvas[title_h:title_h+h, 2*w:3*w] = pred_img
    canvas[title_h:title_h+h, 3*w:4*w] = overlay_img
    if error_map is not None:
        canvas[title_h:title_h+h, 4*w:5*w] = error_map
    
    # 添加指标信息
    if metrics:
        # 第一行：主要指标
        metrics_text1 = f"Precision: {metrics['precision']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1']:.4f}, Accuracy: {metrics['accuracy']:.4f}"
        cv2.putText(canvas, metrics_text1, (10, h + title_h + 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        # 第二行：误差统计
        metrics_text2 = f"TP: {metrics['n_tp']}, FP: {metrics['n_fp']}, FN: {metrics['n_fn']}"
        cv2.putText(canvas, metrics_text2, (10, h + title_h + 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)
        
        # 添加颜色图例 (第三行)
        legend_y = h + title_h + 55
        legend_items = [
            ([255,0,0], "FN", 10),
            ([0,255,255], "TP", 80),
            ([0,0,255], "FP", 150)
        ]
        
        for color, label, x_pos in legend_items:
            cv2.rectangle(canvas, (x_pos, legend_y-10), (x_pos+15, legend_y), color, -1)
            cv2.putText(canvas, label, (x_pos+20, legend_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
    
    return canvas


def save_evaluation_results(metrics, optimal_threshold, threshold_results):
    """保存评估结果到文件"""
    # 生成模型名称
    modules_used = []
    if CONFIG['use_saspp']:
        modules_used.append("SASPP")
    if CONFIG['use_damm']:
        modules_used.append("DAMM")
    if CONFIG['use_dropblock']:
        modules_used.append("DropBlock")
    if CONFIG['use_gcnet']:
        modules_used.append("GCNet")
    
    model_name = f"RSANet {'使用 ' + ' + '.join(modules_used) if modules_used else '无增强模块'}"
    
    with open(CONFIG['paths']['eval_result_file'], 'w', encoding='utf-8') as f:
        f.write(f'评估模型: {model_name}\n')
        f.write(f'权重文件: {CONFIG["weight_path"]}\n')
        f.write('-' * 50 + '\n')
        
        # 阈值搜索信息
        if CONFIG['threshold_search']:
            f.write('阈值搜索结果:\n')
            f.write(f'搜索范围: {CONFIG["threshold_start"]:.3f} - {CONFIG["threshold_end"]:.3f}\n')
            f.write(f'搜索步长: {CONFIG["threshold_step"]:.3f}\n')
            f.write(f'优化指标: {CONFIG["optimize_metric"]}\n')
            f.write(f'最佳阈值: {optimal_threshold:.3f}\n')
        else:
            f.write(f'使用固定阈值: {optimal_threshold:.3f}\n')
        
        f.write('-' * 50 + '\n')
        
        # 最佳结果
        if metrics:
            f.write('最佳结果:\n')
            f.write(f'Sensitivity (敏感度/召回率): {metrics["sensitivity"]:.4f}\n')
            f.write(f'Specificity (特异度): {metrics["specificity"]:.4f}\n')
            f.write(f'Precision (精确度): {metrics["precision"]:.4f}\n')
            f.write(f'F1 Score: {metrics["f1"]:.4f}\n')
            f.write(f'Accuracy (准确率): {metrics["accuracy"]:.4f}\n')
            f.write(f'AUC: {metrics["auc"]:.4f}\n')
            f.write(f'TP: {metrics["tp"]}, TN: {metrics["tn"]}, FP: {metrics["fp"]}, FN: {metrics["fn"]}\n')
        
        # 阈值搜索详细结果
        if CONFIG['threshold_search'] and threshold_results:
            f.write('\n' + '-' * 50 + '\n')
            f.write('阈值搜索详细结果:\n')
            f.write('Threshold\tSensitivity\tSpecificity\tPrecision\tF1\t\tAccuracy\tAUC\n')
            f.write('-' * 80 + '\n')
            
            for result in threshold_results:
                thresh = result['threshold']
                m = result['metrics']
                f.write(f'{thresh:.3f}\t\t{m["sensitivity"]:.4f}\t\t{m["specificity"]:.4f}\t\t'
                       f'{m["precision"]:.4f}\t\t{m["f1"]:.4f}\t\t{m["accuracy"]:.4f}\t\t{m["auc"]:.4f}\n')
    
    print(f"评估结果已保存到: {CONFIG['paths']['eval_result_file']}")
    
    # 打印到控制台
    print('\n' + '='*60)
    print('最终评估结果:')
    print('='*60)
    if CONFIG['threshold_search']:
        print(f'最佳阈值: {optimal_threshold:.3f} (基于 {CONFIG["optimize_metric"]})')
    else:
        print(f'使用阈值: {optimal_threshold:.3f}')
    
    if metrics:
        print(f'Sensitivity (敏感度/召回率): {metrics["sensitivity"]:.4f}')
        print(f'Specificity (特异度): {metrics["specificity"]:.4f}')
        print(f'Precision (精确度): {metrics["precision"]:.4f}')
        print(f'F1 Score: {metrics["f1"]:.4f}')
        print(f'Accuracy (准确率): {metrics["accuracy"]:.4f}')
        print(f'AUC: {metrics["auc"]:.4f}')
        print(f'TP: {metrics["tp"]}, TN: {metrics["tn"]}, FP: {metrics["fp"]}, FN: {metrics["fn"]}')
    print('='*60)


def print_threshold_search_summary(threshold_results, optimal_threshold, optimize_metric):
    """打印阈值搜索摘要"""
    if not threshold_results:
        return
    
    print(f"\n阈值搜索摘要 (基于 {optimize_metric}):")
    print("-" * 80)
    print(f"{'Threshold':<10} {'Sensitivity':<12} {'Specificity':<12} {'Precision':<10} {'F1':<8} {'Accuracy':<10}")
    print("-" * 80)
    
    # 显示前5个和后5个结果，以及最佳结果
    sorted_results = sorted(threshold_results, key=lambda x: x['score'], reverse=True)
    
    # 显示最佳结果
    best_result = sorted_results[0]
    print(f"{'★' + str(best_result['threshold']):<9.3f} "
          f"{best_result['metrics']['sensitivity']:<12.4f} "
          f"{best_result['metrics']['specificity']:<12.4f} "
          f"{best_result['metrics']['precision']:<10.4f} "
          f"{best_result['metrics']['f1']:<8.4f} "
          f"{best_result['metrics']['accuracy']:<10.4f} ← 最佳")
    
    # 显示其他排名靠前的结果
    for i, result in enumerate(sorted_results[1:6]):  # 显示前5个（除了最佳的）
        m = result['metrics']
        print(f"{result['threshold']:<10.3f} {m['sensitivity']:<12.4f} "
              f"{m['specificity']:<12.4f} {m['precision']:<10.4f} "
              f"{m['f1']:<8.4f} {m['accuracy']:<10.4f}")
    
    print("-" * 80)


def main():
    """主函数"""
    print(f"使用设备: {CONFIG['device']}")
    
    # 设置目录
    setup_directories()
    
    # 加载数据
    test_data, x_test, y_test, file_indices = load_test_data()
    
    # 加载模型
    model = load_model()
    
    # 执行预测
    y_pred = predict(model, x_test)
    
    # 处理结果（包含阈值搜索）
    y_test_cropped, y_pred_cropped, y_pred_threshold, optimal_threshold = process_results(
        test_data, y_test, y_pred, file_indices)
    
    print("\n评估完成！")


if __name__ == "__main__":
    main()