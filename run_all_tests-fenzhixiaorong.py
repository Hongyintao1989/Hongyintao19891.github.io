import os
import time
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.font_manager as fm
import hashlib
import shutil

# Set matplotlib to use English font
plt.rcParams['font.family'] = 'DejaVu Sans'

# Create a log file and output to both terminal and file
class Tee:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.file = open(filename, 'w', encoding='utf-8')
        
    def write(self, message):
        self.terminal.write(message)
        self.file.write(message)
        # Ensure content is written immediately to file and terminal
        self.terminal.flush()
        self.file.flush()
        
    def flush(self):
        self.terminal.flush()
        self.file.flush()
        
    def close(self):
        self.file.close()

def run_command(command):
    """Run command and display output in real-time"""
    return os.system(command)

def run_eval_config(config_name, command, output_dir):
    """Run the given evaluation configuration and display progress"""
    print("="*80)
    print(f"Starting evaluation for configuration: {config_name}")
    print("-"*60)
    
    # Create directory for this specific configuration's results
    config_output_dir = os.path.join(output_dir, config_name.replace(" ", "_"))
    os.makedirs(config_output_dir, exist_ok=True)
    
    # Modify eval_chase.py to write results to our output directory
    # Since we can't use --output_suffix, we'll handle file management ourselves
    return_code = run_command(command)
    
    if return_code == 0:
        print(f"\nConfiguration {config_name} evaluation completed successfully!")
        
        # Find and move the evaluation result files to our directory
        base_path = "./Chase/test/"
        result_files = [f for f in os.listdir(base_path) if f.startswith("evaluation_results_eval")]
        
        if result_files:
            # Sort by modification time (newest first)
            result_files.sort(key=lambda x: os.path.getmtime(os.path.join(base_path, x)), reverse=True)
            # Get the most recent file
            latest_file = result_files[0]
            source_file = os.path.join(base_path, latest_file)
            
            # Generate a unique name for the destination file
            config_hash = hashlib.md5(config_name.encode()).hexdigest()[:8]
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dest_filename = f"evaluation_results_{config_name.replace(' ', '_')}_{config_hash}_{timestamp}.txt"
            dest_file = os.path.join(config_output_dir, dest_filename)
            
            # Copy the file
            shutil.copy2(source_file, dest_file)
            print(f"Copied result file to: {dest_file}")
            
            # Return the path to the copied file
            return return_code, dest_file
        else:
            print("No evaluation result files found!")
            return return_code, None
    else:
        print(f"\nConfiguration {config_name} evaluation failed, return code: {return_code}")
        return return_code, None

def parse_eval_results(result_file):
    """Parse evaluation result file"""
    metrics = {}
    try:
        with open(result_file, 'r', encoding='utf-8') as f:
            for line in f:
                if ': ' in line:
                    key, value = line.strip().split(': ')
                    # Try to convert value to float
                    try:
                        metrics[key] = float(value)
                    except ValueError:
                        metrics[key] = value
    except Exception as e:
        print(f"Error parsing evaluation result file: {e}")
    
    return metrics

def visualize_results(results_df, config_name_mapping=None, output_dir='damm_ablation_results'):
    """Generate visualization for results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Only keep performance metric data
    metrics_columns = ['Sensitivity', 'Specificity', 'F1_Score', 'Accuracy', 'AUC']
    if set(metrics_columns).issubset(results_df.columns):
        metrics_df = results_df[metrics_columns]
    else:
        # Try to match possible column names
        available_columns = results_df.columns
        mapping = {}
        for metric in metrics_columns:
            matches = [col for col in available_columns if metric.lower() in col.lower()]
            if matches:
                mapping[metric] = matches[0]
        
        if mapping:
            metrics_df = results_df[[mapping.get(col, col) for col in metrics_columns if col in mapping]]
        else:
            print("Warning: Cannot find performance metric columns! Using all available columns for visualization.")
            metrics_df = results_df
    
    # Translate configuration names if mapping is provided
    if config_name_mapping:
        metrics_df.index = [config_name_mapping.get(idx, idx) for idx in metrics_df.index]
    
    # Create individual bar charts for each metric
    plt.figure(figsize=(20, 12))
    for i, metric in enumerate(metrics_df.columns):
        plt.subplot(len(metrics_df.columns), 1, i+1)
        ax = metrics_df[metric].plot(kind='bar', figsize=(20, 3), color='#3498db')
        plt.title(f'{metric} Comparison', fontsize=14)
        plt.ylabel(metric, fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Add value labels for each bar
        for j, val in enumerate(metrics_df[metric]):
            plt.text(j, val, f'{val:.4f}', ha='center', va='bottom', fontsize=9)
        
        # Only show configuration names on the last subplot
        if i == len(metrics_df.columns) - 1:
            plt.xlabel('Model Configuration', fontsize=12)
            plt.xticks(rotation=45, ha='right')
        else:
            plt.xticks([])
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'damm_ablation_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create radar chart for performance metrics
    categories = metrics_df.columns
    N = len(categories)
    
    # Assign different colors for different configuration groups
    config_groups = {
        'Full DAMM': '#2ecc71',           # Green
        'DAMM without': '#3498db',        # Blue
        'DAMM using two branches': '#e74c3c',  # Red
        'DAMM using single branch': '#f39c12',  # Orange
        'DAMM empty shell': '#9b59b6',    # Purple
        'Baseline model': '#95a5a6'       # Gray
    }
    
    # Group configurations appropriately
    config_types = {}
    for config in metrics_df.index:
        if config == 'Full DAMM':
            config_types[config] = 'Full DAMM'
        elif config.startswith('DAMM without'):
            config_types[config] = 'DAMM without'
        elif config.startswith('DAMM using') and '+' in config:
            config_types[config] = 'DAMM using two branches'
        elif config.startswith('DAMM using') and '+' not in config:
            config_types[config] = 'DAMM using single branch'
        elif config == 'DAMM empty shell':
            config_types[config] = 'DAMM empty shell'
        else:
            config_types[config] = 'Baseline model'
    
    # Create radar chart
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close the radar chart
    
    # Split radar chart into two parts to avoid crowding
    # 1. Full DAMM, removed single branch and baseline model
    group1_configs = [config for config, group in config_types.items() 
                     if group in ['Full DAMM', 'DAMM without', 'Baseline model']]
    
    # 2. Two-branch and single-branch combinations
    group2_configs = [config for config, group in config_types.items() 
                      if group in ['DAMM using two branches', 'DAMM using single branch', 'DAMM empty shell']]
    
    # Draw first group radar chart
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, polar=True)
    
    # Set y-axis limits to focus on the 0.75-1 range
    ax.set_ylim(0.75, 1.0)
    yticks = np.linspace(0.75, 1.0, 6)
    ax.set_yticks(yticks)
    
    for config in group1_configs:
        values = metrics_df.loc[config].values.tolist()
        values += values[:1]  # Close the radar chart
        ax.plot(angles, values, 'o-', linewidth=2, 
                label=config, color=config_groups[config_types[config]])
        ax.fill(angles, values, alpha=0.1, color=config_groups[config_types[config]])
    
    plt.xticks(angles[:-1], categories, fontsize=12)
    plt.yticks(fontsize=10)
    plt.title('DAMM Branch Ablation Performance Comparison (Full Model and Removed Single Branch)', size=15)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'damm_ablation_radar_group1.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Draw second group radar chart - only if we have configs for this group
    if group2_configs:
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, polar=True)
        
        # Set y-axis limits to focus on the 0.75-1 range
        ax.set_ylim(0.75, 1.0)
        yticks = np.linspace(0.75, 1.0, 6)
        ax.set_yticks(yticks)
        
        for config in group2_configs:
            values = metrics_df.loc[config].values.tolist()
            values += values[:1]  # Close the radar chart
            ax.plot(angles, values, 'o-', linewidth=2, 
                    label=config, color=config_groups[config_types[config]])
            ax.fill(angles, values, alpha=0.1, color=config_groups[config_types[config]])
        
        plt.xticks(angles[:-1], categories, fontsize=12)
        plt.yticks(fontsize=10)
        plt.title('DAMM Branch Ablation Performance Comparison (Two Branches and Single Branch)', size=15)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'damm_ablation_radar_group2.png'), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Create heatmap
    plt.figure(figsize=(15, 10))
    # Custom color scheme - light blue to dark blue
    cmap = LinearSegmentedColormap.from_list('BlueMap', ['#d1ecf1', '#0275d8'])
    heatmap = sns.heatmap(metrics_df, annot=True, cmap=cmap, linewidths=.5, fmt='.4f')
    plt.title('DAMM Branch Ablation Performance Heatmap', fontsize=16)
    plt.tight_layout()
    # Adjust y-axis label size
    heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=10)
    plt.savefig(os.path.join(output_dir, 'damm_ablation_heatmap.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Save results to CSV
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(os.path.join(output_dir, f'damm_ablation_results_{timestamp}.csv'))
    
    # Calculate branch contribution
    try:
        if 'Full DAMM' in metrics_df.index:
            full_damm = metrics_df.loc['Full DAMM']
            
            # Calculate impact of removing individual branches
            branch_impact = {}
            if 'DAMM without SPA' in metrics_df.index:
                branch_impact['SPA'] = ((full_damm - metrics_df.loc['DAMM without SPA']) / full_damm * 100).to_dict()
            if 'DAMM without CA' in metrics_df.index:
                branch_impact['CA'] = ((full_damm - metrics_df.loc['DAMM without CA']) / full_damm * 100).to_dict()
            if 'DAMM without PA' in metrics_df.index:
                branch_impact['PA'] = ((full_damm - metrics_df.loc['DAMM without PA']) / full_damm * 100).to_dict()
            if 'DAMM without DIR' in metrics_df.index:
                branch_impact['DIR'] = ((full_damm - metrics_df.loc['DAMM without DIR']) / full_damm * 100).to_dict()
            
            if branch_impact:
                # Create branch impact dataframe
                impact_df = pd.DataFrame(branch_impact).T
                impact_df.to_csv(os.path.join(output_dir, f'damm_branch_impact_{timestamp}.csv'))
                
                # Create bar chart for branch contributions
                plt.figure(figsize=(15, 10))
                ax = impact_df.plot(kind='bar', figsize=(15, 8))
                plt.title('Impact of DAMM Branches on Performance (%)', fontsize=16)
                plt.xlabel('Branch', fontsize=14)
                plt.ylabel('Performance Impact Percentage (%)', fontsize=14)
                plt.xticks(rotation=0)
                plt.grid(axis='y', linestyle='--', alpha=0.7)
                
                # Add value labels for each bar
                for i, metric in enumerate(impact_df.columns):
                    for j, value in enumerate(impact_df[metric]):
                        plt.text(j + (i-len(impact_df.columns)/2+0.5)*(0.8/len(impact_df.columns)), 
                                 value, f'{value:.2f}%', ha='center', va='bottom')
                
                plt.legend(title='Performance Metrics')
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'damm_branch_impact_{timestamp}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                # Create branch contribution radar chart
                categories = list(impact_df.columns)
                N = len(categories)
                angles = [n / float(N) * 2 * np.pi for n in range(N)]
                angles += angles[:1]  # Close the radar chart
                
                fig = plt.figure(figsize=(10, 10))
                ax = fig.add_subplot(111, polar=True)
                
                # Choose different colors for different branches
                branch_colors = {
                    'SPA': '#3498db',  # Blue
                    'CA': '#2ecc71',   # Green
                    'PA': '#e74c3c',   # Red
                    'DIR': '#f39c12'   # Orange
                }
                
                for branch in impact_df.index:
                    values = impact_df.loc[branch].values.tolist()
                    values += values[:1]  # Close the radar chart
                    ax.plot(angles, values, 'o-', linewidth=2, 
                            label=branch, color=branch_colors.get(branch, 'gray'))
                    ax.fill(angles, values, alpha=0.1, color=branch_colors.get(branch, 'gray'))
                
                plt.xticks(angles[:-1], categories, fontsize=12)
                plt.yticks([0, 5, 10, 15], ['0%', '5%', '10%', '15%'], fontsize=10)
                plt.title('Contribution of DAMM Branches to Performance (%)', size=15)
                plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
                plt.grid(True)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'damm_branch_contribution_radar_{timestamp}.png'), dpi=300, bbox_inches='tight')
                plt.close()
                
                return impact_df
                
    except Exception as e:
        print(f"Error calculating branch contribution: {e}")
    
    print(f"Visualization results saved to: {output_dir} directory")
    return None

def main():
    # Set up output to both terminal and file
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = f"damm_ablation_eval_log_{timestamp}.txt"
    tee = Tee(log_file)
    sys.stdout = tee
    
    # Record start time
    start_time = time.time()
    start_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    print(f"\nDAMM Branch Ablation Experiment Evaluation Started at: {start_datetime}")
    
    # Create a unique output directory for this run
    output_dir = f"damm_ablation_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Base command, all evaluations enable DropBlock, GCNet and SASPP modules
    base_cmd = "--use_dropblock --use_gcnet --use_saspp --use_damm"
    
    # Configuration dictionary, key is configuration name, value is (evaluation command, weight file path)
    configs = {
        "Full DAMM": (f"python eval_chase.py {base_cmd} --use_damm_spa --use_damm_ca --use_damm_pa --use_damm_dir", 
                    "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca-pa-dir_dropblock_gcnet.pth"),
        
        "DAMM without SPA": (f"python eval_chase.py {base_cmd} --use_damm_ca --use_damm_pa --use_damm_dir", 
                        "Chase/test/checkpoint/RSAN_saspp_damm-ca-pa-dir_dropblock_gcnet.pth"),
        
        "DAMM without CA": (f"python eval_chase.py {base_cmd} --use_damm_spa --use_damm_pa --use_damm_dir", 
                       "Chase/test/checkpoint/RSAN_saspp_damm-spa-pa-dir_dropblock_gcnet.pth"),
        
        "DAMM without PA": (f"python eval_chase.py {base_cmd} --use_damm_spa --use_damm_ca --use_damm_dir", 
                       "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca-dir_dropblock_gcnet.pth"),
        
        "DAMM without DIR": (f"python eval_chase.py {base_cmd} --use_damm_spa --use_damm_ca --use_damm_pa", 
                        "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca-pa_dropblock_gcnet.pth"),
        
        # Two-branch combinations
        "DAMM using SPA+CA": (f"python eval_chase.py {base_cmd} --use_damm_spa --use_damm_ca", 
                           "Chase/test/checkpoint/RSAN_saspp_damm-spa-ca_dropblock_gcnet.pth"),
        
        "DAMM using SPA+PA": (f"python eval_chase.py {base_cmd} --use_damm_spa --use_damm_pa", 
                           "Chase/test/checkpoint/RSAN_saspp_damm-spa-pa_dropblock_gcnet.pth"),
        
        "DAMM using SPA+DIR": (f"python eval_chase.py {base_cmd} --use_damm_spa --use_damm_dir", 
                            "Chase/test/checkpoint/RSAN_saspp_damm-spa-dir_dropblock_gcnet.pth"),
        
        "DAMM using CA+PA": (f"python eval_chase.py {base_cmd} --use_damm_ca --use_damm_pa", 
                          "Chase/test/checkpoint/RSAN_saspp_damm-ca-pa_dropblock_gcnet.pth"),
        
        "DAMM using CA+DIR": (f"python eval_chase.py {base_cmd} --use_damm_ca --use_damm_dir", 
                           "Chase/test/checkpoint/RSAN_saspp_damm-ca-dir_dropblock_gcnet.pth"),
        
        "DAMM using PA+DIR": (f"python eval_chase.py {base_cmd} --use_damm_pa --use_damm_dir", 
                           "Chase/test/checkpoint/RSAN_saspp_damm-pa-dir_dropblock_gcnet.pth"),
        
        # Single branch
        "DAMM using SPA": (f"python eval_chase.py {base_cmd} --use_damm_spa", 
                        "Chase/test/checkpoint/RSAN_saspp_damm-spa_dropblock_gcnet.pth"),
        
        "DAMM using CA": (f"python eval_chase.py {base_cmd} --use_damm_ca", 
                       "Chase/test/checkpoint/RSAN_saspp_damm-ca_dropblock_gcnet.pth"),
        
        "DAMM using PA": (f"python eval_chase.py {base_cmd} --use_damm_pa", 
                       "Chase/test/checkpoint/RSAN_saspp_damm-pa_dropblock_gcnet.pth"),
        
        "DAMM using DIR": (f"python eval_chase.py {base_cmd} --use_damm_dir", 
                        "Chase/test/checkpoint/RSAN_saspp_damm-dir_dropblock_gcnet.pth"),
        
        "DAMM empty shell": (f"python eval_chase.py {base_cmd}", 
                   "Chase/test/checkpoint/RSAN_saspp_damm-none_dropblock_gcnet.pth"),
        
        "Baseline model": ("python eval_chase.py --use_dropblock --use_gcnet --use_saspp", 
                  "Chase/test/checkpoint/RSAN_saspp_dropblock_gcnet.pth")
    }
    
    # Chinese to English mapping for configuration names
    config_name_mapping = {
        "完整DAMM": "Full DAMM",
        "DAMM不使用SPA": "DAMM without SPA",
        "DAMM不使用CA": "DAMM without CA",
        "DAMM不使用PA": "DAMM without PA",
        "DAMM不使用DIR": "DAMM without DIR",
        "DAMM只使用SPA+CA": "DAMM using SPA+CA",
        "DAMM只使用SPA+PA": "DAMM using SPA+PA",
        "DAMM只使用SPA+DIR": "DAMM using SPA+DIR",
        "DAMM只使用CA+PA": "DAMM using CA+PA",
        "DAMM只使用CA+DIR": "DAMM using CA+DIR",
        "DAMM只使用PA+DIR": "DAMM using PA+DIR",
        "DAMM只使用SPA": "DAMM using SPA",
        "DAMM只使用CA": "DAMM using CA",
        "DAMM只使用PA": "DAMM using PA",
        "DAMM只使用DIR": "DAMM using DIR",
        "DAMM空壳": "DAMM empty shell",
        "基准模型": "Baseline model"
    }
    
    # Store evaluation results
    evaluation_results = {}
    
    # Run all evaluation configurations
    for i, (config_name, (command, weight_path)) in enumerate(configs.items()):
        print(f"Progress: {i+1}/{len(configs)}")
        
        # Check if weight file exists
        if not os.path.exists(weight_path):
            print(f"Warning: Weight file does not exist: {weight_path}")
            user_input = input(f"Continue evaluating this configuration (using default weights)? (y/n): ")
            if user_input.lower() != 'y':
                continue
        
        # Add weight path to command
        full_command = f"{command} --weight_path={weight_path}"
        
        # Run evaluation and check return value
        return_code, result_file = run_eval_config(config_name, full_command, output_dir)
        
        if return_code == 0 and result_file and os.path.exists(result_file):
            # Parse evaluation results
            metrics = parse_eval_results(result_file)
            evaluation_results[config_name] = metrics
            print(f"Parsed evaluation results: {result_file}")
        else:
            print(f"Configuration {config_name} evaluation failed or no result file found.")
    
    # Create results dataframe
    if evaluation_results:
        # Convert to DataFrame
        results_df = pd.DataFrame.from_dict(evaluation_results, orient='index')
        
        # Rename columns to make them more readable
        column_mapping = {
            'Sensitivity (敏感度/召回率)': 'Sensitivity',
            'Specificity (特异度)': 'Specificity',
            'F1 Score': 'F1_Score',
            'Accuracy (准确率)': 'Accuracy',
            'AUC': 'AUC'
        }
        
        # Try to rename columns (if they exist)
        for old_col, new_col in column_mapping.items():
            if old_col in results_df.columns:
                results_df = results_df.rename(columns={old_col: new_col})
        
        # Visualize results
        impact_df = visualize_results(results_df, config_name_mapping, output_dir)
        
        # Print result summary
        print("\nDAMM Branch Ablation Experiment Evaluation Result Summary")
        print("="*80)
        
        # Only show performance metric columns
        metrics_columns = [col for col in ['Sensitivity', 'Specificity', 'F1_Score', 'Accuracy', 'AUC'] 
                           if col in results_df.columns]
        
        if metrics_columns:
            print("\nPerformance Metrics:")
            print(results_df[metrics_columns].to_string(float_format=lambda x: f"{x:.6f}"))
        
        if impact_df is not None:
            print("\nImpact of Each Branch on Performance (%):")
            print(impact_df.to_string(float_format=lambda x: f"{x:.2f}%"))
        
        print("="*80)
    else:
        print("No evaluation results available for visualization!")
    
    # Calculate and display total run time
    total_time = time.time() - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    # Get end time
    end_datetime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    
    print(f"All DAMM Branch Ablation Experiment Evaluations Completed!")
    print(f"Evaluation started at: {start_datetime}")
    print(f"Evaluation ended at: {end_datetime}")
    print(f"Total run time: {int(hours)} hours {int(minutes)} minutes {int(seconds)} seconds")
    
    # Close log file
    sys.stdout = sys.stdout.terminal
    tee.close()
    print(f"Evaluation log saved to: {log_file}")
    print(f"All results saved to: {output_dir}")

if __name__ == "__main__":
    main()