import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
from collections import defaultdict

# matplotlib其实是不支持显示中文的 显示中文需要一行代码设置字体
plt.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False   # 步骤二（解决坐标轴负数的负号显示问题）

def plot_accuracy_updates(data_dict, output_path, title='精度和更新次数随时间的变化'):
    """绘制精度和更新次数随时间的变化图
    Args:
        data_dict: 可以是单个DataFrame或包含多个DataFrame的字典
        output_path: 输出文件路径
        title: 图表标题
    """
    # 创建图表
    fig, ax1 = plt.subplots(figsize=(15, 8))
    
    # 定义颜色方案
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    # 绘制精度折线图（左y轴）
    ax1.set_xlabel('时间 (秒)')
    color = colors[0]
    ax1.set_ylabel('精度')    
    # 创建右y轴
    ax2 = ax1.twinx()
    ax2.set_ylabel('总更新次数')   

    # 判断输入类型
    if isinstance(data_dict, pd.DataFrame):
        # 单个实验
        agg_df = data_dict
        line1 = ax1.plot(agg_df['time'], agg_df['accuracy'], color=color, label='精度')
        ax1.tick_params(axis='y', labelcolor=color)
        line2 = ax2.plot(agg_df['time'], agg_df['total_updates'], color=color, label='总更新次数')
        ax2.tick_params(axis='y', labelcolor=color)

    else:
        # 多个实验对比
        for i, (name, stats) in enumerate(data_dict.items()):
            color = colors[i % len(colors)]
            agg_df = pd.DataFrame(stats['aggregation_records'])
            line1 = ax1.plot(agg_df['time'], agg_df['accuracy'], color=color, label=name)
            ax1.tick_params(axis='y', labelcolor=color)
            
            line2 = ax2.plot(agg_df['time'], agg_df['total_updates'], color=color, label=name, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)
    # 合并两个轴的图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    
    # 添加网格
    ax1.grid(True, linestyle='--', alpha=0.7)
    # 设置标题
    plt.title(title)
    # 调整布局以适应图例
    plt.tight_layout()
    # 保存图表
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

def analyze_log(log_file_path):
    """分析单个日志文件"""
    # 读取日志文件
    with open(log_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # 提取Update uploaded信息
    update_pattern = r'\[Client (\d+)\] Update uploaded \((\d+)/(\d+)\) at ([\d.]+)s'
    updates = []
    
    # 提取Aggregation completed信息
    agg_pattern = r'\[Round (\d+)\] Aggregation completed at ([\d.]+)s\. Get model v(\d+)\. Acc ([\d.]+) Loss ([\d.]+)'
    
    # 记录每个时间点的更新次数
    update_counts = defaultdict(int)
    # 记录聚合完成时的信息
    aggregation_records = []
    
    for line in lines:
        # 处理Update uploaded信息
        update_match = re.search(update_pattern, line)
        if update_match:
            client_id = int(update_match.group(1))
            current_count = int(update_match.group(2))
            buffer_size = int(update_match.group(3))
            time = float(update_match.group(4))
            updates.append({
                'client_id': client_id,
                'current_count': current_count,
                'buffer_size': buffer_size,
                'time': time
            })
            update_counts[time] = len(updates)
        
        # 处理Aggregation completed信息
        agg_match = re.search(agg_pattern, line)
        if agg_match:
            round_num = int(agg_match.group(1))
            time = float(agg_match.group(2))
            model_version = int(agg_match.group(3))
            accuracy = float(agg_match.group(4))
            loss = float(agg_match.group(5))
            
            # 记录该时间点的总更新次数
            total_updates = update_counts[time]
            
            aggregation_records.append({
                'round': round_num,
                'time': time,
                'model_version': model_version,
                'accuracy': accuracy,
                'loss': loss,
                'total_updates': total_updates
            })
    
    # 转换为DataFrame
    df = pd.DataFrame(updates)
    agg_df = pd.DataFrame(aggregation_records)
    
    # 计算每个客户端的更新次数
    client_counts = df['client_id'].value_counts().sort_index()
    
    # 保存统计信息到JSON文件
    output_dir = os.path.dirname(log_file_path)
    stats = {
        'total_updates': len(df),
        'client_counts': client_counts.to_dict(),
        'aggregation_records': aggregation_records
    }
    
    stats_file = os.path.join(output_dir, 'analysis_stats.json')
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 绘制单个实验的图表
    plot_accuracy_updates(agg_df, os.path.join(output_dir, 'accuracy_updates.png'))
    
    return stats

def compare_experiments(experiment_paths):
    """比较多个实验结果"""
    # 读取所有实验的统计数据
    stats = {}
    for path in experiment_paths:
        stats_file = os.path.join(path, 'analysis_stats.json')
        if os.path.exists(stats_file):
            with open(stats_file, 'r', encoding='utf-8') as f:
                stats[os.path.basename(path)] = json.load(f)
    
    if not stats:
        print("没有找到有效的实验数据")
        return
    
    # 获取输出目录（使用第一个实验的目录）
    output_dir = os.path.dirname(experiment_paths[0])
    
    # 绘制对比图
    plot_accuracy_updates(stats, os.path.join(output_dir, 'experiment_comparison.png'), 
                         title='不同实验结果的对比')
    
    # 打印统计信息
    print("\n实验对比统计:")
    for name, data in stats.items():
        print(f"\n{name}:")
        print(f"总更新次数: {data['total_updates']}")
        print(f"最终精度: {data['aggregation_records'][-1]['accuracy']:.4f}")
        print(f"总训练时间: {data['aggregation_records'][-1]['time']:.2f}秒")

if __name__ == "__main__":
    # 单个实验分析
    # log_file_path = r"output\simplecnn\CA2FL_noniid-0.5\output.log"
    # analyze_log(log_file_path)
    
    # 多个实验对比
    experiment_paths = [
        r"output\simplecnn\fedbuffadv_noniid",
        r"output\simplecnn\fedbuff_noniid",
        r"output\simplecnn\CA2FL_noniid-0.5",
        # 添加更多实验路径
    ]
    compare_experiments(experiment_paths) 