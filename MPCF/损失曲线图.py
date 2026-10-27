import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator
import warnings
warnings.filterwarnings('ignore')

# 设置高质量绘图参数
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = ['Arial', 'DejaVu Sans']

def load_and_process_data(csv_file_path):
    """
    加载并处理训练数据 - 统计所有epoch
    """
    # 读取CSV文件
    df = pd.read_csv(csv_file_path)
    
    # 移除epoch限制，处理所有数据
    print(f"数据范围: Epoch {df['epoch'].min()} 到 Epoch {df['epoch'].max()}")
    print(f"总共包含 {len(df)} 条记录，{df['epoch'].nunique()} 个不同的epoch")
    
    # 按epoch分组，计算每个epoch的平均损失值
    epoch_stats = df.groupby('epoch').agg({
        'total_loss': 'mean',
        'mae_loss': 'mean',
        'polarity_loss': 'mean',
        'nce_loss': 'mean',
        'nce2_loss': 'mean',
        'nce3_loss': 'mean',
        'neutral_penalty': 'mean'
    }).reset_index()
    
    return epoch_stats

def create_publication_plot(data, save_path='training_losses_all_epochs.png'):
    """
    创建适合发表的单一图表，包含所有损失曲线
    """
    # 创建图形
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # 定义专业的颜色方案和样式
    loss_config = {
        'total_loss': {'color': '#1f77b4', 'linestyle': '-', 'marker': 'o', 'linewidth': 3.0, 'label': 'Total Loss'},
        'mae_loss': {'color': '#ff7f0e', 'linestyle': '--', 'marker': 's', 'linewidth': 2.5, 'label': 'MAE Loss'},
        'polarity_loss': {'color': '#2ca02c', 'linestyle': '-.', 'marker': '^', 'linewidth': 2.5, 'label': 'Polarity Loss'},
        'nce_loss': {'color': '#d62728', 'linestyle': ':', 'marker': 'v', 'linewidth': 2.5, 'label': 'NCE Loss'},
        'nce2_loss': {'color': '#9467bd', 'linestyle': '-', 'marker': 'D', 'linewidth': 2.0, 'label': 'NCE2 Loss'},
        'nce3_loss': {'color': '#8c564b', 'linestyle': '--', 'marker': 'p', 'linewidth': 2.0, 'label': 'NCE3 Loss'},
        'neutral_penalty': {'color': '#e377c2', 'linestyle': '-.', 'marker': 'h', 'linewidth': 2.0, 'label': 'Neutral Penalty'}
    }
    
    # 根据epoch数量调整标记显示频率
    total_epochs = len(data)
    markevery = max(1, total_epochs // 20)  # 动态调整标记频率，最多显示20个标记点
    
    # 绘制每条损失曲线
    for loss_name, config in loss_config.items():
        ax.plot(data['epoch'], data[loss_name],
                color=config['color'],
                linestyle=config['linestyle'],
                marker=config['marker'],
                linewidth=config['linewidth'],
                markersize=6,
                markevery=markevery,  # 动态调整标记频率
                label=config['label'],
                alpha=0.85)
    
    # 取消顶部标题
    ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
    ax.set_ylabel('Loss Value', fontsize=14, fontweight='bold')
    
    # 动态设置坐标轴范围和刻度
    ax.set_xlim(data['epoch'].min(), data['epoch'].max())
    
    # 根据epoch数量动态调整x轴刻度
    if total_epochs <= 20:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=total_epochs))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=20))
    
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.tick_params(axis='both', which='minor', labelsize=10)
    
    # 设置网格
    ax.grid(True, alpha=0.4, linestyle='-', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # 设置图例
    legend = ax.legend(loc='upper right', 
                      frameon=True, 
                      fancybox=True, 
                      shadow=True,
                      fontsize=12,
                      ncol=1,
                      borderpad=0.8,
                      columnspacing=1.0)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.95)
    legend.get_frame().set_edgecolor('gray')
    legend.get_frame().set_linewidth(1.2)
    
    # 设置坐标轴样式
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    # 调整布局
    plt.tight_layout()
    
    # 取消英文图注描述（原先通过 fig.text 添加）
    
    # 保存高质量图像
    plt.savefig(save_path, 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white', 
                edgecolor='none',
                format='png')
    
    # 同时保存为矢量格式（适合论文发表）
    plt.savefig(save_path.replace('.png', '.pdf'), 
                dpi=300, 
                bbox_inches='tight', 
                facecolor='white', 
                edgecolor='none',
                format='pdf')
    
    plt.show()
    return fig

def print_statistics(data):
    """
    打印统计信息
    """
    print(f"=== Training Loss Statistics (Epochs {data['epoch'].min()}-{data['epoch'].max()}) ===")
    print(f"Number of epochs analyzed: {len(data)}")
    print("\nInitial vs Final Loss Values:")
    
    loss_columns = ['total_loss', 'mae_loss', 'polarity_loss', 'nce_loss', 
                   'nce2_loss', 'nce3_loss', 'neutral_penalty']
    
    for col in loss_columns:
        initial = data[col].iloc[0]
        final = data[col].iloc[-1]
        reduction = ((initial - final) / initial * 100) if initial != 0 else 0
        print(f"{col.replace('_', ' ').title():15}: {initial:.6f} → {final:.6f} ({reduction:+.2f}%)")
    
    print(f"\nLoss Statistics Summary:")
    print(data[loss_columns].describe().round(6))
    
    # 添加额外的统计信息
    print(f"\nAdditional Statistics:")
    for col in loss_columns:
        min_loss = data[col].min()
        min_epoch = data.loc[data[col].idxmin(), 'epoch']
        print(f"{col.replace('_', ' ').title():15}: Min = {min_loss:.6f} at epoch {min_epoch}")

def main():
    """
    主函数 - 运行完整的分析流程
    """
    # 文件路径 - 请修改为您的实际文件路径
    csv_file_path = r"D:\PythonWorkplace\MPCF-main\training_logs\MPCF_mosi_train_metrics_20250909_161005.csv"
    
    try:
        # 加载和处理数据
        print("Loading and processing data...")
        processed_data = load_and_process_data(csv_file_path)
        
        # 打印统计信息
        print_statistics(processed_data)
        
        # 创建可视化图表
        print("\nCreating visualization...")
        fig = create_publication_plot(processed_data, 'training_losses_all_epochs.png')
        
        print("✓ Visualization completed!")
        print("✓ Files saved: training_losses_all_epochs.png and training_losses_all_epochs.pdf")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{csv_file_path}'")
        print("Please make sure the CSV file exists in the current directory or update the file path.")
    except Exception as e:
        print(f"Error: {str(e)}")

# 运行主程序
if __name__ == "__main__":
    main()