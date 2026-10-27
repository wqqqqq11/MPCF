def normalize_mae_for_comparison(mae_value, original_range, target_range=(-1, 1)):
    """
    将不同标签范围的MAE值标准化到相同范围，便于公平对比
    
    Args:
        mae_value: 原始MAE值
        original_range: 原始标签范围，如(-3, 3)或(-1, 1)
        target_range: 目标标准化范围，默认(-1, 1)
    
    Returns:
        标准化后的MAE值
    """
    original_span = original_range[1] - original_range[0]
    target_span = target_range[1] - target_range[0]
    
    # 计算相对误差比例
    relative_error = mae_value / original_span
    
    # 转换到目标范围
    normalized_mae = relative_error * target_span
    
    return normalized_mae

def compare_cross_dataset_performance(results_dict):
    """
    跨数据集性能对比工具
    
    Args:
        results_dict: {
            'dataset_name': {
                'mae': float,
                'corr': float, 
                'f1': float,
                'acc': float,
                'label_range': tuple
            }
        }
    """
    print("=" * 60)
    print("📊 跨数据集性能对比报告")
    print("=" * 60)
    
    # 标准化MAE对比
    print("\n🎯 MAE对比 (标准化到[-1,1]范围):")
    for dataset, metrics in results_dict.items():
        original_mae = metrics['mae']
        normalized_mae = normalize_mae_for_comparison(
            original_mae, 
            metrics['label_range']
        )
        relative_error = original_mae / (metrics['label_range'][1] - metrics['label_range'][0])
        
        print(f"  {dataset:>8}: MAE={original_mae:.4f} → 标准化MAE={normalized_mae:.4f} (相对误差:{relative_error:.1%})")
    
    # 直接可比指标
    print("\n📈 直接可比指标:")
    print(f"{'Dataset':>8} | {'Correlation':>11} | {'F1-Score':>8} | {'Accuracy':>8}")
    print("-" * 45)
    for dataset, metrics in results_dict.items():
        print(f"{dataset:>8} | {metrics['corr']:>11.4f} | {metrics['f1']:>8.4f} | {metrics['acc']:>8.4f}")
    
    # 排名分析
    print("\n🏆 性能排名:")
    datasets = list(results_dict.keys())
    
    # 按标准化MAE排名（越小越好）
    mae_ranking = sorted(datasets, key=lambda d: normalize_mae_for_comparison(
        results_dict[d]['mae'], results_dict[d]['label_range']
    ))
    
    # 按相关系数排名（越大越好）
    corr_ranking = sorted(datasets, key=lambda d: results_dict[d]['corr'], reverse=True)
    
    print(f"  MAE排名: {' > '.join(mae_ranking)} (越右越好)")
    print(f"  相关系数排名: {' > '.join(corr_ranking)} (越右越好)")

# 使用示例
if __name__ == "__main__":
    # 示例数据
    example_results = {
        'SIMS': {
            'mae': 0.441,
            'corr': 0.595,
            'f1': 0.736,
            'acc': 0.740,
            'label_range': (-1, 1)
        },
        'MOSI': {
            'mae': 0.917,
            'corr': 0.750,
            'f1': 0.800,
            'acc': 0.820,
            'label_range': (-3, 3)
        },
        'MOSEI': {
            'mae': 0.599,
            'corr': 0.850,
            'f1': 0.880,
            'acc': 0.900,
            'label_range': (-3, 3)
        }
    }
    
    compare_cross_dataset_performance(example_results) 