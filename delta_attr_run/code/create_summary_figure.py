import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Load data
base = Path("../results/_summary")
delta = pd.read_csv(base/"delta_summary.csv")
standard = pd.read_csv(base/"standard_summary.csv")

# Normalize column names
column_mappings = {
    'algorithm': 'algo',
    'delta_mag_l1': 'DeltaMag_L1',
    'bac': 'BAC',
    'dce': 'DCE',
    'delta_js': 'JSD_mean',
    'rank_overlap_mean': 'RankOverlapAt10_mean',
    'delta_topk_frac': 'DeltaTopK10'
}

for old_name, new_name in column_mappings.items():
    if old_name in delta.columns:
        delta = delta.rename(columns={old_name: new_name})
    if old_name in standard.columns:
        standard = standard.rename(columns={old_name: new_name})

# Create figure
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)

# 1. BAC Distribution by Dataset and Algorithm
ax1 = fig.add_subplot(gs[0, 0])
bac_data = delta.groupby(['dataset', 'algo'])['BAC'].mean().unstack()
bac_data.plot(kind='bar', ax=ax1, width=0.8)
ax1.set_title('Average BAC by Dataset and Algorithm', fontsize=14, fontweight='bold')
ax1.set_ylabel('BAC Score')
ax1.set_xlabel('Dataset')
ax1.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.tick_params(axis='x', rotation=45)

# 2. DCE Distribution by Dataset and Algorithm
ax2 = fig.add_subplot(gs[0, 1])
dce_data = delta.groupby(['dataset', 'algo'])['DCE'].mean().unstack()
dce_data.plot(kind='bar', ax=ax2, width=0.8)
ax2.set_title('Average DCE by Dataset and Algorithm', fontsize=14, fontweight='bold')
ax2.set_ylabel('DCE Score')
ax2.set_xlabel('Dataset')
ax2.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.tick_params(axis='x', rotation=45)

# 3. Delta Magnitude Distribution
ax3 = fig.add_subplot(gs[0, 2])
delta_mag_data = delta.groupby(['dataset', 'algo'])['DeltaMag_L1'].mean().unstack()
delta_mag_data.plot(kind='bar', ax=ax3, width=0.8)
ax3.set_title('Average Delta Magnitude by Dataset and Algorithm', fontsize=14, fontweight='bold')
ax3.set_ylabel('Delta Magnitude (L1)')
ax3.set_xlabel('Dataset')
ax3.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
ax3.tick_params(axis='x', rotation=45)

# 4. BAC vs DCE Scatter Plot
ax4 = fig.add_subplot(gs[1, 0])
scatter = ax4.scatter(delta['DCE'], delta['BAC'], 
                     c=delta['DeltaMag_L1'], cmap='viridis', 
                     s=100, alpha=0.7)
ax4.set_xlabel('DCE Score')
ax4.set_ylabel('BAC Score')
ax4.set_title('BAC vs DCE with Delta Magnitude', fontsize=14, fontweight='bold')
cbar = plt.colorbar(scatter, ax=ax4)
cbar.set_label('Delta Magnitude (L1)')

# 5. Performance Impact Analysis
ax5 = fig.add_subplot(gs[1, 1])
standard['acc_delta'] = standard['accuracy_B'] - standard['accuracy_A']
perf_data = standard.groupby(['dataset', 'algo'])['acc_delta'].mean().unstack()
perf_data.plot(kind='bar', ax=ax5, width=0.8)
ax5.set_title('Average Performance Change (B - A)', fontsize=14, fontweight='bold')
ax5.set_ylabel('Accuracy Delta')
ax5.set_xlabel('Dataset')
ax5.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
ax5.tick_params(axis='x', rotation=45)
ax5.axhline(y=0, color='red', linestyle='--', alpha=0.7)

# 6. TopK10 Distribution
ax6 = fig.add_subplot(gs[1, 2])
topk_data = delta.groupby(['dataset', 'algo'])['DeltaTopK10'].mean().unstack()
topk_data.plot(kind='bar', ax=ax6, width=0.8)
ax6.set_title('Average Delta TopK10 by Dataset and Algorithm', fontsize=14, fontweight='bold')
ax6.set_ylabel('Delta TopK10 Score')
ax6.set_xlabel('Dataset')
ax6.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
ax6.tick_params(axis='x', rotation=45)

# 7. Algorithm Performance Heatmap
ax7 = fig.add_subplot(gs[2, 0])
metrics = ['BAC', 'DCE', 'DeltaMag_L1', 'DeltaTopK10']
heatmap_data = delta.groupby('algo')[metrics].mean()
sns.heatmap(heatmap_data.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax7)
ax7.set_title('Algorithm Performance Heatmap', fontsize=14, fontweight='bold')

# 8. Dataset Performance Heatmap
ax8 = fig.add_subplot(gs[2, 1])
heatmap_data2 = delta.groupby('dataset')[metrics].mean()
sns.heatmap(heatmap_data2.T, annot=True, fmt='.3f', cmap='RdYlBu_r', ax=ax8)
ax8.set_title('Dataset Performance Heatmap', fontsize=14, fontweight='bold')

# 9. Pair Analysis
ax9 = fig.add_subplot(gs[2, 2])
pair_data = delta.groupby('pair')[metrics].mean()
pair_data.plot(kind='bar', ax=ax9, width=0.8)
ax9.set_title('Average Metrics by Pair', fontsize=14, fontweight='bold')
ax9.set_ylabel('Score')
ax9.set_xlabel('Pair')
ax9.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax9.tick_params(axis='x', rotation=45)

# 10. Stability Analysis (if available)
ax10 = fig.add_subplot(gs[3, 0])
if 'delta_stability_sigma001' in delta.columns and 'delta_stability_sigma005' in delta.columns:
    stability_data = delta.groupby('algo')[['delta_stability_sigma001', 'delta_stability_sigma005']].mean()
    stability_data.plot(kind='bar', ax=ax10, width=0.8)
    ax10.set_title('Stability Analysis by Algorithm', fontsize=14, fontweight='bold')
    ax10.set_ylabel('Stability Score')
    ax10.set_xlabel('Algorithm')
    ax10.legend(['σ=0.01', 'σ=0.05'])
    ax10.tick_params(axis='x', rotation=45)
else:
    ax10.text(0.5, 0.5, 'Stability data not available', ha='center', va='center', transform=ax10.transAxes)
    ax10.set_title('Stability Analysis', fontsize=14, fontweight='bold')

# 11. JSD Distribution
ax11 = fig.add_subplot(gs[3, 1])
jsd_data = delta.groupby(['dataset', 'algo'])['JSD_mean'].mean().unstack()
jsd_data.plot(kind='bar', ax=ax11, width=0.8)
ax11.set_title('Average JSD by Dataset and Algorithm', fontsize=14, fontweight='bold')
ax11.set_ylabel('JSD Score')
ax11.set_xlabel('Dataset')
ax11.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
ax11.tick_params(axis='x', rotation=45)

# 12. Rank Overlap Analysis
ax12 = fig.add_subplot(gs[3, 2])
rank_data = delta.groupby(['dataset', 'algo'])['RankOverlapAt10_mean'].mean().unstack()
rank_data.plot(kind='bar', ax=ax12, width=0.8)
ax12.set_title('Average Rank Overlap by Dataset and Algorithm', fontsize=14, fontweight='bold')
ax12.set_ylabel('Rank Overlap Score')
ax12.set_xlabel('Dataset')
ax12.legend(title='Algorithm', bbox_to_anchor=(1.05, 1), loc='upper left')
ax12.tick_params(axis='x', rotation=45)

# Add overall title
fig.suptitle('Delta Attribution Analysis: Comprehensive Results Summary', 
             fontsize=20, fontweight='bold', y=0.98)

# Save the figure
plt.savefig('delta_attribution_summary.png', dpi=300, bbox_inches='tight', facecolor='white')
plt.savefig('delta_attribution_summary.pdf', bbox_inches='tight', facecolor='white')

print("Summary figure saved as 'delta_attribution_summary.png' and 'delta_attribution_summary.pdf'")

# Print key statistics
print("\n=== KEY STATISTICS ===")
print(f"Total experiments: {len(delta)}")
print(f"Datasets: {', '.join(delta['dataset'].unique())}")
print(f"Algorithms: {', '.join(delta['algo'].unique())}")
print(f"Pairs: {', '.join(delta['pair'].unique())}")

print("\n=== TOP PERFORMERS ===")
print("Best BAC scores:")
top_bac = delta.nlargest(5, 'BAC')[['dataset', 'algo', 'pair', 'BAC']]
print(top_bac.to_string(index=False))

print("\nLowest DCE scores:")
top_dce = delta.nsmallest(5, 'DCE')[['dataset', 'algo', 'pair', 'DCE']]
print(top_dce.to_string(index=False))

print("\nHighest Delta Magnitude:")
top_delta = delta.nlargest(5, 'DeltaMag_L1')[['dataset', 'algo', 'pair', 'DeltaMag_L1']]
print(top_delta.to_string(index=False))

print("\n=== PERFORMANCE IMPACT ===")
print("Largest performance improvements:")
top_perf = standard.nlargest(5, 'acc_delta')[['dataset', 'algo', 'pair', 'acc_delta']]
print(top_perf.to_string(index=False))

print("\nLargest performance degradations:")
worst_perf = standard.nsmallest(5, 'acc_delta')[['dataset', 'algo', 'pair', 'acc_delta']]
print(worst_perf.to_string(index=False)) 