# auto_audit.py
import pandas as pd
from pathlib import Path

base = Path("../results/_summary")
std = pd.read_csv(base/"standard_summary.csv")
delta = pd.read_csv(base/"delta_summary.csv")

# Normalize column names to handle column name mappings
column_mappings = {
    'algorithm': 'algo',
    'delta_mag_l1': 'DeltaMag_L1',
    'bac': 'BAC',
    'dce': 'DCE',
    'delta_js': 'JSD_mean',
    'rank_overlap_mean': 'RankOverlapAt10_mean',
    'delta_topk_frac': 'DeltaTopK10'
}

# Apply mappings to delta dataframe
for old_name, new_name in column_mappings.items():
    if old_name in delta.columns:
        delta = delta.rename(columns={old_name: new_name})

# Apply mappings to std dataframe
for old_name, new_name in column_mappings.items():
    if old_name in std.columns:
        std = std.rename(columns={old_name: new_name})

def top(df, col, k=5, asc=False):
    s = df.sort_values(col, ascending=asc).head(k)
    return s[['dataset','algo','pair',col]]

print("\n=== Headline checks ===")
for col in ['BAC','DCE','DeltaTopK10','JSD_mean','RankOverlapAt10_mean']:
    if col in delta.columns:
        print(f"\nTop by {col}:")
        print(top(delta, col, k=5, asc=(col in ['DCE'])).to_string(index=False))

print("\n=== Expected-small-Δ pairs sanity ===")
small_pairs = delta.query("(algo=='RandomForest' and pair=='Pair1') or \
                           (algo=='KNeighbors' and pair=='Pair3') or \
                           (algo=='LogisticRegression' and pair=='Pair3')")
print(small_pairs[['dataset','algo','pair','DeltaMag_L1','BAC','DCE']].to_string(index=False))

print("\n=== Kernel-change (should be big-Δ) ===")
big_pairs = delta.query("algo=='SVC' and pair in ['Pair1','Pair3']")
print(big_pairs[['dataset','pair','DeltaMag_L1','JSD_mean','RankOverlapAt10_mean','BAC']].to_string(index=False))

print("\n=== Flags ===")
flags = []
# BAC negative or near zero with large Δ
tmp = delta[(delta['DeltaMag_L1']>delta['DeltaMag_L1'].median()) & (delta['BAC']<0.05)]
flags += [("Low BAC with large Δ", tmp)]
# Very large DCE
tmp = delta[delta['DCE'] > delta['DCE'].median()*2]
flags += [("High DCE (Δ not conserving behaviour)", tmp)]
# Stability oddities
if 'DeltaStability_sigma001' in delta.columns and 'DeltaStability_sigma005' in delta.columns:
    tmp = delta[delta['DeltaStability_sigma005'] < delta['DeltaStability_sigma001']]
    flags += [("Δ-Stability decreases with larger noise", tmp)]

if flags:
    for name, df in flags:
        if len(df):
            print(f"\n{name}:")
            print(df[['dataset','algo','pair','DeltaMag_L1','BAC','DCE']].to_string(index=False))
else:
    print("No obvious red flags.")

print("\n=== Performance deltas (context only) ===")
std['acc_delta'] = std['accuracy_B'] - std['accuracy_A']
print(std[['dataset','algo','pair','acc_delta','macro_f1_A','macro_f1_B']].sort_values('acc_delta').to_string(index=False))
