import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap

##########################################################################################

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", ["green", "white", "green"], N=100)

prompt_type_order = ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']
size_order = [1.5, 3, 7, 32, 72]

size_rename = {1.5: '1.5B', 3: '3B', 7: '7B', 32: '32B', 72: '72B'}
prompt_type_rename = {'agentic': 'Agentic', 'zero-shot': 'Zero-Shot', 'few-shot': 'Few-Shot', 'parallel': 'Parallel', 'chain-of-thought': 'Chain-of-\nThought'}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
axes = axes.flatten()

# Loop through metrics and groupings
for idx, (metric, grouping) in enumerate([('Z-Score', 'material'), ('Z-Score', 'design-criteria'), ('MAE', 'material'), ('MAE', 'design-criteria')]):
    df = pd.read_csv(f'Data Evaluation/Results/{metric.lower()}_{grouping}.csv')

    # Create pivot table from data
    df['Prompt Type'] = pd.Categorical(df['prompt type'], categories=prompt_type_order, ordered=True)
    df['Size'] = pd.Categorical(df['size'], categories=size_order, ordered=True)
    pivot_table = df.pivot_table(index='Size', columns='Prompt Type', values=metric.lower(), aggfunc='mean', observed=True)

    # Get mean metric for each column and row
    pivot_table['Mean'] = pivot_table.mean(axis=1)
    pivot_table.loc['Mean'] = pivot_table.mean()

    # Plot mean values as heatmap on corresponding subplot
    ax = axes[idx]
    pivot_table.rename(columns=prompt_type_rename, index=size_rename, inplace=True)
    sns.heatmap(pivot_table, annot=True, cmap=custom_cmap, center=0, linewidths=0.5, cbar_kws={'label': metric}, ax=ax)

    mean_row_index = pivot_table.index.get_loc('Mean')
    mean_col_index = pivot_table.columns.get_loc('Mean')
    ax.hlines(mean_row_index, xmin=-0.5, xmax=len(pivot_table.columns), color='black', linewidth=1, linestyle='-')
    ax.vlines(mean_col_index, ymin=-0.5, ymax=len(pivot_table.index), color='black', linewidth=1, linestyle='-')
    ax.hlines(0, xmin=-0.5, xmax=len(pivot_table.columns), color='black', linewidth=2, linestyle='-')
    ax.hlines(len(pivot_table.index), xmin=-0.5, xmax=len(pivot_table.columns), color='black', linewidth=2, linestyle='-')
    ax.vlines(0, ymin=-0.5, ymax=len(pivot_table.index), color='black', linewidth=2, linestyle='-')
    ax.vlines(len(pivot_table.columns), ymin=-0.5, ymax=len(pivot_table.index), color='black', linewidth=2, linestyle='-')
    grouping_title = ' and '.join([word.capitalize() for word in grouping.split('-')])
    grouping_title = grouping_title.replace('Criteria', 'Criterion')
    ax.set_title(f'Average {metric} for Grouping by {grouping_title}')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")

plt.tight_layout(pad=2.0)
plt.savefig('average_metrics.png')
plt.show()

for metric, grouping in [('Z-Score', 'material'), ('Z-Score', 'design-criteria'), ('MAE', 'material'), ('MAE', 'design-criteria')]:
    df = pd.read_csv(f'Data Evaluation/Results/{metric.lower()}_{grouping}.csv')
    df['Prompt Type'] = pd.Categorical(df['prompt type'], categories=prompt_type_order, ordered=True)
    df['Size'] = pd.Categorical(df['size'], categories=size_order, ordered=True)
    pivot_table = df.pivot_table(index='Prompt Type', columns='Size', values=metric.lower(), aggfunc='mean', observed=True)
    pivot_table['Mean'] = pivot_table.mean(axis=1)
    pivot_table.loc['Mean'] = pivot_table.mean()

    output_filename = f'Data Evaluation/Results/{metric.lower()}_{grouping}_pivot_table.csv'
    pivot_table.to_csv(output_filename)