import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

##########################################################################################

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14
custom_cmap = LinearSegmentedColormap.from_list("custom_diverging", ["green", "white", "green"], N=100)

prompt_type_order = ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']
size_order = [1.5, 3, 7, 32, 72]

size_rename = {1.5: '1.5B', 3: '3B', 7: '7B', 32: '32B', 72: '72B'}
prompt_type_rename = {'agentic': 'Agentic', 'zero-shot': 'Zero-Shot', 'few-shot': 'Few-Shot', 'parallel': 'Parallel', 'chain-of-thought': 'Chain-of-\nThought'}

# Create heatmap for a given metric
def plot_heatmap(metric, cmap, center=None, title=None, filename=None):
    df = pd.read_csv(f'Data Evaluation/Results/{metric.lower()}_material.csv')
    df['Prompt Type'] = pd.Categorical(df['prompt type'], categories=prompt_type_order, ordered=True)
    df['Size'] = pd.Categorical(df['size'], categories=size_order, ordered=True)
    pivot_table = df.pivot_table(index='Size', columns='Prompt Type', values=metric.lower(), aggfunc='mean', observed=True)

    # Compute mean values
    pivot_table['Mean'] = pivot_table.mean(axis=1)
    pivot_table.loc['Mean'] = pivot_table.mean()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    pivot_table.rename(columns=prompt_type_rename, index=size_rename, inplace=True)
    ax = sns.heatmap(pivot_table, annot=True, annot_kws={'size': 18}, cmap=cmap, center=center, linewidths=0.5, cbar_kws={'label': metric})
    
    mean_row_index = pivot_table.index.get_loc('Mean')
    mean_col_index = pivot_table.columns.get_loc('Mean')
    plt.hlines(mean_row_index, xmin=-0.5, xmax=len(pivot_table.columns), color='black', linewidth=1, linestyle='-')
    plt.vlines(mean_col_index, ymin=-0.5, ymax=len(pivot_table.index), color='black', linewidth=1, linestyle='-')
    plt.hlines(0, xmin=-0.5, xmax=len(pivot_table.columns), color='black', linewidth=2, linestyle='-')
    plt.hlines(len(pivot_table.index), xmin=-0.5, xmax=len(pivot_table.columns), color='black', linewidth=2, linestyle='-')
    plt.vlines(0, ymin=-0.5, ymax=len(pivot_table.index), color='black', linewidth=2, linestyle='-')
    plt.vlines(len(pivot_table.columns), ymin=-0.5, ymax=len(pivot_table.index), color='black', linewidth=2, linestyle='-')

    plt.title(title,fontsize=20)
    plt.xticks(rotation=45, ha="right", fontsize=16)
    plt.yticks(rotation=45, ha="right", fontsize=16)
    ax.set_xlabel(ax.get_xlabel(), fontsize=18) 
    ax.set_ylabel(ax.get_ylabel(), fontsize=18)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

# Create heatmap for both metrics
plot_heatmap(metric='Z-Score', cmap=custom_cmap, center=0, title='Mean Z-Scores', filename='z_score_heatmap.png')
plot_heatmap(metric='MAE', cmap='Greens', title='Mean Absolute Errors (MAEs) to Survey Data', filename='mae_heatmap.png')

# Create and save pivot tables for both metrics
for metric in ['MAE', 'Z-Score']:
    df = pd.read_csv(f'Data Evaluation/Results/{metric.lower()}_material.csv')
    df['Prompt Type'] = pd.Categorical(df['prompt type'], categories=prompt_type_order, ordered=True)
    df['Size'] = pd.Categorical(df['size'], categories=size_order, ordered=True)
    pivot_table = df.pivot_table(index='Prompt Type', columns='Size', values=metric.lower(), aggfunc='mean', observed=True)
    pivot_table['Mean'] = pivot_table.mean(axis=1)
    pivot_table.loc['Mean'] = pivot_table.mean()
    
    output_filename = f'Data Evaluation/Results/{metric.lower()}_material_pivot_table.csv'
    pivot_table.to_csv(output_filename)