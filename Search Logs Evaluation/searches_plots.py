import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('Search Logs Data/searches_summary.csv')

# Extract values
model_sizes = df['modelsize']
rerun_counts = df['total_reruns_avg']
successful_rerun_counts = df['success_reruns_avg']
query_counts = df['total_queries_avg']
unique_query_counts = df['unique_queries_avg']
tokens_counts = df['completion_tokens_avg']

# Extract standard deviations
rerun_counts_std = df['total_reruns_std']
successful_rerun_counts_std = df['success_reruns_std']
query_counts_std = df['total_queries_std']
unique_query_counts_std = df['unique_queries_std']
tokens_counts_std = df['completion_tokens_std']

x = np.arange(len(model_sizes))
width = 0.4

# Create plot summarizing rerun counts
fig1, ax1 = plt.subplots(figsize=(10, 5.5))
palette = sns.color_palette('hls', 2)

for i, (category, values, errors) in enumerate(zip(
    ['Rerun Count', 'Successful Rerun Count'],
    [rerun_counts, successful_rerun_counts],
    [rerun_counts_std, successful_rerun_counts_std]
)):
    bars = ax1.bar(x + i * width - (1.5 * width), values, width, yerr=errors, capsize=5, label=category, color=palette[i])
    for bar, std in zip(bars, errors):
        height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        offset = -5 if i == 0 else 5
        
        ax1.annotate(f'{height:.1f} ±\n{std:.1f}', xy=(x_pos, height + std + 0.2),  xytext=(offset, 2), textcoords='offset points', ha='center', va='bottom', fontname='Georgia', fontsize=16)

ax1.set_xlabel('Model Size', fontname='Georgia', fontsize=18)
ax1.set_ylabel('Values', fontname='Georgia', fontsize=18)
ax1.set_xticks(x - width)
ax1.set_xticklabels(model_sizes, fontname='Georgia', fontsize=16)
ax1.set_ylim((0, 7))
ax1.set_yticklabels(ax1.get_yticks(), fontname='Georgia', fontsize=16)
ax1.set_title('Mean Rerun Values', fontname='Georgia', fontsize=20)
legend = ax1.legend()
for text in legend.get_texts():
    text.set_fontname('Georgia')
    text.set_fontsize(16)

plt.tight_layout()
plt.show()

# Create plot summarizing query counts
fig2, ax2 = plt.subplots(figsize=(10, 5.5))

for i, (category, values, errors) in enumerate(zip(
    ['Query Count', 'Unique Query Count'],
    [query_counts, unique_query_counts],
    [query_counts_std, unique_query_counts_std]
)):
    bars = ax2.bar(x + i * width - (1.5 * width), values, width, yerr=errors, capsize=5, label=category, color=palette[i])
    for bar, std in zip(bars, errors):
        height = bar.get_height()
        x_pos = bar.get_x() + bar.get_width() / 2
        offset = -5 if i == 0 else 5
        
        ax2.annotate(f'{height:.1f} ±\n{std:.1f}', xy=(x_pos, height + std + 0.1),  xytext=(offset, 2), textcoords='offset points', ha='center', va='bottom', fontname='Georgia', fontsize=16)

ax2.set_xlabel('Model Size', fontname='Georgia', fontsize=18)
ax2.set_ylabel('Values', fontname='Georgia', fontsize=18)
ax2.set_xticks(x - width)
ax2.set_xticklabels(model_sizes, fontname='Georgia', fontsize=16)
ax2.set_ylim((0, 5))
ax2.set_yticklabels(ax2.get_yticks(), fontname='Georgia', fontsize=16)
ax2.set_title('Mean Query Values', fontname='Georgia', fontsize=20)
legend = ax2.legend()
for text in legend.get_texts():
    text.set_fontname('Georgia')
    text.set_fontsize(16)

plt.tight_layout()
plt.show()

# Create plot summarizing completion tokens use
fig3, ax3 = plt.subplots(figsize=(10, 5.5))
palette = sns.color_palette('hls', 1)
bars = ax3.bar(x - (1.5 * width), tokens_counts, width, yerr=tokens_counts_std, capsize=5, label='Completion Tokens', color=palette[0])

for bar, std in zip(bars, tokens_counts_std):
    height = bar.get_height()
    ax3.annotate(f'{int(height)} ± {int(std)}', xy=(bar.get_x() + bar.get_width() / 2, height+std+3), 
                 xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontname='Georgia', fontsize=16)

ax3.set_xlabel('Model Size', fontname='Georgia', fontsize=18)
ax3.set_ylabel('Values', fontname='Georgia', fontsize=18)
ax3.set_xticks(x-width*1.5)
ax3.set_xticklabels(model_sizes, fontname='Georgia', fontsize=16)
ax3.set_ylim((0, 140))
ax3.set_yticklabels(ax3.get_yticks(), fontname='Georgia', fontsize=16)
ax3.set_title('Mean Completion Tokens Count', fontname='Georgia', fontsize=20)
for text in legend.get_texts():
    text.set_fontname('Georgia')
    text.set_fontsize(16)

plt.tight_layout()
plt.show()
