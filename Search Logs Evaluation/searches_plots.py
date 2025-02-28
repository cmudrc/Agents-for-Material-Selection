import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

##########################################################################################

df = pd.read_csv('Search Logs Data/searches_summary.csv')
print(df)
model_sizes = df['modelsize']
rerun_counts = df['total_reruns_avg']
successful_rerun_counts = df['success_reruns_avg']
query_counts = df['total_queries_avg']
unique_query_counts = df['unique_queries_avg']
tokens_counts = df['completion_tokens_avg']

x = np.arange(6)
width = 0.4

# Create plot summarizing rerun counts
fig1, ax1 = plt.subplots(figsize=(8, 6))
palette = sns.color_palette('hls', 2)
for i, (category, values) in enumerate(zip(['Rerun Count', 'Successful Rerun Count'], [rerun_counts, successful_rerun_counts])):
    bars = ax1.bar(x + i * width - (1.5 * width), values, width, label=category, color=palette[i])
    for bar in bars:
        height = bar.get_height()
        ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontname='Georgia', fontsize=10)

ax1.set_xlabel('Model Size', fontname='Georgia', fontsize=12)
ax1.set_ylabel('Values', fontname='Georgia', fontsize=12)
ax1.set_xticks(x - width)
ax1.set_xticklabels(model_sizes, fontname='Georgia', fontsize=10)
ax1.set_yticklabels(ax1.get_yticks(), fontname='Georgia', fontsize=10)
ax1.set_title('Mean Rerun Values', fontname='Georgia', fontsize=16)
legend = ax1.legend()
for text in legend.get_texts():
    text.set_fontname('Georgia')
    text.set_fontsize(12)

plt.tight_layout()
plt.show()

# Create plot summarizing query counts
fig2, ax2 = plt.subplots(figsize=(8, 6))
for i, (category, values) in enumerate(zip(['Query Count', 'Unique Query Count'], [query_counts, unique_query_counts])):
    bars = ax2.bar(x + i * width - (1.5 * width), values, width, label=category, color=palette[i])
    for bar in bars:
        height = bar.get_height()
        ax2.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontname='Georgia', fontsize=10)

ax2.set_xlabel('Model Size', fontname='Georgia', fontsize=12)
ax2.set_ylabel('Values', fontname='Georgia', fontsize=12)
ax2.set_xticks(x - width)
ax2.set_xticklabels(model_sizes, fontname='Georgia', fontsize=10)
ax2.set_yticklabels(ax2.get_yticks(), fontname='Georgia', fontsize=10)
ax2.set_title('Mean Query Values', fontname='Georgia', fontsize=16)
legend = ax2.legend()
for text in legend.get_texts():
    text.set_fontname('Georgia')
    text.set_fontsize(12)

plt.tight_layout()
plt.show()

# Create plot summarizing completion tokens use
fig1, ax1 = plt.subplots(figsize=(8, 6))
palette = sns.color_palette('hls', 1)
bars = ax1.bar(x - (1.5 * width), tokens_counts, width, label=category, color=palette[0])
for bar in bars:
    height = bar.get_height()
    ax1.annotate(f'{height:.2f}', xy=(bar.get_x() + bar.get_width() / 2, height), xytext=(0, 3), textcoords='offset points', ha='center', va='bottom', fontname='Georgia', fontsize=10)

ax1.set_xlabel('Model Size', fontname='Georgia', fontsize=12)
ax1.set_ylabel('Values', fontname='Georgia', fontsize=12)
ax1.set_xticks(x-width*1.5)
ax1.set_xticklabels(model_sizes, fontname='Georgia', fontsize=10)
ax1.set_yticklabels(ax1.get_yticks(), fontname='Georgia', fontsize=10)
ax1.set_title('Mean Completion Tokens Count', fontname='Georgia', fontsize=16)
for text in legend.get_texts():
    text.set_fontname('Georgia')
    text.set_fontsize(12)

plt.tight_layout()
plt.show()