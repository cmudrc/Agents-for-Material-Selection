import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb

##########################################################################################

# Read survey results
combined_df = pd.read_csv("Data/survey_responses_mapped.csv")
combined_df = combined_df.dropna(how="any")
combined_df["material"] = combined_df["material"].replace("aluminium", "aluminum")
combined_df["Size"] = "Survey"
combined_df["Prompt Type"] = "Survey"
palette = sns.color_palette("hls", 6)
palette = [to_rgb(color) for color in palette]
palette = [(min(1, r + 0.15), min(1, g + 0.15), min(1, b + 0.15)) for r, g, b in palette]

# Read all LLM results
for modelsize in ['1.5B', '3B', '7B', '32B', '72B']:
    for question_type in ['Agentic', 'Zero-Shot', 'Few-Shot', 'Parallel', 'Chain-of-Thought']:
        df = pd.read_csv(f"Data/qwen_{modelsize.lower()}_{question_type.lower()}.csv")
        df = df.dropna(how="any")
        df['Size'] = modelsize
        df['Prompt Type'] = question_type
        combined_df = pd.concat([combined_df, df])

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="Size", y="response", data=combined_df, palette=palette)

size_order = ['Survey', '1.5B', '3B', '7B', '32B', '72B']
medians = combined_df.groupby('Size')['response'].median().reindex(size_order)
positions = range(len(size_order))
for pos, (size, median) in zip(positions, medians.items()):
    ax.text(pos, median - 0.25, f'{int(median)}', horizontalalignment='center', verticalalignment='top', color='black')
    
plt.title("Scores by Model Size", fontname='Georgia', fontsize=16)
plt.ylabel("Score", fontname='Georgia', fontsize=12)
plt.xlabel("Model Size", fontname='Georgia', fontsize=12)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="Prompt Type", y="response", data=combined_df, palette=palette)

prompt_order = ['Survey', 'Agentic', 'Zero-Shot', 'Few-Shot', 'Parallel', 'Chain-of-Thought']
medians = combined_df.groupby('Prompt Type')['response'].median().reindex(prompt_order)
positions = range(len(prompt_order))
for pos, (ptype, median) in zip(positions, medians.items()):
    ax.text(pos, median - 0.25, f'{int(median)}', horizontalalignment='center', verticalalignment='top', color='black')
    
plt.title("Scores by Prompt Type", fontname='Georgia', fontsize=16)
plt.ylabel("Score", fontname='Georgia', fontsize=12)
plt.xlabel("Prompt Type", fontname='Georgia', fontsize=12)
plt.tight_layout()
plt.show()