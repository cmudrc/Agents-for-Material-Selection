import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##########################################################################################

# Read survey results
combined_df = pd.read_csv("Data/survey_responses_mapped.csv")
combined_df = combined_df.dropna(how="any")
combined_df["material"] = combined_df["material"].replace("aluminium", "aluminum")
combined_df["Size"] = "Survey"
combined_df["Prompt Type"] = "Survey"
palette = sns.color_palette("hls", 6)

# Read all LLM results
for modelsize in ['1.5B', '3B', '7B', '32B', '72B']:
    for question_type in ['Agentic', 'Zero-Shot', 'Few-Shot', 'Parallel', 'Chain-of-Thought']:
        df = pd.read_csv(f"Data/qwen_{modelsize.lower()}_{question_type.lower()}.csv")
        df = df.dropna(how="any")
        df['Size'] = modelsize
        df['Prompt Type'] = question_type
        combined_df = pd.concat([combined_df, df])

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="Size", y="response", data=combined_df, palette=palette)
plt.title("Scores by Model Size")
plt.ylabel("Score")
plt.xlabel("Model Size")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
ax = sns.boxplot(x="Prompt Type", y="response", data=combined_df, palette=palette)
plt.title("Scores by Prompt Type")
plt.ylabel("Score")
plt.xlabel("Prompt Type")
plt.tight_layout()
plt.show()