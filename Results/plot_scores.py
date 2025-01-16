import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

##########################################################################################

combined_df = pd.DataFrame()

# read survey results
survey_df = pd.read_csv("Results/Data/survey_responses_mapped.csv")
survey_df = survey_df.dropna(how="any")
survey_df["material"] = survey_df["material"].replace("aluminium", "aluminum")
survey_df["type"] = "Survey"
combined_df = pd.concat([combined_df, survey_df])

# read agent results
for model in ["Llama 3B", "Qwen 1.5b", "Qwen 3B", "Qwen 7B"]:
    df = pd.read_csv(f"Results/Data/{model.lower().replace(" ", "_")}.csv")
    df["type"] = f"{model}\nModel"
    combined_df = pd.concat([combined_df, df])

# read previously generated mechgpt, gpt-4, and mixtral results
for model in ["MechGPT", "GPT-4", "Mixtral"]:
    for question_type in ["Zero-Shot", "Few-Shot"]:
        df = pd.read_csv(f"Results/Data/{question_type.lower()}_{model.lower()}.csv")
        df = df.dropna(how="any")
        df["material"] = df["material"].replace("aluminium", "aluminum")
        df["type"] = f"{question_type}\n{model}"
        combined_df = pd.concat([combined_df, df])

# compare responses
plt.figure(figsize=(14, 8))
palette = sns.color_palette("hls", 11)
ax = sns.boxplot(x="type", y="response", data=combined_df, palette=palette)
plt.title("Scores by Type")

# compare scores by material for zero-shot prompting
palette = sns.color_palette("hls", 6)
plt.figure(figsize=(10,6))
curr_df = combined_df[combined_df["type"].isin(["Survey", "Llama 3B\nModel", "Qwen 3B\nModel", "Zero-Shot\nGPT-4", "Zero-Shot\nMixtral", "Zero-Shot\nMechGPT"])]
ax = sns.boxplot(x="material", y="response", hue="type", data=curr_df, palette=palette)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
plt.tight_layout()
plt.title("Scores by Material")

# compare scores by material for few-shot prompting
plt.figure(figsize=(10,6))
curr_df = combined_df[combined_df["type"].isin(["Survey", "Llama 3B\nModel", "Qwen 3B\nModel", "Few-Shot\nGPT-4", "Few-Shot\nMixtral", "Few-Shot\nMechGPT"])]
palette = sns.color_palette("hls", 6)
ax = sns.boxplot(x="material", y="response", hue="type", data=curr_df, palette=palette)
sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1.1), ncol=3, title=None, frameon=False)
plt.tight_layout()
plt.title("Scores by Material")
plt.show()