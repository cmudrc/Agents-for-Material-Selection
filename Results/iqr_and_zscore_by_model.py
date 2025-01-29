import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr
from collections import defaultdict

##########################################################################################

# Analyze data based on grouping
def iqr_and_zscore_by_model(grouping):
    iqr_dict = defaultdict(list)
    zscore_dict = defaultdict(list)

    # Read survey results
    survey_df = pd.read_csv('Results/Data/survey_responses_mapped.csv')
    survey_df['material'] = survey_df['material'].replace('aluminium', 'aluminum')
    survey_df = survey_df.dropna(how='any')
    survey_stats = survey_df.groupby(grouping)['response'].agg(['mean', 'std', lambda x: iqr(x)]).rename(columns={'<lambda_0>': 'iqr'}).reset_index()

    # Read agent results
    for model in ['Llama 3B', 'Qwen 3B']:
        data_df = pd.read_csv(f'Results/Data/{model.lower().replace(' ', '_')}.csv')
        model = model + '\nAgent'
        data_df['Type'] = model
        stats_df = data_df.groupby(grouping)['response'].agg(['mean', 'std', lambda x: iqr(x)]).rename(columns={'<lambda_0>': 'iqr'}).reset_index()
        for iqr_value in list(stats_df['iqr']):
            iqr_dict[model].append(iqr_value)
        merged_df = pd.merge(data_df, survey_stats, on=grouping, how='left')
        merged_df['z-score'] = (merged_df['response'] - merged_df['mean']) / merged_df['std']
        merged_stats = merged_df.groupby(grouping)['z-score'].agg(['mean']).reset_index()
        for zscore in list(merged_stats['mean']):
            zscore_dict[model].append(zscore)

    # Read previously generated gpt-4, mixtral, and mechgpt results
    for type in ['Zero-Shot', 'Few-Shot']:
        for model in ['GPT-4', 'Mixtral', 'MechGPT']:
            data_df = pd.read_csv(f'Results/Data/{type.lower() + '_' + model.lower()}.csv')
            data_df = data_df.dropna(how='any')
            data_df['material'] = data_df['material'].replace('aluminium', 'aluminum')
            type_model = type + '\n' + model
            data_df['Type'] = type_model
            stats_df = data_df.groupby(grouping)['response'].agg(['mean', 'std', lambda x: iqr(x)]).rename(columns={'<lambda_0>': 'iqr'}).reset_index()
            for iqr_value in list(stats_df['iqr']):
                iqr_dict[type_model].append(iqr_value)
            merged_df = pd.merge(data_df, survey_stats, on=grouping, how='left')
            merged_df['z-score'] = (merged_df['response'] - merged_df['mean']) / merged_df['std']
            merged_stats = merged_df.groupby(grouping)['z-score'].agg(['mean']).reset_index()
            for zscore in list(merged_stats['mean']):
                zscore_dict[type_model].append(zscore)

    # Create dataframe of iqr values
    iqr_values = []
    labels = []
    for key, values in iqr_dict.items():
        iqr_values.extend(values)
        labels.extend([key] * len(values))
    iqr_df = pd.DataFrame({'Model and Prompt Type': labels, 'IQRs': iqr_values})

    # Create dataframe of z-score values
    zscore_values = []
    labels = []
    for key, values in zscore_dict.items():
        zscore_values.extend(values)
        labels.extend([key] * len(values))
    zscore_df = pd.DataFrame({'Model and Prompt Type': labels, 'Z-Scores': zscore_values})

    # Plot iqr by model and prompt type
    palette = sns.color_palette('hls', 8)
    plt.figure(figsize=(10,6))
    ax = sns.boxplot(x='Model and Prompt Type', y='IQRs', data=iqr_df, palette=palette)
    plt.title(f'Score Distribution Grouped By {' and '.join(word.capitalize() for word in grouping)}')
    plt.tight_layout()

    # Plot iqr by model and prompt type
    plt.figure(figsize=(10,6))
    ax = sns.boxplot(x='Model and Prompt Type', y='Z-Scores', data=zscore_df, palette=palette)
    plt.title(f'Score Proximity to Survey Results Grouped By {' and '.join(word.capitalize() for word in grouping)}')
    plt.tight_layout()

##########################################################################################

iqr_and_zscore_by_model(['design', 'criteria'])
iqr_and_zscore_by_model(['material'])
plt.show()