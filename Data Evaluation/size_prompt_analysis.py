import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import iqr
from collections import defaultdict

##########################################################################################

# Keep track of figure for saving plots
figure = 1

def plot(num, df, grouping, y_val, modelsize, question_type):
    global figure
    palette = sns.color_palette('hls', num)
    if num == 30: figsize = (20, 12)
    elif num == 10: figsize = (14, 8)
    else: figsize = (10, 6)
    plt.figure(figsize=figsize)
    if modelsize: df = df[df['Model Size and Prompt Type'].str.startswith(f'{modelsize}B')]
    elif question_type: df = df[df['Model Size and Prompt Type'].str.endswith(question_type)]
    ax = sns.boxplot(x='Model Size and Prompt Type', y=y_val, data=df, palette=palette)
    if modelsize:
        new_labels = [prompt_type_rename[label.split("\n")[1]] for label in df['Model Size and Prompt Type'].unique()]
        plt.title(f'{y_val} for {modelsize}B Model Grouped By {' and '.join(word.capitalize() for word in grouping)}')
        plt.xlabel("Prompt Type")
    elif question_type:
        new_labels = [label.split("\n")[0] for label in df['Model Size and Prompt Type'].unique()]
        plt.title(f'{y_val} for {prompt_type_rename[question_type]} Prompting Grouped By {' and '.join(word.capitalize() for word in grouping)}')
        plt.xlabel("Model Size")
    plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=45)
    # plt.savefig(f'figure{figure}')
    figure += 1

def plot_prompt(num, df, survey_df, grouping, y_val):
    global figure
    palette = sns.color_palette('hls', num)
    plt.figure(figsize=(10, 6))
    df[['Model Size', 'Prompt Type']] = df['Model Size and Prompt Type'].str.split("\n", expand=True)
    ax = sns.boxplot(x='Prompt Type', y=y_val, hue='Model Size', data=df, palette=palette)
    # sns.scatterplot(x='Prompt Type', y='response', data=survey_df, color='black', s=100, label='Survey Mean')
    new_labels = [prompt_type_rename[label] for label in df['Prompt Type'].unique()]
    plt.title(f'{y_val} for Results Grouped By {' and '.join(word.capitalize() for word in grouping)}')
    plt.xlabel("Prompt Type")
    plt.xticks(ticks=range(len(new_labels)), labels=new_labels, rotation=0)
    figure += 1

##########################################################################################

def size_prompt_analysis(grouping):
    mae_dict = defaultdict(list)
    zscore_dict = defaultdict(list)

    # Add MAE values from previously generated csv file
    mae_df = pd.read_csv('Data Evaluation/Results/mean_error.csv')
    for modelsize in [1.5, 3, 7, 32, 72]:
        for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
            df = mae_df[(mae_df['model_size'] == modelsize) & (mae_df['question_type'] == question_type)]
            df = df.dropna(how='any')
            stats_df = df.groupby(grouping)['mean_error'].agg(['mean']).reset_index()
            for mean_error in list(stats_df['mean']):
                mae_dict[str(modelsize)+'B\n'+question_type].append(mean_error)

    # Read survey results
    survey_df = pd.read_csv('Data/survey_responses_mapped.csv')
    survey_df['material'] = survey_df['material'].replace('aluminium', 'aluminum')
    survey_df = survey_df.dropna(how='any')
    survey_stats = survey_df.groupby(grouping)['response'].agg(['mean', 'std', lambda x: iqr(x)]).rename(columns={'<lambda_0>': 'iqr'}).reset_index()

    # Read LLM results
    for modelsize in ['1.5', '3', '7', '32', '72']:
        for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
            df = pd.read_csv(f'Data/qwen_{modelsize}B_{question_type}.csv')
            df['response'] = pd.to_numeric(df['response'], errors='coerce')
            df_stats = df.groupby(grouping)['response'].agg(['mean', 'std']).reset_index()
            merged_df = pd.merge(df, survey_stats, on=grouping, how='left')
            merged_df['z-score'] = (merged_df['response'] - merged_df['mean']) / merged_df['std']
            merged_stats = merged_df.groupby(grouping)['z-score'].agg(['mean']).reset_index()
            for zscore in list(merged_stats['mean']):
                zscore_dict[modelsize+'B\n'+question_type].append(zscore)

    # Create dataframe of mae values
    mae_values = []
    labels = []
    for key, values in mae_dict.items():
        mae_values.extend(values)
        labels.extend([key] * len(values))
    mae_df = pd.DataFrame({'Model Size and Prompt Type': labels, 'MAEs': mae_values})

    # Create dataframe of z-score values
    zscore_values = []
    labels = []
    for key, values in zscore_dict.items():
        zscore_values.extend(values)
        labels.extend([key] * len(values))
    zscore_df = pd.DataFrame({'Model Size and Prompt Type': labels, 'Z-Scores': zscore_values})

    plot_prompt(5, mae_df, survey_df, grouping, "MAEs")
    plot_prompt(5, zscore_df, survey_df, grouping, "Z-Scores")

    # # Plot by size
    # for modelsize in ['1.5', '3', '7', '32', '72']:
    #     plot(5, mae_df, grouping, "MAEs", modelsize, None)
    #     plot(5, zscore_df, grouping, "Z-Scores", modelsize, None)

    # # Plot by prompt type
    # for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
    #     plot(5, mae_df, grouping, "MAEs", None, question_type)
    #     plot(5, zscore_df, grouping, "Z-Scores", None, question_type)

##########################################################################################

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
prompt_type_rename = {'agentic': 'Agentic', 'zero-shot': 'Zero-Shot', 'few-shot': 'Few-Shot', 'parallel': 'Parallel', 'chain-of-thought': 'Chain-of-Thought'}

size_prompt_analysis(['design', 'criteria'])
size_prompt_analysis(['material'])
plt.tight_layout()
plt.show()