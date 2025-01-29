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
    if num == 12: figsize = (14, 8)
    elif num == 5: figsize = (10, 6)
    else: figsize = (8, 5)
    plt.figure(figsize=figsize)
    if modelsize != None: df = df[df['Model Size and Prompt Type'].str.startswith(modelsize)]
    elif question_type != None: df = df[df['Model Size and Prompt Type'].str.endswith(question_type)]
    ax = sns.boxplot(x='Model Size and Prompt Type', y=y_val, data=df, palette=palette)
    plt.title(f'{y_val} Grouped By {' and '.join(word.capitalize() for word in grouping)}')
    plt.tight_layout()
    plt.savefig(f'figure{figure}')
    figure += 1

##########################################################################################

def size_prompt_analysis(grouping):
    mae_dict = defaultdict(list)
    # iqr_dict = defaultdict(list)
    zscore_dict = defaultdict(list)

    # Add MAE values from previously generated csv file
    mae_df = pd.read_csv('Results/Data/mean_error.csv')
    for modelsize in [1.5, 3, 7]:
        for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
            df = mae_df[(mae_df['model_size'] == modelsize) & (mae_df['question_type'] == question_type)]
            df = df.dropna(how='any')
            stats_df = df.groupby(grouping)['mean_error'].agg(['mean']).reset_index()
            for mean_error in list(stats_df['mean']):
                mae_dict[str(modelsize)+'B\n'+question_type].append(mean_error)

    # Read survey results
    survey_df = pd.read_csv('Results/Data/survey_responses_mapped.csv')
    survey_df['material'] = survey_df['material'].replace('aluminium', 'aluminum')
    survey_df = survey_df.dropna(how='any')
    survey_stats = survey_df.groupby(grouping)['response'].agg(['mean', 'std', lambda x: iqr(x)]).rename(columns={'<lambda_0>': 'iqr'}).reset_index()

    # Read LLM results
    for modelsize in ['1.5', '3', '7']:
        for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
            df = pd.read_csv(f'Results/Data/qwen_{modelsize}B_{question_type}.csv')
            df['response'] = pd.to_numeric(df['response'], errors='coerce')
            df_stats = df.groupby(grouping)['response'].agg(['mean', 'std']).reset_index()
            # df_stats = df.groupby(grouping)['response'].agg(['mean', 'std', lambda x: iqr(x)]).rename(columns={'<lambda_0>': 'iqr'}).reset_index()
            # for iqr_value in list(df_stats['iqr']):
            #     iqr_dict[modelsize+'B\n'+question_type].append(iqr_value)
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

    # # Create dataframe of IQR values
    # iqr_values = []
    # labels = []
    # for key, values in iqr_dict.items():
    #     iqr_values.extend(values)
    #     labels.extend([key] * len(values))
    # iqr_df = pd.DataFrame({'Model Size and Prompt Type': labels, 'IQRs': iqr_values})

    # Create dataframe of z-score values
    zscore_values = []
    labels = []
    for key, values in zscore_dict.items():
        zscore_values.extend(values)
        labels.extend([key] * len(values))
    zscore_df = pd.DataFrame({'Model Size and Prompt Type': labels, 'Z-Scores': zscore_values})

    plot(12, mae_df, grouping, "MAEs", None, None)
    # plot(12, iqr_df, grouping, "IQRs")
    plot(12, zscore_df, grouping, "Z-Scores", None, None)

    # Plot by size
    for modelsize in ['1.5', '3', '7']:
        plot(5, mae_df, grouping, "MAEs", modelsize, None)
        # plot(5, iqr_df, grouping, "IQRs", modelsize, None)
        plot(5, zscore_df, grouping, "Z-Scores", modelsize, None)

    # Plot by prompt type
    for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
        plot(3, mae_df, grouping, "MAEs", None, question_type)
        # plot(3, iqr_df, grouping, "IQRs", None, question_type)
        plot(3, zscore_df, grouping, "Z-Scores", None, question_type)

##########################################################################################

size_prompt_analysis(['design', 'criteria'])
size_prompt_analysis(['material'])