import pandas as pd

##########################################################################################

# Read survey results
survey_df = pd.read_csv('Data/survey_responses_mapped.csv')
survey_df['material'] = survey_df['material'].replace('aluminium', 'aluminum')
survey_df = survey_df.dropna(how='any')
survey_stats = survey_df.groupby('material')['response'].agg(['mean', 'std']).reset_index()

# Combine all data across model sizes and prompt types
results_df = pd.DataFrame(columns=['size', 'prompt type', 'z-score'])
for modelsize in [1.7, 4, 8, 14, 32]:
    for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
        df = pd.read_csv(f'Data/qwen3_{str(modelsize)}B_{question_type}.csv')
        df = df.dropna()
        df['response'] = pd.to_numeric(df['response'], errors='coerce')
        merged_df = pd.merge(df, survey_stats, on='material', how='left')
        merged_df['z-score'] = (merged_df['response'] - merged_df['mean']) / merged_df['std']
        merged_stats = merged_df.groupby('material')['z-score'].agg(['mean']).reset_index()
        merged_stats.rename(columns={'mean':'z-score'}, inplace=True)
        merged_stats['size'] = modelsize
        merged_stats['prompt type'] = question_type
        results_df = pd.concat([results_df, merged_stats[['size', 'prompt type', 'z-score']]])
results_df.to_csv(f'Data Evaluation/Results/z-score_material.csv', index=False)

# Read in MAE data
mae_df = pd.read_csv('Data Evaluation/Results/mean_error.csv')

# Combine all data across model sizes and prompt types
results_df = pd.DataFrame(columns=['size', 'prompt type', 'mae'])
for modelsize in [1.7, 4, 8, 14, 32]:
    for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
        df = mae_df[(mae_df['model_size'] == modelsize) & (mae_df['question_type'] == question_type)]
        stats_df = df.groupby('material')['mean_error'].agg(['mean']).reset_index()
        stats_df.rename(columns={'mean':'mae'}, inplace=True)
        stats_df['size'] = modelsize
        stats_df['prompt type'] = question_type
        results_df = pd.concat([results_df, stats_df[['size', 'prompt type', 'mae']]])
results_df = results_df.dropna(how='any')
results_df.to_csv(f'Data Evaluation/Results/mae_material.csv', index=False)