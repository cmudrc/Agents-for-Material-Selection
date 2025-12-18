import pandas as pd
from sklearn.metrics import r2_score

##########################################################################################

def calculate_rss(row, matching_responses):
    return ((matching_responses - row['response']) ** 2).mean() # actual - predicted squared

r2_results = []

# Load the survey responses CSV
survey_responses_df = pd.read_csv('Data/survey_responses_mapped.csv')

# Drop the rows with nan values
survey_responses_df = survey_responses_df.dropna()

# Find mean survey response for each design, criteria, and material combination
mean_survey_df = survey_responses_df.groupby(['design', 'criteria', 'material'])['response'].mean().reset_index()

for modelsize in [1.7, 4, 8, 14, 32]:
    for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:

        # Load model data
        model_data_df = pd.read_csv(f'Data/qwen3_{modelsize}B_{question_type}.csv')
        model_data_df.dropna(subset=['response'], inplace=True)
        
        # Join model data with corresponding mean survey responses
        merged_df = pd.merge(
            model_data_df, 
            mean_survey_df,
            on=['design', 'criteria', 'material'], 
            how='inner',
            suffixes=('', '_survey')
        )

        # Calculate r^2
        r2 = r2_score(merged_df['response_survey'], merged_df['response'])

        # Save the r^2 value in the results array
        r2_results.append({'size': modelsize,
                            'prompt type': question_type,
                            'r2': r2})
                    
# Save the results to a CSV file
r2_results_df = pd.DataFrame(r2_results)
r2_results_df = r2_results_df.dropna()
r2_results_df.to_csv('Data Evaluation/Results/r2_material.csv', index=False)