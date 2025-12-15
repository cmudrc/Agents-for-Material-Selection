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

for modelsize in [1.7, 4, 8, 14, 32]:
    for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
        # load model data
        model_data_df = pd.read_csv(f'Data/qwen3_{modelsize}B_{question_type}.csv')
        
        for design in ['kitchen utensil grip', 'spacecraft component', 'underwater component', 'safety helmet']:
            for criteria in ['lightweight', 'heat resistant', 'corrosion resistant', 'high strength']:
                for material in ["steel", "aluminum", "titanium", "glass", "wood", "thermoplastic", "elastomer", "thermoset", "composite"]:

                    # filter data
                    filtered_df = model_data_df[
                        (model_data_df['design'] == design) & 
                        (model_data_df['criteria'] == criteria) &
                        (model_data_df['material'] == material)
                    ].dropna()

                    # Filter the survey responses for matching design, criteria, and material
                    matching_responses = survey_responses_df[(survey_responses_df['design'] == design) &
                                                (survey_responses_df['criteria'] == criteria) &
                                                (survey_responses_df['material'] == material)]['response'] 
                    
                    # Calculate average RSS for each design, criteria, and material combination
                    filtered_df['rss'] = filtered_df.apply(lambda row: calculate_rss(row, matching_responses), axis=1)

                    # Calculate TSS
                    tss = ((matching_responses - matching_responses.mean()) ** 2).sum()
                    
                    # Calculate r^2
                    r2 = 1 - (filtered_df['rss'].sum() / tss)
                    
                    # Save the r^2 value in the results array
                    r2_results.append({'model_size': modelsize,
                                    'question_type': question_type,
                                    'design': design,
                                    'criteria': criteria,
                                    'material': material,
                                    'r2': r2})
                    
# Save the results to a CSV file
r2_results_df = pd.DataFrame(r2_results)
r2_results_df = r2_results_df.dropna()
r2_results_df.to_csv('Data Evaluation/Results/r2.csv', index=False)