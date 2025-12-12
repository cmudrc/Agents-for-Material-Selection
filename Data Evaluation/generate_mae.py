import pandas as pd

##########################################################################################
# Code adapted from: https://github.com/grndnl/llm_material_selection_jcise/blob/main/evaluation/mean_error_per_design_criteria_material.py

def calculate_mean_error(row, survey_df):
    # Filter the survey responses for matching design, criteria, and material
    matching_responses = survey_df[(survey_df['design'] == row['design']) &
                                   (survey_df['criteria'] == row['criteria']) &
                                   (survey_df['material'] == row['material'])]['response']

    # Calculate the absolute distance between the generated value and each survey response
    distances = abs(row['response'] - matching_responses)

    # Calculate and return the mean of these distances
    return distances.mean()

mae_results = []

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

                    filtered_df['mean_distance'] = filtered_df.apply(lambda row: calculate_mean_error(row, survey_responses_df), axis=1)

                    # Save the mean error in the results array
                    mean_error = filtered_df['mean_distance'].mean()
                    mae_results.append({'model_size': modelsize,
                                    'question_type': question_type,
                                    'design': design,
                                    'criteria': criteria,
                                    'material': material,
                                    'mean_error': mean_error})
                    
# Save the results to a CSV file
mae_results_df = pd.DataFrame(mae_results)
mae_results_df = mae_results_df.dropna()
mae_results_df.to_csv('Data Evaluation/Results/mean_error.csv', index=False)