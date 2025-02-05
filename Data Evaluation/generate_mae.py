import pandas as pd

##########################################################################################
# Code adapted from: https://github.com/grndnl/llm_material_selection_jcise/blob/main/evaluation/mean_error_per_design_criteria_material.py

def calculate_mean_error(row, survey_df):
    # Filter the survey responses for matching design, criteria, and material
    matching_responses = survey_df[(survey_df['design'] == row['design']) &
                                   (survey_df['criteria'] == row['criteria']) &
                                   (survey_df['material'] == row['material'])]['response']

    # Calculate the absolute distance between the generated value and each survey response
    distances = row['response'] - matching_responses

    # Calculate and return the mean of these distances
    return distances.mean()

results = []

# Load the survey responses CSV
survey_responses_df = pd.read_csv('Data/survey_responses_mapped.csv')

# drop the rows with nan values
survey_responses_df = survey_responses_df.dropna()

# Group the survey responses by 'design', 'criteria', 'material' and calculate mean and std
grouped_survey_stats = survey_responses_df.groupby(['design', 'criteria', 'material'])['response'].agg(['mean', 'std']).reset_index()

for modelsize in [1.5, 3, 7, 14, 32, 72]:
    for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
        for design in ['kitchen utensil grip', 'spacecraft component', 'underwater component', 'safety helmet']:
            for criteria in ['lightweight', 'heat resistant', 'corrosion resistant', 'high strength']:
                for material in ["steel", "aluminum", "titanium", "glass", "wood", "thermoplastic", "elastomer", "thermoset", "composite"]:
                    # load model data
                    model_data_df = pd.read_csv(f'Data/qwen_{str(modelsize)}B_{question_type}.csv')

                    # filter data
                    model_data_df = model_data_df[model_data_df['design'] == design]
                    model_data_df = model_data_df[model_data_df['criteria'] == criteria]
                    model_data_df = model_data_df[model_data_df['material'] == material]

                    # # clean the response column by only taking the first digits in the string
                    # try:
                    #     model_data_df['response'] = model_data_df['response'].str.extract(r'(\d+)').astype(int)
                    # except:
                    #     pass

                    # # floor all values above 10 to 10, and all values below 0 to 0
                    # model_data_df['response'] = model_data_df['response'].apply(lambda x: min(10, max(0, x)))

                    model_data_df['mean_distance'] = model_data_df.apply(lambda row: calculate_mean_error(row, survey_responses_df), axis=1)

                    # Save the mean error in the results array
                    mean_error = model_data_df['mean_distance'].mean()
                    results.append({'model_size': modelsize,
                                    'question_type': question_type,
                                    'design': design,
                                    'criteria': criteria,
                                    'material': material,
                                    'mean_error': mean_error})
                    
# Save the results to a CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('Data/mean_error.csv', index=False)