import pandas as pd

##########################################################################################

omissions_df = pd.DataFrame(columns=['size', 'omissions', 'percent omitted'])

for modelsize in [1.7, 4, 8, 14, 32]:
    model_data_df = pd.read_csv(f'Data/qwen3_{modelsize}B_agentic.csv')
    omissions = model_data_df['response'].isna().sum()
    new_row = pd.DataFrame([{'size': modelsize, 'omissions': omissions,  'percent omitted': omissions / 144}])
    omissions_df = pd.concat([omissions_df, new_row], ignore_index=True)

omissions_df.to_csv(f'Data Evaluation/Results/omissions.csv', index=False)