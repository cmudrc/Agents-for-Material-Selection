import pandas as pd
import sklearn.metrics as metrics
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import OneHotEncoder
import statsmodels.formula.api as smf
import statsmodels.api as sm

##########################################################################################
# Code adapted from: https://stackoverflow.com/a/57239611

# Get regression metrics
def regression_results(y_true, y_pred, y_val, grouping):
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print(f"\n\033[1mRegression for {y_val} Results Grouped by {' and '.join(word.capitalize() for word in grouping)}:\033[0m")
    print('Explained variance:', round(explained_variance,4))
    print('Median absolute error:', round(median_absolute_error,4))
    print('r\u00b2:', round(r2,4))
    print('MAE:', round(mean_absolute_error,4))
    print('MSE:', round(mse,4))
    print('RMSE:', round(np.sqrt(mse),4))

##########################################################################################
# Code adapted from: https://saturncloud.io/blog/linear-regression-with-sklearn-using-categorical-variables/

def ols_regression(results_df, y_val, grouping):
    # Separate features (X) and target variable (y)
    x = results_df[['Size', 'Prompt Type']]
    y = results_df[y_val]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # One-Hot Encoding:
    encoder_one_hot = OneHotEncoder()
    x_train_one_hot = encoder_one_hot.fit_transform(x_train[['Prompt Type']])

    # Build linear regression model
    model_one_hot = LinearRegression().fit(x_train_one_hot, y_train)

    # Evaluate model on the test set
    x_test_one_hot = encoder_one_hot.transform(x_test[['Prompt Type']])
    y_pred = model_one_hot.predict(x_test_one_hot)
    regression_results(y_test, y_pred, y_val, grouping)

    # Get ordinary least squares summary
    x = sm.add_constant(x)
    results_df.columns = results_df.columns.str.replace(' ', '_')
    results_df.columns = results_df.columns.str.replace('-', '_')
    model = smf.ols(f"{y_val.replace('-', '_')} ~ Size + C(Prompt_Type)", data=results_df).fit()
    print(model.summary())

##########################################################################################

# Read survey results
survey_df = pd.read_csv('Results/Data/survey_responses_mapped.csv')
survey_df['material'] = survey_df['material'].replace('aluminium', 'aluminum')
survey_df = survey_df.dropna(how='any')

def zscore_regression(grouping):
    survey_stats = survey_df.groupby(grouping)['response'].agg(['mean', 'std']).reset_index()
    # Combine all data across model sizes and prompt types
    results_df = pd.DataFrame(columns=['Size', 'Prompt Type', 'Z-Score'])
    for modelsize in [1.5, 3, 7]:
        for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
            df = pd.read_csv(f'Results/Data/qwen_{str(modelsize)}B_{question_type}.csv')
            df['response'] = pd.to_numeric(df['response'], errors='coerce')
            merged_df = pd.merge(df, survey_stats, on=grouping, how='left')
            merged_df['Z-Score'] = (merged_df['response'] - merged_df['mean']) / merged_df['std']
            merged_stats = merged_df.groupby(grouping)['Z-Score'].agg(['mean']).reset_index()
            merged_stats.rename(columns={'mean':'Z-Score'}, inplace=True)
            merged_stats['Size'] = modelsize
            merged_stats['Prompt Type'] = question_type
            results_df = pd.concat([results_df, merged_stats[['Size', 'Prompt Type', 'Z-Score']]])
    
    ols_regression(results_df, 'Z-Score', grouping)

# Read in MAE data
mae_df = pd.read_csv('Results/Data/mean_error.csv')

def mae_regression(grouping):
    # Combine all data across model sizes and prompt types
    results_df = pd.DataFrame(columns=['Size', 'Prompt Type', 'MAE'])
    for modelsize in [1.5, 3, 7]:
        for question_type in ['agentic', 'zero-shot', 'few-shot', 'parallel', 'chain-of-thought']:
            df = mae_df[(mae_df['model_size'] == modelsize) & (mae_df['question_type'] == question_type)]
            stats_df = df.groupby(grouping)['mean_error'].agg(['mean']).reset_index()
            stats_df.rename(columns={'mean':'MAE'}, inplace=True)
            stats_df['Size'] = modelsize
            stats_df['Prompt Type'] = question_type
            results_df = pd.concat([results_df, stats_df[['Size', 'Prompt Type', 'MAE']]])
    results_df = results_df.dropna(how='any')

    ols_regression(results_df, 'MAE', grouping)

##########################################################################################

zscore_regression(['design', 'criteria'])
zscore_regression(['material'])
mae_regression(['design', 'criteria'])
mae_regression(['material'])