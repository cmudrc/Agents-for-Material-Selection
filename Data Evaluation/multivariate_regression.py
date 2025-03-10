import pandas as pd
import sklearn.metrics as metrics
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import OneHotEncoder
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.patches import Patch

##########################################################################################
# Code adapted from: https://stackoverflow.com/a/57239611 and https://www.geeksforgeeks.org/extracting-regression-coefficients-from-statsmodelsapi/#

# Get regression metrics
def regression_results(y_true, y_pred, y_val):
    explained_variance=metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error=metrics.mean_absolute_error(y_true, y_pred) 
    mse=metrics.mean_squared_error(y_true, y_pred) 
    median_absolute_error=metrics.median_absolute_error(y_true, y_pred)
    r2=metrics.r2_score(y_true, y_pred)
    print(f'\n\033[1mRegression for {y_val}:\033[0m')
    print('Explained variance:', round(explained_variance,4))
    print('Median absolute error:', round(median_absolute_error,4))
    print('r\u00b2:', round(r2,4))
    print('MAE:', round(mean_absolute_error,4))
    print('MSE:', round(mse,4))
    print('RMSE:', round(np.sqrt(mse),4))


##########################################################################################

def get_coef_table(lin_reg):
    coef_df = pd.DataFrame({
        'coef': lin_reg.params.values,
        'pvalue': lin_reg.pvalues.round(4),
        'ci_lower': lin_reg.conf_int()[0],
        'ci_upper': lin_reg.conf_int()[1]
    }, index=lin_reg.params.index)
    return coef_df

##########################################################################################
# Code adapted from: https://saturncloud.io/blog/linear-regression-with-sklearn-using-categorical-variables/

def ols_regression(results_df, y_val):
    # Separate features (X) and target variable (y)
    x = results_df[['size', 'prompt type']]
    y = results_df[y_val.lower()]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # One-Hot Encoding:
    encoder_one_hot = OneHotEncoder()
    x_train_one_hot = encoder_one_hot.fit_transform(x_train[['prompt type']])

    # Build linear regression model
    model_one_hot = LinearRegression().fit(x_train_one_hot, y_train)

    # Evaluate model on the test set
    x_test_one_hot = encoder_one_hot.transform(x_test[['prompt type']])
    y_pred = model_one_hot.predict(x_test_one_hot)
    regression_results(y_test, y_pred, y_val)

    # Get ordinary least squares summary
    x = sm.add_constant(x)
    results_df.columns = results_df.columns.str.replace(' ', '_')
    results_df.columns = results_df.columns.str.replace('-', '_')
    formula = f'{y_val.lower().replace('-', '_')} ~ size + C(prompt_type) + (size * C(prompt_type))'
    model = smf.ols(formula, data=results_df).fit()
    coef_df = get_coef_table(model)
    coef_df.to_csv(f'Data Evaluation/Results/{y_val.lower()}_coefficients.csv')
    print(model.summary())
    plot_contributions(coef_df, y_val)

##########################################################################################

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 14

def plot_contributions(coef_df, y_val):
    palette = sns.color_palette('hls', 2)

    # Extract renamed terms, coefficients, and p-values
    coef_df = coef_df.drop(index='Intercept', errors='ignore')
    contributions = coef_df['coef'].abs()
    coef_df = coef_df.loc[contributions.sort_values(ascending=True).index]
    contributions = coef_df['coef'].abs()
    rename_dict = {
        'C(prompt_type)[T.chain-of-thought]': 'Cᴄ',
        'C(prompt_type)[T.few-shot]': 'Cꜰ',
        'C(prompt_type)[T.parallel]': 'Cᴘ',
        'C(prompt_type)[T.zero-shot]': 'Cᴢ',
        'size': 'S',
        'size:C(prompt_type)[T.chain-of-thought]': 'Cᴄ × S',
        'size:C(prompt_type)[T.few-shot]': 'Cꜰ × S',
        'size:C(prompt_type)[T.parallel]': 'Cᴘ × S',
        'size:C(prompt_type)[T.zero-shot]': 'Cᴢ × S'
    }
    coef_df.index = coef_df.index.map(lambda x: rename_dict.get(x, x))
    terms = coef_df.index

    # Create significance labels
    def generate_significance_labels(p_values):
        sig_labels = []
        for p in p_values:
            if p < 0.01:
                sig_labels.append('***')
            elif p < 0.05:
                sig_labels.append('**')
            elif p < 0.1:
                sig_labels.append('*')
            else:
                sig_labels.append('')
        return sig_labels

    # Separate plots for size vs. prompt type for better scaling
    terms1 = [term for term in coef_df.index if term in ['Cᴢ', 'Cꜰ', 'Cᴘ', 'Cᴄ']]
    df1 = coef_df.loc[coef_df.index.isin(terms1)]
    contributions1 = df1['coef'].abs()
    colors1 = [palette[1] if val > 0 else palette[0] for val in df1['coef']]
    pvalues1 = df1['pvalue']
    sig_labels1 = generate_significance_labels(pvalues1)

    terms2 = [term for term in terms if term not in terms1]
    df2 = coef_df.loc[terms2]
    contributions2 = df2['coef'].abs()
    colors2 = [palette[1] if val > 0 else palette[0] for val in df2['coef']]
    pvalues2 = df2['pvalue']
    sig_labels2 = generate_significance_labels(pvalues2)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    bars = axes[0].barh(terms1, contributions1, color=colors1, edgecolor='black')

    # Plot significance labels
    for bar, label in zip(bars, sig_labels1):
        width = bar.get_width()
        axes[0].text(width + 0.003, bar.get_y() + bar.get_height() / 2, label, ha='left', va='center', fontsize=16, fontweight='bold')

    axes[0].set_xlabel('Coefficient Value', fontname='Georgia', fontsize=16)
    axes[0].set_xlim([0, max(contributions1) + 0.05])
    bars = axes[1].barh(terms2, contributions2, color=colors2, edgecolor='black')

    for bar, label in zip(bars, sig_labels2):
        width = bar.get_width()
        axes[1].text(width + 0.0001, bar.get_y() + bar.get_height() / 2, label, ha='left', va='center', fontsize=16, fontweight='bold')

    axes[1].set_xlabel('Coefficient Value', fontname='Georgia', fontsize=16)
    axes[1].set_xlim([0, max(contributions2) + 0.0005])
    legend_elements = [Patch(facecolor=palette[1], edgecolor='black', label='Positive'), Patch(facecolor=palette[0], edgecolor='black', label='Negative')]
    fig.legend(bbox_to_anchor=(1.002, 0.9), handles=legend_elements, loc='upper right', fontsize=16)

    # Add box explaining stars and terms
    explanation_text = (
        '* :   p < 0.1\n'
        '** :  p < 0.05\n'
        '*** : p < 0.01\n\n'
        'S:  Size\n'
        'Cᴄ:  Chain-of-\n       Thought\n'
        'Cꜰ:  Few-shot\n'
        'Cᴘ:  Parallel\n'
        'Cᴢ:  Zero-shot'
    )

    axes[1].text(1.05, 0.81, explanation_text, transform=axes[1].transAxes, fontsize=16, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', edgecolor='lightgrey', boxstyle='square,pad=0.5'))
    plt.suptitle(f'Contribution to Predicting {y_val}', fontname='Georgia', fontsize=20)

    plt.tight_layout()
    plt.savefig(f'regression_{y_val.lower()}.png')
    plt.show()

##########################################################################################

results_df = pd.read_csv('Data Evaluation/Results/z-score_material.csv')
ols_regression(results_df, 'Z-Score')

results_df = pd.read_csv('Data Evaluation/Results/mae_material.csv')
ols_regression(results_df, 'MAE')