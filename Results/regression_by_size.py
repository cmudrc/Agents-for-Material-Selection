import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import iqr
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress

##########################################################################################

def regression_by_size(val_list, y_val):
    plt.figure()
    size, values = zip(*val_list)
    slope, intercept, r_value, p_value, std_err = linregress(size, values)
    r_squared = r_value ** 2
    print(f"Model Size vs. {y_val}:")
    print(f"r: {round(r_value, 4)}")
    print(f"r\u00b2: {round(r_squared, 4)}")
    print(f"p-value: {round(p_value, 4)}\n")
    size = np.array(size).reshape(-1, 1)
    values = np.array(values)
    model = LinearRegression()
    model.fit(size, values)
    y_pred = model.predict(size)
    plt.scatter(size, values)
    plt.plot(size, y_pred)
    plt.xlabel("Model Size (B)")
    plt.ylabel(y_val)
    plt.title(f"Qwen Model Size vs. {y_val}")

##########################################################################################

iqr_list = []
zscore_list = []

# read survey results
survey_df = pd.read_csv("Results/Data/survey_responses_mapped.csv")
survey_df["material"] = survey_df["material"].replace("aluminium", "aluminum")
survey_df = survey_df.dropna(how="any")
survey_stats = survey_df.groupby(["design", "criteria"])['response'].agg(["mean", "std", lambda x: iqr(x)]).rename(columns={"<lambda_0>": "iqr"}).reset_index()

# read agent results, and calculate iqr and z-score
for size in [1.5, 3, 7]:
    df = pd.read_csv(f"Results/Data/qwen_{str(size)}b.csv")
    stats_df = df.groupby(["design", "criteria"])['response'].agg(["mean", "std", lambda x: iqr(x)]).rename(columns={"<lambda_0>": "iqr"}).reset_index()
    for iqr_value in list(stats_df["iqr"]):
        iqr_list.append((size, iqr_value))
    merged_df = pd.merge(df, survey_stats, on=["design", "criteria"], how='left')
    merged_df["z-score"] = (merged_df["response"] - merged_df["mean"]) / merged_df["std"]
    merged_stats = merged_df.groupby(["design", "criteria"])["z-score"].agg(["mean"]).reset_index()
    for zscore in list(merged_stats["mean"]):
        zscore_list.append((size, zscore))

##########################################################################################

regression_by_size(iqr_list, "IQR")
regression_by_size(zscore_list, "Z-Score")
plt.show()