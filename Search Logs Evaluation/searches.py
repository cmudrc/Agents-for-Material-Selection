import pandas as pd
import re
import numpy as np
from collections import defaultdict

##########################################################################################

rerun_pattern = re.compile(r"- INFO - Retrying run for (.*?): (\d)/5")  # Matches retries
result_pattern = re.compile(r"- INFO - Keeping result for (.*)")  # Matches successful results
search_pattern = re.compile(r"- INFO - Searches: (.*)")  # Matches search queries

results = []
total_reruns = defaultdict(list)
successful_reruns = defaultdict(list)
total_queries = defaultdict(list)
unique_queries = defaultdict(list)
unique_proportion = defaultdict(list)
log_directory = 'Search Logs'

for modelsize in [1.5, 3, 7, 14, 32, 72]:
    with open(f'Search Logs/{modelsize}B_logs.txt', 'r') as file:
        current_combination = None
        rerun_count = 0
        for line in file:
            # Check for reruns
            rerun_match = rerun_pattern.search(line)
            if rerun_match:
                rerun_count = int(rerun_match.groups()[1])
                if rerun_count == 5:
                    total_reruns[modelsize].append(rerun_count)
                    rerun_count = 0
                continue
                 
            # Check for searches
            search_match = search_pattern.search(line)
            if search_match:
                queries = search_match.group(1).split(',')
                total, unique = len(queries), len(set(queries))
                total_queries[modelsize].append(total)
                unique_queries[modelsize].append(unique)
                unique_proportion[modelsize].append(unique/total)
                continue

            # Check for successful results, moved on to next design, criterion, material combination
            result_match = result_pattern.search(line)
            if result_match:
                combination = result_match.group(1)
                successful_reruns[modelsize].append(rerun_count)
                total_reruns[modelsize].append(rerun_count)
                rerun_count = 0

##########################################################################################

def calculate_stats(data_dict):
    return {modelsize: (np.mean(data), np.std(data)) for modelsize, data in data_dict.items()}

total_reruns_stats = calculate_stats(total_reruns)
successful_reruns_stats = calculate_stats(successful_reruns)
total_queries_stats = calculate_stats(total_queries)
unique_queries_stats = calculate_stats(unique_queries)
unique_proportion_stats = calculate_stats(unique_proportion)

data = [
    {
        'modelsize': modelsize,
        'total_reruns_avg': total_reruns_stats[modelsize][0],
        'total_reruns_std': total_reruns_stats[modelsize][1],
        'success_reruns_avg': successful_reruns_stats[modelsize][0],
        'success_reruns_std': successful_reruns_stats[modelsize][1],
        'total_queries_avg': total_queries_stats[modelsize][0],
        'total_queries_std': total_queries_stats[modelsize][1],
        'unique_queries_avg': unique_queries_stats[modelsize][0],
        'unique_queries_std': unique_queries_stats[modelsize][1],
        'unique_prop_avg': unique_proportion_stats[modelsize][0],
        'unique_prop_std': unique_proportion_stats[modelsize][1]
    }
    for modelsize in total_reruns_stats
]

df = pd.DataFrame(data).round(4)
# df.to_csv('searches_summary.csv', index=False)
pd.options.display.max_columns = None
print(df)