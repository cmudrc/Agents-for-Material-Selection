import re
import pandas as pd
import numpy as np
from collections import defaultdict

##########################################################################################

# Create regex patterns to match retries, successful results, and search queries
rerun_pattern = re.compile(r"- INFO - Retrying run for (.*?): (\d)/5")
result_pattern = re.compile(r"- INFO - Keeping result for (.*)")
search_pattern = re.compile(r"- INFO - Searches: (.*)")

results = []
total_reruns = defaultdict(list)
successful_reruns = defaultdict(list)
total_queries = defaultdict(list)
size = []
query = []
unique_queries = defaultdict(list)
unique_proportion = defaultdict(list)
log_directory = 'Search Logs'

# Iterate through logs for each model size and extract data
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
                queries = [query.lower().strip() for query in search_match.group(1).split(',')]
                query.extend(queries)
                size.extend([modelsize] * len(queries))
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

# Calculate mean and standard deviation by model size
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

# Create dataframe summarizing total number of reruns, total number of reruns for
# successful prompts, total queries, unique queries, and proportion of unique queries
summary_df = pd.DataFrame(data).round(4)
summary_df.to_csv('Search Logs Data/searches_summary.csv', index=False)

# Create dataframe of all search queries
query_df = pd.DataFrame({'Size': size, 'Query': query})
query_df.to_csv('Search Logs Data/searches_by_size.csv', index=False)

# Change queries to filler words for better analysis of word embeddings
replacements = {
    r'\b(kitchen utensil grip|kitchen utensil grips|kitchen utensils grips|utensil grip|utensil grips|grip|grips|kitchen utensil|kitchen utensils|safety helmet|safety helmets|safety equipment|underwater component|underwater components|underwater environments|underwater|marine environment|spacecraft component|spacecraft components|spacecraft|space environments|space applications)\b': '{design}',
    r'\b(stainless steel|steel|aluminum alloys|aluminum|titanium|glass|wood|thermoplastic|thermoset materials|thermoset|thermosets|elastomer materials|elastomer|elastomers|composite material|composite materials|composite|composites)\b': '{material}',
    r'\b(heat resistance scale|heat resistance|heat resistant|melting point|heat conductivity|heat-resistant|heat performance|thermal|corrosion resistance|corrosion resistant|corrosion-resistant|corrosion|corrosive|high strength|high-strength|strength|lightweight|weight|density)\b': '{criterion}',
}

def replace_phrases(query, replacements):
    if isinstance(query, str):
        for pattern, replacement in replacements.items():
            query = re.sub(pattern, replacement, query)
    return query

query_df['Query'] = query_df['Query'].apply(lambda x: replace_phrases(x, replacements))
query_df.to_csv('Search Logs Data/filtered_searches_by_size.csv', index=False)