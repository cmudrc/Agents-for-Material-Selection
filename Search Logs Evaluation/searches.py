import pandas as pd
import re

##########################################################################################

pattern = re.compile(r"- (Design and criterion|Search Query): (.*)")

# Count search count, unique search count, and unique search proportion for each design, criterion, and material combination
results = []
for modelsize in [1.5, 3, 7, 14, 32, 72]:
    with open(f'Search Logs/model_{modelsize}B.txt', 'r') as file:
        unique_count = 0
        search_count = 0
        queries = set()
        for line in file:
            match = pattern.search(line)
            category, detail = match.groups()
            if category == "Design and criterion":
                if search_count != 0:
                    results.append({'model_size': modelsize,
                                    'search': search_count,
                                    'unique': unique_count,
                                    'proportion': unique_count/search_count})
                unique_count = 0
                queries = set()
                search_count = 0
            else:
                if detail not in queries:
                    queries.add(detail)
                    unique_count += 1
                search_count += 1

# Get average and standard deviation of search count, unique search count, and unique search proportion for each model size
df = pd.DataFrame(results)
summary = df.groupby('model_size')[['search', 'unique', 'proportion']].agg(['mean', 'std'])
summary.columns = [f'{col[0]}_{col[1]}' for col in summary.columns]
summary.reset_index(inplace=True)
summary = summary.round(4)
summary.to_csv('Search Logs Evaluation/searches_summary.csv', index=False)
print(summary)