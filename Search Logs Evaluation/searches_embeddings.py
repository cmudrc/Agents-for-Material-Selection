import os
from huggingface_hub import login
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

##########################################################################################

os.environ["TOKENIZERS_PARALLELISM"] = "false"

login(token=os.getenv('HUGGINGFACE_API_TOKEN'))
model = SentenceTransformer('all-MiniLM-L12-v2')

embeddings = defaultdict(list)
modelsizes = [1.5, 3, 7, 32, 72]

# Create embeddings for queries in system prompt examples
prompt_queries = ['properties of wood', 'necessary properties of cutting boards', 'wood use in cutting boards', 'properties of thermoplastic'
, 'necessary properties of cooking pans', 'thermoplastic use in cooking pans']
for query in prompt_queries:
    embeddings['Prompt'].append(model.encode(query))
prompt_embeddings = np.array(embeddings['Prompt'])

# Create embeddings for search queries used by agent
searches_df = pd.read_csv('Search Logs Data/searches_by_size.csv')
for _, row in searches_df.iterrows():
    modelsize = row['Size']
    line = row['Query']
    if pd.notna(line): embeddings[modelsize].append(model.encode(line))

# Combine all embeddings
all_embeddings = prompt_embeddings.copy()  
labels = ['Prompt'] * len(prompt_embeddings) 
for modelsize in modelsizes:
    agent_embeddings = np.array(embeddings[modelsize])
    all_embeddings = np.vstack((all_embeddings, agent_embeddings))
    labels.extend([f'{modelsize}B'] * len(agent_embeddings))

# Standardize all embeddings
scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)

# Apply PCA and t-SNE to all embeddings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_pca = pca.fit_transform(all_embeddings_scaled)
reduced_tsne = tsne.fit_transform(all_embeddings_scaled)

palette = sns.color_palette("hls", len(modelsizes))
color_map = {'1.5B':palette[0], '3B':palette[1], '7B':palette[2], '32B':palette[3], '72B':palette[4]}

# Plot word embeddings from agent queries
for label in ['1.5B', '3B', '7B', '32B', '72B']:
    indices = [i for i, l in enumerate(labels) if l == label]
    ax1.scatter(reduced_pca[indices, 0], reduced_pca[indices, 1], label=label, color=color_map[label], alpha=0.6)
for label in ['1.5B', '3B', '7B', '32B', '72B']:
    indices = [i for i, l in enumerate(labels) if l == label]
    ax2.scatter(reduced_tsne[indices, 0], reduced_tsne[indices, 1], label=label, color=color_map[label], alpha=0.6)
    
# Plot word embeddings from system prompt
prompt_indices = [i for i, l in enumerate(labels) if l == 'Prompt']
ax1.scatter(reduced_pca[prompt_indices, 0], reduced_pca[prompt_indices, 1], label='Prompt', color='black', alpha=0.8, edgecolors='white', linewidth=0.6)
ax2.scatter(reduced_tsne[prompt_indices, 0], reduced_tsne[prompt_indices, 1], label='Prompt', color='black', alpha=0.8, edgecolors='white', linewidth=0.6)

plt.rcParams['font.family'] = 'serif'
ax1.set_xlabel('PCA Component 1', fontname='Georgia', fontsize=12)
ax1.set_ylabel('PCA Component 2', fontname='Georgia', fontsize=12)
ax1.set_title('PCA of Query Embeddings', fontname='Georgia', fontsize=16)
ax1.grid(True)
ax1.legend()
ax2.set_xlabel('t-SNE Component 1', fontname='Georgia', fontsize=12)
ax2.set_ylabel('t-SNE Component 2', fontname='Georgia', fontsize=12)
ax2.set_title('t-SNE of Query Embeddings', fontname='Georgia', fontsize=16)
ax2.grid(True)
ax2.legend()
plt.tight_layout()
plt.show()