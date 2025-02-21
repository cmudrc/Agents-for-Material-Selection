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

##########################################################################################

login(token=os.getenv('HUGGINGFACE_API_TOKEN'))
model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = defaultdict(list)
modelsizes = [1.5, 3, 7, 32, 72]
palette = sns.color_palette("hls", len(modelsizes))

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
all_embeddings = prompt_embeddings.copy()  # Start with prompt embeddings
for modelsize in modelsizes:
    agent_embeddings = np.array(embeddings[modelsize])
    all_embeddings = np.vstack((all_embeddings, agent_embeddings))

# Standardize all embeddings
scaler = StandardScaler()
all_embeddings_scaled = scaler.fit_transform(all_embeddings)

# Split back the scaled embeddings into prompt and agent embeddings
prompt_embeddings_scaled = all_embeddings_scaled[:len(prompt_embeddings)]
agent_embeddings_scaled = all_embeddings_scaled[len(prompt_embeddings):]

# Apply PCA and t-SNE to all embeddings
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
pca = PCA(n_components=2)
tsne = TSNE(n_components=2, random_state=42, perplexity=5)

# Process and plot for agent queries
for modelsize in modelsizes:
    agent_embeddings = np.array(embeddings[modelsize])
    color_index = modelsizes.index(modelsize)

    # Apply PCA to agent queries
    reduced_agent_embeddings_pca = pca.fit_transform(agent_embeddings_scaled[:len(agent_embeddings)])  # Use fit_transform for the first time
    ax1.scatter(reduced_agent_embeddings_pca[:, 0], reduced_agent_embeddings_pca[:, 1], label=f'{modelsize}B', color=palette[color_index], alpha=0.6)

    # Apply t-SNE to agent queries
    reduced_agent_embeddings_tsne = tsne.fit_transform(agent_embeddings_scaled[:len(agent_embeddings)])  # Apply fit_transform for each agent
    ax2.scatter(reduced_agent_embeddings_tsne[:, 0], reduced_agent_embeddings_tsne[:, 1], label=f'{modelsize}B', color=palette[color_index], alpha=0.6)

# PCA for prompt embeddings
reduced_prompt_embeddings_pca = pca.fit_transform(prompt_embeddings_scaled)
ax1.scatter(reduced_prompt_embeddings_pca[:, 0], reduced_prompt_embeddings_pca[:, 1], label='Prompt', color='black', alpha=0.6)

# t-SNE for prompt embeddings
reduced_prompt_embeddings_tsne = tsne.fit_transform(prompt_embeddings_scaled)
ax2.scatter(reduced_prompt_embeddings_tsne[:, 0], reduced_prompt_embeddings_tsne[:, 1], label='Prompt', color='black', alpha=0.6)

ax1.set_xlabel('PCA Component 1')
ax1.set_ylabel('PCA Component 2')
ax1.set_title('PCA of Query Embeddings')
ax1.grid(True)
ax1.legend()

ax2.set_xlabel('t-SNE Component 1')
ax2.set_ylabel('t-SNE Component 2')
ax2.set_title('t-SNE of Query Embeddings')
ax2.grid(True)
ax2.legend()

plt.tight_layout()
plt.show()