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
from dotenv import load_dotenv

##########################################################################################

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

login(token=os.getenv('HUGGINGFACE_API_TOKEN'))
model = SentenceTransformer('all-MiniLM-L12-v2')

embeddings = defaultdict(list)
modelsizes = [1.7, 4, 8, 14, 32]

# Create embeddings for queries in system prompt examples
prompt_queries = ['properties of {material}', 'necessary properties of {design}', '{material} use in {design}']
for query in prompt_queries:
    embeddings['Prompt'].append(model.encode(query))
prompt_embeddings = np.array(embeddings['Prompt'])

# Create embeddings for search queries used by agent
searches_df = pd.read_csv('Search Logs Data/filtered_searches_by_size.csv')
for _, row in searches_df.iterrows():
    modelsize = row['Size']
    line = row['Query']
    if pd.notna(line): 
        embeddings[modelsize].append(model.encode(line))

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

# Apply PCA
pca = PCA(n_components=2)
reduced_pca = pca.fit_transform(all_embeddings_scaled)

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
reduced_tsne = tsne.fit_transform(all_embeddings_scaled)

# Define color palette
palette = sns.color_palette("hls", len(modelsizes))
color_map = {f'{size}B': color for size, color in zip(modelsizes, palette)}

# Function to plot embeddings
def plot_embeddings(reduced_embeddings, title, xlabel, ylabel, filename):
    plt.figure(figsize=(8, 5.5))
    
    # Plot agent query embeddings
    for label in ['1.7B', '4B', '8B', '14B', '32B']:
        indices = [i for i, l in enumerate(labels) if l == label]
        plt.scatter(reduced_embeddings[indices, 0], reduced_embeddings[indices, 1], label=label, color=color_map[label], alpha=0.6)

    # Plot prompt embeddings
    prompt_indices = [i for i, l in enumerate(labels) if l == 'Prompt']
    plt.scatter(reduced_embeddings[prompt_indices, 0], reduced_embeddings[prompt_indices, 1], label='Prompt', color='black', alpha=0.8, edgecolors='white', linewidth=0.6)

    plt.xlabel(xlabel, fontname='Georgia', fontsize=18)
    plt.ylabel(ylabel, fontname='Georgia', fontsize=18)
    plt.title(title, fontname='Georgia', fontsize=20)
    plt.grid(True)
    legend = plt.legend()
    for text in legend.get_texts():
        text.set_fontname('Georgia')
        text.set_fontsize(16)
    plt.xticks(fontname='Georgia', fontsize=16)
    plt.yticks(fontname='Georgia', fontsize=16)
    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

plot_embeddings(reduced_pca, "PCA of Query Embeddings", "PCA Component 1", "PCA Component 2", "pca_plot.png")
plot_embeddings(reduced_tsne, "t-SNE of Query Embeddings", "t-SNE Component 1", "t-SNE Component 2", "tsne_plot.png")