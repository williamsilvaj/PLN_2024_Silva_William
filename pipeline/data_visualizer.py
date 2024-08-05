import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
import plotly.express as px
import json
from pymilvus import MilvusClient


class DataVisualizer:
    def __init__(self, embeddings_file_path, query_file_path, milvus_db_path, perplexity=30):
        """
        Initializes the DataVisualizer with paths to the necessary files and Milvus database.

        :param embeddings_file_path: Path to the JSON file containing vector embeddings.
        :param query_file_path: Path to the JSON file containing query information.
        :param milvus_db_path: Path to the Milvus database file.
        :param perplexity: t-SNE perplexity parameter. Must be less than the number of samples.
        """
        self.embeddings_file_path = embeddings_file_path
        self.query_file_path = query_file_path
        self.milvus_db_path = milvus_db_path
        self.perplexity = perplexity
        self.client = MilvusClient(self.milvus_db_path)

        self.embeddings_np = None
        self.reduced_embeddings = None
        self.df = None
        self.df_search = None
        self.search_texts = None

    def load_data(self):
        """
        Loads vector embeddings and query information from JSON files.
        """
        with open(self.embeddings_file_path, 'r') as f:
            embeddings = json.load(f)['pix_transactions_embeddings']

        self.embeddings_np = np.array(embeddings)

        # Reduce dimensionality with t-SNE
        tsne = TSNE(n_components=2, perplexity=self.perplexity,
                    random_state=42)
        self.reduced_embeddings = tsne.fit_transform(self.embeddings_np)

        # Apply DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        clusters = dbscan.fit_predict(self.reduced_embeddings)

        # Create a DataFrame for visualization
        self.df = pd.DataFrame(self.reduced_embeddings, columns=['x', 'y'])
        self.df['cluster'] = clusters

    def perform_search(self):
        """
        Performs a search using the query vectors and retrieves results from Milvus.
        """
        with open(self.query_file_path, 'r') as file:
            transaction_query = json.load(file)

        query = transaction_query["query"]

        # Perform the search using Cosine similarity
        res = self.client.search(
            collection_name="transactions_collection",
            data=query,
            limit=20,
            output_fields=["text", "vector"],
        )

        # Extract search results
        search_embeddings = []
        self.search_texts = []
        for result in res[0]:
            entity = result['entity']
            self.search_texts.append(entity['text'])
            search_embeddings.append(entity['vector'])

        # Convert search embeddings to numpy array
        search_embeddings_np = np.array(search_embeddings)

        # Reduce dimensionality of search results
        combined_embeddings = np.vstack(
            [self.embeddings_np, search_embeddings_np])
        tsne = TSNE(n_components=2, perplexity=self.perplexity,
                    random_state=42)
        reduced_combined_embeddings = tsne.fit_transform(combined_embeddings)

        # Split the combined reduced embeddings back
        reduced_original = reduced_combined_embeddings[:-len(
            search_embeddings_np)]
        reduced_search = reduced_combined_embeddings[-len(
            search_embeddings_np):]

        # Create DataFrames for search results
        self.df_search = pd.DataFrame(reduced_search, columns=['x', 'y'])
        self.df_search['text'] = self.search_texts

    def plot_visualizations(self):
        """
        Plots the visualizations using Matplotlib and Plotly.
        """
        # Add coordinates to original DataFrame for hover information
        self.df['text'] = 'Cluster ' + self.df['cluster'].astype(str)

        # Plot with Matplotlib: full dataset with highlighted search results
        plt.figure(figsize=(10, 7))
        scatter = plt.scatter(
            self.df['x'], self.df['y'], c=self.df['cluster'], cmap='viridis', s=5)
        # Highlight search results
        plt.scatter(self.df_search['x'],
                    self.df_search['y'], c='red', marker='x', s=50)
        plt.colorbar(scatter)
        plt.title(
            't-SNE visualization of Pix transaction embeddings with search results')
        plt.show()

        # Plot with Plotly: full dataset with highlighted search results
        fig = px.scatter(self.df, x='x', y='y', color='cluster', hover_data={'x': True, 'y': True, 'text': True},
                         title='t-SNE visualization of Pix transaction embeddings')
        search_scatter = px.scatter(self.df_search, x='x', y='y', color_discrete_sequence=['red'],
                                    hover_data={'x': True, 'y': True, 'text': True}, title='Search Results')
        for trace in search_scatter.data:
            fig.add_trace(trace)
        fig.show()

        # Plot with Plotly: zoomed-in view of search results
        fig_search_only = px.scatter(self.df_search, x='x', y='y', hover_data={'x': True, 'y': True, 'text': True},
                                     title='Zoomed-in view of search results')
        fig_search_only.show()

        # Print the search results with text
        for i, text in enumerate(self.search_texts):
            print(f"Result {i+1}: {text}")


# Example usage
if __name__ == "__main__":
    embeddings_file_path = '/home/william/Desktop/NLP_Exercise/dataset/pix_transactions_embeddings.json'
    query_file_path = '/home/william/Desktop/NLP_Exercise/dataset/pix_transaction_query.json'
    milvus_db_path = 'milvus_pix_transactions.db'

    visualizer = DataVisualizer(
        embeddings_file_path, query_file_path, milvus_db_path, perplexity=30)
    visualizer.load_data()
    visualizer.perform_search()
    visualizer.plot_visualizations()
