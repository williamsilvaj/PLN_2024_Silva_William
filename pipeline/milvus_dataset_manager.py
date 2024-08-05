from pymilvus import MilvusClient, connections, Collection, FieldSchema, CollectionSchema, DataType
import json


class MilvusDatasetLoader:
    def __init__(self, text_file_path, vector_file_path, query_file_path, milvus_db_path):
        """
        Initializes the MilvusDatasetLoader with paths to the necessary files and Milvus database.

        :param text_file_path: Path to the JSON file containing text transactions.
        :param vector_file_path: Path to the JSON file containing vector embeddings.
        :param query_file_path: Path to the JSON file containing query information.
        :param milvus_db_path: Path to the Milvus database file.
        """
        self.text_file_path = text_file_path
        self.vector_file_path = vector_file_path
        self.query_file_path = query_file_path
        self.milvus_db_path = milvus_db_path
        self.client = MilvusClient(self.milvus_db_path)

        self.text_transactions = None
        self.vector_transactions = None
        self.transaction_query = None
        self.dataset = None
        self.query_embeddings = None

    def get_information_of_dataset(self):
        print(self.client.get_collection_stats("transactions_collection"))

    def load_files(self):
        """
        Loads JSON files containing text transactions, vector embeddings, and query information.
        """
        with open(self.text_file_path, "r") as file:
            self.text_transactions = json.load(file)

        with open(self.vector_file_path, "r") as file:
            self.vector_transactions = json.load(file)

        with open(self.query_file_path, "r") as file:
            self.transaction_query = json.load(file)

    def prepare_dataset(self):
        """
        Prepares the dataset for insertion into Milvus.
        """
        text_keys = list(self.text_transactions["pix_transactions"].keys())

        # Ensure that the keys match the length of the embeddings
        if len(text_keys) != len(self.vector_transactions["pix_transactions_embeddings"]):
            raise ValueError(
                "Mismatch between number of text keys and number of vector embeddings.")

        # Create the dataset list
        self.dataset = [
            {
                "id": int(key),
                "vector": self.vector_transactions["pix_transactions_embeddings"][text_keys.index(key)],
                "text": self.text_transactions["pix_transactions"][key]
            }
            for key in text_keys
        ]

    def save_dataset_to_db(self):

        if self.client.has_collection(collection_name="transactions_collection"):
            self.client.drop_collection(
                collection_name="transactions_collection")
        self.client.create_collection(
            collection_name="transactions_collection",
            dimension=1024
        )

        result = self.client.insert(
            collection_name="transactions_collection", data=self.dataset)

        print(result)

    def load_collection(self):
        self.client.create_collection(
            collection_name="transactions_collection",
            dimension=1024
        )
        result = self.client.insert(
            collection_name="transactions_collection", data=self.dataset)

        print(result)

    def get_query_embeddings(self):
        """
        Retrieves the query embeddings from the loaded query file.
        """
        self.query_embeddings = self.transaction_query["query"]

    def setup_pre_populated_db(self):
        """
        Loads the files and prepares the dataset already populated
        """
        self.load_files()
        # self.load_collection()
        self.get_query_embeddings()
        print("Dataset and query prepared successfully.")

    def setup(self):
        """
        Loads the files and prepares the dataset.
        """
        self.load_files()
        self.prepare_dataset()
        self.save_dataset_to_db()
        self.get_query_embeddings()
        print("Dataset and query prepared successfully.")

    def search(self, collection_name, limit):
        """
        Perform a search using the query embeddings in the specified Milvus collection.

        :param collection_name: Name of the Milvus collection to search in.
        :param limit: Number of returned entities.
        :return: Search results.
        """
        if self.query_embeddings is None:
            raise ValueError(
                "Query embeddings not loaded. Call setup() before searching.")

        # Perform the search using Cosine similarity
        res = self.client.search(
            collection_name=collection_name,  # target collection
            data=self.query_embeddings,  # query vectors
            limit=limit,  # number of returned entities
            # specifies fields to be returned
            output_fields=["text", "vector"],
        )

        # Extract and print the results
        results = []
        for result in res[0]:
            entity = result['entity']
            results.append({
                "id": result['id'],
                "distance": result['distance'],
                "text": entity['text']
            })

        return results
