from sentence_transformers import SentenceTransformer
import json


class EmbeddingProcessor:
    def __init__(self, model_name, descriptions_file_path, embeddings_file_path, query_file_path):
        """
        Initializes the EmbeddingProcessor with the model and file paths.

        :param model_name: Name of the SentenceTransformer model.
        :param descriptions_file_path: Path to the JSON file containing descriptions.
        :param embeddings_file_path: Path where the embeddings JSON file will be saved.
        :param query_file_path: Path where the query embeddings JSON file will be saved.
        """
        self.model_name = model_name
        self.descriptions_file_path = descriptions_file_path
        self.embeddings_file_path = embeddings_file_path
        self.query_file_path = query_file_path

        # Initialize the SentenceTransformer model
        self.model = SentenceTransformer(model_name)

    def load_descriptions(self):
        """
        Loads descriptions from the JSON file and returns them as a list.
        """
        with open(self.descriptions_file_path, 'r') as file:
            pix_transactions_data = json.load(file)

        # Extract descriptions
        transactions_descriptions = pix_transactions_data["pix_transactions"]
        return list(transactions_descriptions.values())

    def encode_descriptions(self, descriptions):
        """
        Encodes the descriptions using the SentenceTransformer model.

        :param descriptions: List of descriptions to encode.
        :return: List of embeddings.
        """
        return self.model.encode(descriptions).tolist()

    def save_embeddings(self, embeddings):
        """
        Saves the embeddings to a JSON file.

        :param embeddings: List of embeddings to save.
        """
        embeddings_dictionary = {"pix_transactions_embeddings": embeddings}
        with open(self.embeddings_file_path, 'w') as file:
            json.dump(embeddings_dictionary, file)
        print("Embeddings created and saved to", self.embeddings_file_path)

    def encode_query(self, query_text):
        """
        Encodes a single query text and saves it to a JSON file.

        :param query_text: The text to encode.
        """
        query_embedding = self.model.encode(query_text).tolist()
        query_dictionary = {"query": [query_embedding]}
        with open(self.query_file_path, "w") as json_file:
            json.dump(query_dictionary, json_file, indent=4)
        print("Query embedding completed and saved to", self.query_file_path)
