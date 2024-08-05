import pandas as pd

RANDOM_STATE = 42


class SampleDataset:

    def __init__(self, dataset_csv_file_path):
        """
        Initializes the SampleDataset object with a CSV file path.

        :param dataset_csv_file_path: Path to the CSV file containing the dataset.
        """
        self.dataset_csv_file_path = dataset_csv_file_path
        self.dataframe = self.load_data()

    def load_data(self):
        """
        Loads the dataset from the CSV file and handles potential errors.

        :return: DataFrame loaded from the CSV file.
        """
        try:
            dataframe = pd.read_csv(self.dataset_csv_file_path)
            print(
                f"Data loaded successfully from {self.dataset_csv_file_path}")
            return dataframe
        except FileNotFoundError:
            raise FileNotFoundError(
                f"File not found: {self.dataset_csv_file_path}")
        except pd.errors.EmptyDataError:
            raise ValueError("No data found in the file.")
        except pd.errors.ParserError:
            raise ValueError("Error parsing the file.")

    def sample_dataset(self, amount_of_samples):
        """
        Samples a specified number of rows from the dataframe and saves the sample as a JSON file.

        :param amount_of_samples: Number of samples to extract from the dataframe.
        """
        if amount_of_samples <= 0:
            raise ValueError("Amount of samples must be a positive integer.")

        if amount_of_samples > len(self.dataframe):
            raise ValueError(
                "Amount of samples exceeds the number of available rows.")

        sampled_dataframe = self.dataframe.sample(
            amount_of_samples, random_state=RANDOM_STATE)

        sampled_dataframe["pix_transactions"] = sampled_dataframe.apply(
            self.create_description, axis=1)

        description_dataframe = sampled_dataframe[["pix_transactions"]]

        sampled_description_csv_file = "/home/william/Desktop/NLP_Exercise/dataset/pix_transactions_sample.json"

        description_dataframe.to_json(
            sampled_description_csv_file, index=False)

        print(description_dataframe.head())
        print("Sample saved to: /dataset/pix_transactions_sample.json")

    def create_description(self, row):
        """
        Creates a description string from a row of the dataframe.

        :param row: Row of the dataframe.
        :return: Description string for the transaction.
        """
        return (f"AnoMes: {row['AnoMes']}, PAG_PFPJ: {row['PAG_PFPJ']}, REC_PFPJ: {row['REC_PFPJ']}, "
                f"PAG_REGIAO: {row['PAG_REGIAO']}, REC_REGIAO: {row['REC_REGIAO']}, PAG_IDADE: {row['PAG_IDADE']}, "
                f"REC_IDADE: {row['REC_IDADE']}, FORMAINICIACAO: {row['FORMAINICIACAO']}, NATUREZA: {row['NATUREZA']}, "
                f"FINALIDADE: {row['FINALIDADE']}, VALOR: {row['VALOR']}, QUANTIDADE: {row['QUANTIDADE']}")
