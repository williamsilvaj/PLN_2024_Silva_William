from sample_dataset import SampleDataset
from embeddings_processor import EmbeddingProcessor
from milvus_dataset_manager import MilvusDatasetLoader
from data_visualizer import DataVisualizer


def main():
    model = "PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir"
    csv_file_path = "/home/william/Desktop/NLP_Exercise/dataset/estatisticas_de_transações_pix.csv"
    sample_json_file_path = "/home/william/Desktop/NLP_Exercise/dataset/pix_transactions_sample.json"
    embeddings_json_file_path = "/home/william/Desktop/NLP_Exercise/dataset/pix_transactions_embeddings.json"
    query_json_file_path = "/home/william/Desktop/NLP_Exercise/dataset/pix_transaction_query.json"
    milvus_db_path = "/home/william/Desktop/NLP_Exercise/dataset/milvus_pix_transactions.db"

    # 1. Sample Dataset
    print("Sampling dataset...")
    sampler = SampleDataset(csv_file_path)
    sampler.sample_dataset(50)  # Sample 20 entries

    # 2. Process Embeddings
    print("Processing embeddings...")
    processor = EmbeddingProcessor(model_name=model,
                                   descriptions_file_path=sample_json_file_path,
                                   embeddings_file_path=embeddings_json_file_path,
                                   query_file_path=query_json_file_path)

    descriptions = processor.load_descriptions()
    embeddings = processor.encode_descriptions(descriptions)
    processor.save_embeddings(embeddings)

    query_text = "AnoMes: 202212, PAG_PFPJ: PF, REC_PFPJ: PF, PAG_REGIAO: NORDESTE, REC_REGIAO: NORDESTE, PAG_IDADE: Nao informado, REC_IDADE: entre 20 e 29 anos, FORMAINICIACAO: DICT, NATUREZA: P2P, FINALIDADE: Pix, VALOR: 2963,41, QUANTIDADE: 21"
    processor.encode_query(query_text)

    # 3. Load Data into Milvus
    print("Loading data into Milvus...")
    loader = MilvusDatasetLoader(text_file_path=sample_json_file_path,
                                 vector_file_path=embeddings_json_file_path,
                                 query_file_path=query_json_file_path,
                                 milvus_db_path=milvus_db_path)
    loader.setup_pre_populated_db()
    loader.get_information_of_dataset()

    # Perform a search
    search_results = loader.search(
        collection_name="transactions_collection", limit=20)
    for result in search_results:
        print(result)

    # 4. Visualize Data
    print("Visualizing data...")
    visualizer = DataVisualizer(embeddings_file_path=embeddings_json_file_path,
                                query_file_path=query_json_file_path,
                                milvus_db_path=milvus_db_path)
    visualizer.load_data()
    visualizer.perform_search()
    visualizer.plot_visualizations()


if __name__ == "__main__":
    main()
