# Processamento de Linguagem Natural Utilizando Dados Abertos Governamentais

**William Santos Silva**  
05/08/2024

## Introdução
![Visualização das transações Pix utilizando t-SNE](https://github.com/user-attachments/assets/d775c29e-0629-4ac6-aeef-fed9a2bff85c)
*Figura 1: Visualização das transações Pix utilizando t-SNE (t-distributed Stochastic Neighbor Embedding).*

Nesta atividade, exploramos a integração de técnicas avançadas de Processamento de Linguagem Natural (PLN) com dados abertos governamentais fornecidos pelo Banco Central do Brasil. O foco principal é na criação e utilização de embeddings por meio de modelos de transformers de última geração.

Para o armazenamento e consulta dos dados vetoriais, utilizamos o Milvus, uma ferramenta robusta para o gerenciamento de grandes volumes de dados vetoriais. A visualização dos dados é realizada com o algoritmo t-SNE (t-distributed Stochastic Neighbor Embedding), que reduz a dimensionalidade dos dados e facilita a representação visual de suas relações em um espaço bidimensional.

O objetivo é utilizar essas técnicas para identificar transações Pix potencialmente fraudulentas. Partindo da hipótese de que uma transação é suspeita de fraude, realizamos uma busca semântica na base de dados para encontrar transações com características semelhantes, aumentando a probabilidade de identificar atividades fraudulentas.

## Seleção do Conjunto de Dados

O conjunto de dados selecionado foi obtido do Banco Central do Brasil e contém estatísticas detalhadas sobre transações Pix referentes ao mês Novembro do ano de 2020. Este conjunto de dados inclui os seguintes parâmetros:
![Exemplo do conjunto de dados de transações Pix](https://github.com/user-attachments/assets/6c373b89-19c8-46f4-a485-7c83a59a1bc0)
*Figura 2: Exemplo do conjunto de dados de transações Pix.*

- **AnoMes**: Período da transação.
- **PAG_PFPJ**: Tipo de pagador (Pessoa Física ou Jurídica).
- **REC_PFPJ**: Tipo de receptor (Pessoa Física ou Jurídica).
- **PAG_REGIAO**: Região do pagador.
- **REC_REGIAO**: Região do receptor.
- **PAG_IDADE**: Faixa etária do pagador.
- **REC_IDADE**: Faixa etária do receptor.
- **FORMAINICIACAO**: Formação de iniciação da transação.
- **NATUREZA**: Natureza da transação.
- **FINALIDADE**: Finalidade da transação.
- **VALOR**: Valor da transação.
- **QUANTIDADE**: Quantidade de transações.

Este conjunto foi escolhido devido à sua amplitude, com mais de 400 mil registros, e à sua relevância para análises financeiras.

Para tornar o processamento viável, foi selecionada uma amostra de 10 mil registros. A amostra foi obtida utilizando a função `sample` do pandas com a semente (seed) 42, garantindo a reprodutibilidade dos dados amostrados por meio de um processo pseudoaleatório que pode ser consistentemente replicado.

Antes do processo de embedding, os dados de cada linha foram concatenados em uma única string. Essa abordagem permite representar os dados de uma transação como um vetor n-dimensional. Por exemplo, uma transação pode ser representada da seguinte forma:

> `"AnoMes: 202212, PAG_PFPJ: PF, REC_PFPJ: PF, PAG_REGIAO: NORDESTE, REC_REGIAO: NORDESTE, PAG_IDADE: Nao informado, REC_IDADE: entre 20 e 29 anos, FORMAINICIACAO: DICT, NATUREZA: P2P, FINALIDADE: Pix, VALOR: 2963,41, QUANTIDADE: 21"`

Essa concatenação facilita a criação de embeddings que capturam a complexidade dos dados transacionais em um espaço vetorial.

**URL de Acesso:**  
[Conjunto de Dados Pix](https://olinda.bcb.gov.br/olinda/servico/Pix_DadosAbertos/versao/v1/aplicacao#!/recursos/EstatisticasTransacoesPix)

### Justificativa

O conjunto de dados das transações Pix foi selecionado devido à sua relevância no contexto econômico e financeiro do Brasil, além de atender aos critérios de volume de dados exigidos pela atividade.

## Escolha do Modelo de Embeddings

Para a criação dos embeddings, foi escolhido o modelo `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir`, disponível no Hugging Face. Este modelo é um *sentence-transformer* especializado na língua portuguesa, projetado para mapear sentenças e parágrafos em um espaço vetorial denso de 1024 dimensões. É particularmente adequado para tarefas como clustering e busca semântica.

**URL do Modelo:**  
[PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir](https://huggingface.co/PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir)

### Justificativa

O modelo `PORTULAN/serafim-335m-portuguese-pt-sentence-encoder-ir` foi selecionado por sua eficácia na geração de embeddings de alta qualidade para textos em português. Sua capacidade de mapear sentenças e parágrafos em um espaço vetorial denso de 1024 dimensões é ideal para análises semânticas precisas. Além disso, o modelo é afinado para tarefas de *Information Retrieval* (IR), tornando-o particularmente útil para aplicações de busca semântica e clustering, atendendo aos requisitos da nossa análise.

O uso deste modelo permite capturar as nuances e a semântica dos dados textuais, contribuindo significativamente para a eficácia das análises realizadas com os embeddings gerados.

## Criação dos Embeddings

O processo de criação dos embeddings envolveu a utilização do modelo selecionado para converter as descrições das transações Pix em vetores de alta dimensão. Esses vetores capturam as características semânticas das descrições, permitindo a realização de análises posteriores.

### Passos Envolvidos

1. **Carregamento das Descrições:** As descrições das transações foram carregadas a partir do arquivo JSON.
2. **Codificação das Descrições:** Utilizando o modelo de embeddings, as descrições foram convertidas em vetores. Cada descrição foi representada como uma string no formato:
   
   > `"AnoMes: 202212, PAG_PFPJ: PF, REC_PFPJ: PF, PAG_REGIAO: NORDESTE, REC_REGIAO: NORDESTE, PAG_IDADE: Nao informado, REC_IDADE: entre 20 e 29 anos, FORMAINICIACAO: DICT, NATUREZA: P2P, FINALIDADE: Pix, VALOR: 2963,41, QUANTIDADE: 21"`

   Esta string foi então utilizada pelo modelo para gerar um vetor de 1024 dimensões. Para o processamento, foi utilizado o Google Colab que disponibiliza GPU para o processamento.
   
   ![Ambiente Google Colab utilizado para processamento com GPU](https://github.com/user-attachments/assets/229a9a90-8dc4-49a5-9e85-85ffd9b204d4)
   *Figura 3: Ambiente Google Colab utilizado para processamento com GPU.*

4. **Armazenamento dos Embeddings:** Os vetores resultantes foram armazenados em um arquivo JSON para uso posterior.
![Vetor de 1024 dimensões gerado a partir da string acima](https://github.com/user-attachments/assets/887190e3-362f-4a59-a412-2898340a14dd)
*Figura 4: Vetor de 1024 dimensões gerado a partir da string acima.*

## Armazenamento dos Embeddings no Banco de Dados Vetorial Milvus

Os embeddings foram armazenados no banco de dados vetorial Milvus, que permite consultas eficientes utilizando distâncias como a euclidiana e cosseno, entre outras, para similaridade semântica.

### Passos Envolvidos

1. **Criação da Conexão com Milvus:** Estabeleceu-se uma conexão com o banco de dados Milvus.
2. **Preparação dos Dados:** Os dados foram preparados para serem inseridos no banco de dados, garantindo a correspondência entre os textos originais e seus embeddings.
   ![Schema do banco de dados contendo ID, vetor e texto original](https://github.com/user-attachments/assets/f67f8be8-2ac3-41c1-af07-da55b42e34cb)
   *Figura 5: Schema do banco de dados contendo ID, vetor e texto original.*

3. **Inserção dos Dados:** Os embeddings foram inseridos no banco de dados Milvus.

## Consultas de Similaridade Semântica

Utilizando a funcionalidade de busca por similaridade do Milvus, foram realizadas consultas para identificar transações Pix semelhantes com base nos embeddings gerados. No exemplo em questão, foi utilizada a string:

> `"AnoMes: 202212, PAG_PFPJ: PF, REC_PFPJ: PF, PAG_REGIAO: NORDESTE, REC_REGIAO: NORDESTE, PAG_IDADE: Nao informado, REC_IDADE: entre 20 e 29 anos, FORMAINICIACAO: DICT, NATUREZA: P2P, FINALIDADE: Pix, VALOR: 2963,41, QUANTIDADE: 21"`

Como hipótese de uma transação potencialmente fraudulenta, com o objetivo de encontrar outras transações semelhantes.

### Passos Envolvidos

1. **Formulação da Consulta:** Consultas foram formuladas para identificar descrições de transações semelhantes, possivelmente fraudulentas.
2. **Execução da Busca:** A busca foi executada no banco de dados Milvus utilizando a distância cosseno.
3. **Interpretação dos Resultados:** Os resultados foram interpretados para identificar padrões e insights. Por exemplo, ao buscar por 20 transações semelhantes, obteve-se o gráfico mostrado abaixo:
   ![20 transações semelhantes à string pesquisada](https://github.com/user-attachments/assets/93c2ce27-53ce-4f45-a9b5-a771d2dc8320)
   *Figura 6: 20 transações semelhantes à string pesquisada.*

## Visualização dos Dados

Para a visualização dos dados, foram utilizadas técnicas de redução de dimensionalidade e clustering, combinadas com ferramentas de visualização interativa.

### Passos Envolvidos

1. **Redução de Dimensionalidade:** Utilizou-se t-SNE para reduzir a dimensionalidade dos embeddings.
2. **Clustering:** Aplicou-se DBSCAN para identificar clusters nos dados.
3. **Visualização:** Os dados foram visualizados utilizando Plotly para visualização interativa e Matplotlib para visualizações estáticas.

## Conclusão

A atividade demonstrou como técnicas avançadas de PLN e ferramentas modernas de armazenamento e consulta de dados vetoriais podem ser aplicadas para analisar grandes volumes de dados governamentais. A integração dessas ferramentas permite realizar análises sofisticadas, oferecendo novos insights sobre os dados.
