"""
Класс для оценки семантического поиска.
"""

import pandas as pd
import os
import json

import chromadb
from langchain_chroma import Chroma

from chromadb.utils import embedding_functions
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

from sentence_transformers import SentenceTransformer

from tqdm import tqdm


class SemanticSearchEvaluation:

    def __init__(
        self,
        emb_path: str
    ):
        # Загружаем эмбеддинги
        self.sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=emb_path)
        self.sentence_transformer_ef_cl = HuggingFaceEmbeddings(model_name=emb_path)
        self.result = dict()

    def evaluate(self,
                 db_path: str,
                 qa_table_name: str,
                 column_name: str,
                 displacement: int=0):
        # Загружаем базу Chroma
        persistent_client_serv = chromadb.PersistentClient(path=os.path.join(os.getcwd(), db_path))
        collection_serv = persistent_client_serv.get_or_create_collection("serv_collection", embedding_function=self.sentence_transformer_ef)

        # Загружаем датасет
        df = pd.read_csv(os.path.join(os.getcwd(), qa_table_name), sep=";")

        # Создаём клиента векторной базы
        vector_store_from_client = Chroma(
            client=persistent_client_serv,
            collection_name="serv_collection",
            embedding_function=self.sentence_transformer_ef_cl
        )

        total = 0    # Вопросов всего
        correct = 0  # Правильных индексов (первый верный)
        both = 0   # Оба индекса верные
        present = 0  # Индекс есть в списке из 5

        for i in tqdm(range(displacement, df.shape[0])):
            if df.iloc[i][column_name] == "-":
                continue
                
            try:
                qa = json.loads(df.iloc[i][column_name])
                for pr in qa:
                    curr_indexes = []
                    vect_res = vector_store_from_client.similarity_search(
                        pr['Question'],
                        k=5
                    )
                    len_vec = len(vect_res)
                    for j in range(len_vec):
                        curr_indexes.append(int(vect_res[j].metadata['Index']))
                    total += 1
                    if len(curr_indexes) and i == curr_indexes[0]:
                        correct += 1
                    if len(curr_indexes) and (i == curr_indexes[0] or i == curr_indexes[1]):
                        both += 1
                    if i in curr_indexes:
                        present += 1

            except json.JSONDecodeError:
                # В случае, если часть не может быть декодирована
                print(f"Ошибка декодирования JSON индекс: {i} текст: {df.iloc[i][column_name][:50]}")

        self.result["total"] = total
        self.result["correct"] = correct
        self.result["both"] = both
        self.result["present"] = present
        self.result["top1"] = self.result["correct"]/self.result["total"]*100
        self.result["top2"] = self.result["both"]/self.result["total"]*100
        self.result["top5"] = self.result["present"]/self.result["total"]*100

        return self.result

    def show_result(self):
        print(f'Вопросов всего: {self.result["total"]}')
        print(f'Правильных индексов (первый верный): {self.result["correct"]} доля: {self.result["top1"]:.2f}%')
        print(f'Первый или второй верные: {self.result["both"]} доля: {self.result["top2"]:.2f}%')
        print(f'Индекс есть в списке из 5: {self.result["present"]} доля: {self.result["top5"]:.2f}%')
        