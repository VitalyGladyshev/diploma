import logging
import os
import pandas as pd
import tiktoken
from dataclasses import dataclass
from dotenv import load_dotenv
from typing import Dict, Optional, Any, Union, List

from graphrag.query.context_builder.entity_extraction import EntityVectorStoreKey
from graphrag.query.indexer_adapters import (
    # read_indexer_covariates,
    read_indexer_entities,
    read_indexer_relationships,
    read_indexer_reports,
    read_indexer_text_units,
)
from graphrag.config.enums import ModelType
from graphrag.config.models.language_model_config import LanguageModelConfig
from graphrag.language_model.manager import ModelManager
from graphrag.query.structured_search.local_search.mixed_context import (
    LocalSearchMixedContext,
)
from graphrag.query.structured_search.local_search.search import LocalSearch
from graphrag.vector_stores.lancedb import LanceDBVectorStore

# Настройка логирования
logger = logging.getLogger(__name__)
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.INFO)


@dataclass
class GraphSearchConfig:
    """Конфигурация для поиска в графе знаний"""
    graphrag_dir: str = None
    input_dir: str = None
    lancedb_uri: str = None
    community_report_table: str = "community_reports"
    community_table: str = "communities"
    entity_table: str = "entities"
    relationship_table: str = "relationships"
    covariate_table: str = "covariates"
    text_unit_table: str = "text_units"
    community_level: int = 2
    llm_model: str = "gpt-4o-mini"
    embedding_model: str = "text-embedding-3-small"
    api_base: str = "https://api.vsegpt.ru/v1"
    api_key_env: str = "GRAPHRAG_API_KEY"
    max_tokens_context: int = 12_000
    max_tokens_response: int = 2_000
    temperature: float = 0.0

    def __post_init__(self):
        """Инициализирует пути, если они не указаны"""
        if not self.graphrag_dir:
            self.graphrag_dir = os.path.join(os.getcwd(), "ragtestour")
        
        if not self.input_dir:
            self.input_dir = os.path.join(self.graphrag_dir, "output")
            
        if not self.lancedb_uri:
            self.lancedb_uri = os.path.join(self.input_dir, "lancedb")


class GraphLocalSearch:
    """
    Класс для локального поиска в графе знаний.
    
    Предоставляет интерфейс для семантического поиска информации
    в графе знаний с использованием языковых моделей.
    """

    def __init__(self, config: Optional[GraphSearchConfig] = None):
        """
        Инициализирует поисковой движок графа знаний.
        
        Args:
            config: Конфигурация поиска. Если не указана, используется конфигурация по умолчанию.
        """
        load_dotenv()
        self.config = config or GraphSearchConfig()
        self.search_engine = None
        logger.info("Инициализирован GraphLocalSearch с конфигурацией")

    def initialize(self) -> None:
        """
        Инициализирует поисковой движок и все необходимые компоненты.
        
        Вызывается автоматически при первом поиске, если движок еще не инициализирован.
        Можно вызвать вручную для предварительной загрузки.
        
        Raises:
            FileNotFoundError: Если не найдены файлы данных
            ValueError: Если не установлен API ключ или другая ошибка конфигурации
            Exception: При других ошибках инициализации
        """
        if self.search_engine:
            return
            
        try:
            # Проверка наличия API ключа
            api_key = os.environ.get(self.config.api_key_env)
            if not api_key:
                raise ValueError(f"API ключ не найден в переменной окружения {self.config.api_key_env}")

            # Загрузка данных
            logger.info("Загрузка данных из файлов...")
            entities = self._load_entities()
            relationships = self._load_relationships()
            reports = self._load_reports()
            text_units = self._load_text_units()
            
            # Инициализация векторных хранилищ
            logger.info("Подключение к векторным хранилищам...")
            description_embedding_store = self._initialize_vector_store(
                "default-entity-description"
            )
            full_content_embedding_store = self._initialize_vector_store(
                "default-community-full_content"
            )
            
            # Инициализация языковой модели
            logger.info(f"Инициализация языковой модели {self.config.llm_model}...")
            chat_model = self._initialize_chat_model(api_key)
            text_embedder = self._initialize_embedding_model(api_key)
            token_encoder = tiktoken.encoding_for_model(self.config.llm_model)
            
            # Инициализация построителя контекста
            logger.info("Инициализация построителя контекста...")
            context_builder = self._initialize_context_builder(
                entities, relationships, reports, text_units,
                description_embedding_store, text_embedder, token_encoder
            )
            
            # Инициализация параметров поиска
            context_params = self._get_context_params()
            model_params = self._get_model_params()
            
            # Инициализация поискового движка
            logger.info("Инициализация поискового движка...")
            self.search_engine = LocalSearch(
                model=chat_model,
                context_builder=context_builder,
                token_encoder=token_encoder,
                model_params=model_params,
                context_builder_params=context_params,
                response_type="multiple paragraphs",
            )
            
            logger.info("Поисковый движок успешно инициализирован")
            
        except FileNotFoundError as e:
            logger.error(f"Ошибка при загрузке файлов данных: {e}")
            raise
        except ValueError as e:
            logger.error(f"Ошибка конфигурации: {e}")
            raise
        except Exception as e:
            logger.error(f"Ошибка при инициализации поискового движка: {e}")
            raise

    async def graph_local_search(self, 
                                query: str, 
                                response_type: str = "multiple paragraphs") -> str:
        """
        Выполняет локальный поиск по графу знаний.
        
        Args:
            query: Поисковый запрос на естественном языке
            response_type: Тип ожидаемого ответа (например, "multiple paragraphs", 
                          "prioritized list", "single paragraph", "report")
        
        Returns:
            Результат поиска в виде текста
            
        Raises:
            Exception: При ошибке поиска или инициализации
        """
        try:
            if not self.search_engine:
                logger.info("Поисковый движок не инициализирован. Выполняется инициализация...")
                self.initialize()
                
            logger.info(f"Выполнение поиска по запросу: {query}")
            self.search_engine.response_type = response_type
            result = await self.search_engine.search(query)
            logger.info("Поиск выполнен успешно")
            return result
            
        except Exception as e:
            logger.error(f"Ошибка при выполнении поиска: {e}")
            raise

    def _load_entities(self) -> Dict[str, Any]:
        """Загружает данные сущностей"""
        entity_path = os.path.join(self.config.input_dir, f"{self.config.entity_table}.parquet")
        community_path = os.path.join(self.config.input_dir, f"{self.config.community_table}.parquet")
        
        logger.debug(f"Загрузка сущностей из {entity_path}")
        entity_df = pd.read_parquet(entity_path)
        
        logger.debug(f"Загрузка сообществ из {community_path}")
        community_df = pd.read_parquet(community_path)
        
        return read_indexer_entities(
            entity_df, community_df, self.config.community_level
        )

    def _load_relationships(self) -> Dict[str, Any]:
        """Загружает данные отношений"""
        path = os.path.join(self.config.input_dir, f"{self.config.relationship_table}.parquet")
        logger.debug(f"Загрузка отношений из {path}")
        relationship_df = pd.read_parquet(path)
        return read_indexer_relationships(relationship_df)

    def _load_reports(self) -> Dict[str, Any]:
        """Загружает данные отчетов"""
        report_path = os.path.join(self.config.input_dir, f"{self.config.community_report_table}.parquet")
        community_path = os.path.join(self.config.input_dir, f"{self.config.community_table}.parquet")
        
        logger.debug(f"Загрузка отчетов из {report_path}")
        report_df = pd.read_parquet(report_path)
        
        logger.debug(f"Загрузка сообществ из {community_path}")
        community_df = pd.read_parquet(community_path)
        
        return read_indexer_reports(
            report_df, community_df, self.config.community_level
        )

    def _load_text_units(self) -> Dict[str, Any]:
        """Загружает текстовые единицы"""
        path = os.path.join(self.config.input_dir, f"{self.config.text_unit_table}.parquet")
        logger.debug(f"Загрузка текстовых единиц из {path}")
        text_unit_df = pd.read_parquet(path)
        return read_indexer_text_units(text_unit_df)

    def _initialize_vector_store(self, collection_name: str) -> LanceDBVectorStore:
        """Инициализирует векторное хранилище"""
        vector_store = LanceDBVectorStore(collection_name=collection_name)
        vector_store.connect(db_uri=self.config.lancedb_uri)
        return vector_store

    def _initialize_chat_model(self, api_key: str) -> Any:
        """Инициализирует модель чата"""
        chat_config = LanguageModelConfig(
            api_key=api_key,
            api_base=self.config.api_base,
            type=ModelType.OpenAIChat,
            model=self.config.llm_model,
            max_retries=20,
        )
        return ModelManager().get_or_create_chat_model(
            name="local_search",
            model_type=ModelType.OpenAIChat,
            config=chat_config,
        )

    def _initialize_embedding_model(self, api_key: str) -> Any:
        """Инициализирует модель эмбеддингов"""
        embedding_config = LanguageModelConfig(
            api_key=api_key,
            api_base=self.config.api_base,
            type=ModelType.OpenAIEmbedding,
            model=self.config.embedding_model,
            max_retries=20,
        )
        return ModelManager().get_or_create_embedding_model(
            name="local_search_embedding",
            model_type=ModelType.OpenAIEmbedding,
            config=embedding_config,
        )

    def _initialize_context_builder(
        self, entities, relationships, reports, text_units,
        description_embedding_store, text_embedder, token_encoder
    ) -> LocalSearchMixedContext:
        """Инициализирует построитель контекста"""
        return LocalSearchMixedContext(
            community_reports=reports,
            text_units=text_units,
            entities=entities,
            relationships=relationships,
            covariates=None,
            entity_text_embeddings=description_embedding_store,
            embedding_vectorstore_key=EntityVectorStoreKey.ID,
            text_embedder=text_embedder,
            token_encoder=token_encoder,
        )

    def _get_context_params(self) -> Dict[str, Any]:
        """Возвращает параметры контекста для поиска"""
        return {
            "text_unit_prop": 0.5,
            "community_prop": 0.1,
            "conversation_history_max_turns": 5,
            "conversation_history_user_turns_only": True,
            "top_k_mapped_entities": 10,
            "top_k_relationships": 10,
            "include_entity_rank": True,
            "include_relationship_weight": True,
            "include_community_rank": False,
            "return_candidate_context": False,
            "embedding_vectorstore_key": EntityVectorStoreKey.ID,
            "max_tokens": self.config.max_tokens_context,
        }

    def _get_model_params(self) -> Dict[str, Any]:
        """Возвращает параметры модели для поиска"""
        return {
            "max_tokens": self.config.max_tokens_response,
            "temperature": self.config.temperature,
        }
    