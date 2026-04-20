"""
Configuration Loader Module
Handles loading and validation of YAML configuration files with environment variable support
"""

import os
import yaml
import re
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field


@dataclass
class LLMConfig:
    """LLM Provider Configuration"""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 2048
    timeout: int = 120
    model_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EmbeddingConfig:
    """Embedding Model Configuration"""
    provider: str
    model_name: str
    api_key: Optional[str] = None
    batch_size: int = 100
    max_token_size: int = 8192
    embedding_dim: Optional[int] = None
    timeout: int = 60


@dataclass
class VectorStoreConfig:
    """Vector Store Configuration"""
    backend: str
    url: Optional[str] = None
    collection_name: str = "lightrag_vectors"
    distance_metric: str = "cosine"
    cache_enabled: bool = True


@dataclass
class GraphStoreConfig:
    """Graph Store Configuration"""
    backend: str
    uri: Optional[str] = None
    username: Optional[str] = None
    password: Optional[str] = None
    workspace: str = "default"
    max_connection_pool_size: int = 50
    connection_timeout: float = 30.0
    connection_acquisition_timeout: float = 30.0


@dataclass
class RAGConfig:
    """RAG Engine Configuration"""
    chunk_size: int = 1200
    chunk_overlap: int = 100
    top_k: int = 60
    chunk_top_k: int = 30
    query_mode: str = "hybrid"
    max_gleaning: int = 1
    entity_types: list = field(default_factory=lambda: ["Person", "Organization", "Location"])
    language: str = "English"
    cosine_threshold: float = 0.3  # Cosine similarity threshold for vector search


@dataclass
class Config:
    """Main Configuration Container"""
    llm: LLMConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    graph_store: GraphStoreConfig
    rag: RAGConfig
    document_processing: Dict[str, Any] = field(default_factory=dict)
    performance: Dict[str, Any] = field(default_factory=dict)
    persistence: Dict[str, Any] = field(default_factory=dict)
    logging: Dict[str, Any] = field(default_factory=dict)
    webui: Dict[str, Any] = field(default_factory=dict)
    prompts: Dict[str, str] = field(default_factory=dict)


class ConfigLoader:
    """Load and validate configuration from YAML files"""
    
    def __init__(self, config_path: str = "config/config.yaml", prompts_path: str = "config/prompts.yaml"):
        self.config_path = Path(config_path)
        self.prompts_path = Path(prompts_path)
        self._raw_config: Dict[str, Any] = {}
        self._raw_prompts: Dict[str, Any] = {}
    
    def load(self) -> Config:
        """Load and parse configuration files"""
        # Load main config
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self._raw_config = yaml.safe_load(f)
        
        # Load prompts
        if self.prompts_path.exists():
            with open(self.prompts_path, 'r', encoding='utf-8') as f:
                self._raw_prompts = yaml.safe_load(f)
        
        # Resolve environment variables
        self._raw_config = self._resolve_env_vars(self._raw_config)

        # Build Config object
        return self._build_config()
    
    def _resolve_env_vars(self, data: Any) -> Any:
        """Recursively resolve ${ENV_VAR} placeholders"""
        if isinstance(data, dict):
            return {k: self._resolve_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._resolve_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Match ${VAR_NAME} pattern
            pattern = r'\$\{([^}]+)\}'
            matches = re.findall(pattern, data)
            for var_name in matches:
                env_value = os.getenv(var_name, "")
                data = data.replace(f"${{{var_name}}}", env_value)
            return data
        else:
            return data
    
    def _build_config(self) -> Config:
        """Build Config dataclass from raw dict"""
        llm_data = self._raw_config.get("llm", {})
        embedding_data = self._raw_config.get("embedding", {})
        vector_data = self._raw_config.get("vector_store", {})
        graph_data = self._raw_config.get("graph_store", {})
        rag_data = self._raw_config.get("rag", {})
        
        return Config(
            llm=LLMConfig(**llm_data),
            embedding=EmbeddingConfig(**embedding_data),
            vector_store=VectorStoreConfig(**vector_data),
            graph_store=GraphStoreConfig(**graph_data),
            rag=RAGConfig(**rag_data),
            document_processing=self._raw_config.get("document_processing", {}),
            performance=self._raw_config.get("performance", {}),
            persistence=self._raw_config.get("persistence", {}),
            logging=self._raw_config.get("logging", {}),
            webui=self._raw_config.get("webui", {}),
            prompts=self._raw_prompts
        )
    
    def validate(self, config: Config) -> bool:
        """Validate configuration values"""
        errors = []
        
        # Validate LLM provider
        valid_llm_providers = ["openai", "ollama", "anthropic", "gemini", "bedrock", "zhipu"]
        if config.llm.provider not in valid_llm_providers:
            errors.append(f"Invalid LLM provider: {config.llm.provider}. Must be one of {valid_llm_providers}")
        
        # Validate query mode
        valid_modes = ["naive", "local", "global", "hybrid"]
        if config.rag.query_mode not in valid_modes:
            errors.append(f"Invalid query mode: {config.rag.query_mode}. Must be one of {valid_modes}")
        
        # Check required API keys for cloud providers
        if config.llm.provider == "openai" and not config.llm.api_key:
            errors.append("OPENAI_API_KEY is required for OpenAI provider")
        
        if errors:
            raise ValueError("Configuration validation failed:\n" + "\n".join(errors))
        
        return True


def load_config(config_path: str = "config/config.yaml", prompts_path: str = "config/prompts.yaml") -> Config:
    """Convenience function to load and validate config"""
    loader = ConfigLoader(config_path, prompts_path)
    config = loader.load()
    loader.validate(config)
    return config

