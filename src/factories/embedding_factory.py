"""
Embedding Factory Module
Creates embedding functions based on provider configuration
"""

import os
import asyncio
from typing import Callable
from functools import partial


class EmbeddingFactory:
    """Factory class for creating embedding functions"""
    
    @staticmethod
    def create(provider: str, model_name: str, api_key: str = None,
               embedding_dim: int = None, max_token_size: int = 8192, **kwargs) -> Callable:
        """
        Create embedding function based on provider

        Args:
            provider: Embedding provider name
            model_name: Model name/identifier
            api_key: API key for cloud providers
            embedding_dim: Embedding dimension (auto-detect if None)
            max_token_size: Maximum token size
            **kwargs: Additional parameters

        Returns:
            Callable embedding function
        """
        provider = provider.lower()

        if provider == "openai":
            return EmbeddingFactory._create_openai(model_name, api_key, **kwargs)
        elif provider == "ollama":
            return EmbeddingFactory._create_ollama(model_name, embedding_dim, max_token_size, **kwargs)
        elif provider == "sentence-transformers" or provider == "huggingface":
            return EmbeddingFactory._create_sentence_transformers(model_name, **kwargs)
        elif provider == "jina":
            return EmbeddingFactory._create_jina(model_name, api_key, **kwargs)
        elif provider == "gemini":
            return EmbeddingFactory._create_gemini(model_name, api_key, embedding_dim, **kwargs)
        else:
            raise ValueError(f"Unsupported embedding provider: {provider}")
    
    @staticmethod
    def _create_openai(model_name: str, api_key: str, **kwargs) -> Callable:
        """Create OpenAI embedding function"""
        try:
            from lightrag.llm.openai import openai_embed
            
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            
            return openai_embed
            
        except ImportError as e:
            raise ImportError(f"Failed to import OpenAI embeddings: {e}")
    
    @staticmethod
    def _create_ollama(model_name: str, embedding_dim: int, max_token_size: int, **kwargs) -> Callable:
        """Create Ollama embedding function"""
        try:
            from lightrag.llm.ollama import ollama_embed
            from lightrag.utils import EmbeddingFunc

            host = kwargs.get("host", "http://localhost:11434")

            # Wrap ollama_embed with EmbeddingFunc
            # 注意：官方 ollama_embed 使用 embed_model 参数，不是 model
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim or 1024,  # bge-m3 默认是 1024 维
                max_token_size=max_token_size,
                func=partial(
                    ollama_embed.func if hasattr(ollama_embed, 'func') else ollama_embed,
                    embed_model=model_name,  # 修复：使用 embed_model 而不是 model
                    host=host
                )
            )

            return embedding_func

        except ImportError as e:
            raise ImportError(f"Failed to import Ollama embeddings: {e}")
    
    @staticmethod
    def _create_sentence_transformers(model_name: str, **kwargs) -> Callable:
        """Create Sentence-Transformers embedding function"""
        try:
            from sentence_transformers import SentenceTransformer
            from lightrag.utils import EmbeddingFunc
            import numpy as np
            
            # Load model
            model = SentenceTransformer(model_name)
            embedding_dim = model.get_sentence_embedding_dimension()
            
            async def embed_func(texts):
                """Async wrapper for sentence-transformers"""
                embeddings = model.encode(texts, convert_to_numpy=True)
                return np.array(embeddings)
            
            return EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=512,  # Default for most models
                func=embed_func
            )
            
        except ImportError as e:
            raise ImportError(
                f"Failed to import Sentence-Transformers: {e}. "
                "Install with: pip install sentence-transformers"
            )
    
    @staticmethod
    def _create_jina(model_name: str, api_key: str, **kwargs) -> Callable:
        """Create Jina AI embedding function"""
        try:
            from lightrag.llm.jina import jina_embed

            if api_key:
                os.environ["JINA_API_KEY"] = api_key

            return jina_embed

        except ImportError as e:
            raise ImportError(f"Failed to import Jina embeddings: {e}")

    @staticmethod
    def _create_gemini(model_name: str, api_key: str, embedding_dim: int = None, **kwargs) -> Callable:
        """Create Google Gemini embedding function using LightRAG's official implementation"""
        try:
            from lightrag.llm.gemini import gemini_embed
            from lightrag.utils import wrap_embedding_func_with_attrs
            import numpy as np

            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key

            # Gemini embedding models and their dimensions
            # text-embedding-004: 768 dimensions
            # gemini-embedding-001: 768 dimensions
            if embedding_dim is None:
                embedding_dim = 768  # Default for Gemini models

            # Use LightRAG's official gemini_embed implementation
            # Wrap it with proper attributes
            @wrap_embedding_func_with_attrs(
                embedding_dim=embedding_dim,
                max_token_size=kwargs.get("max_token_size", 2048),
                model_name=model_name
            )
            async def gemini_embed_func(texts: list[str]) -> np.ndarray:
                """Wrapper for LightRAG's official Gemini embedding function"""
                return await gemini_embed.func(
                    texts=texts,
                    model=model_name,
                    api_key=api_key,
                    embedding_dim=embedding_dim
                )

            return gemini_embed_func

        except ImportError as e:
            raise ImportError(
                f"Failed to import LightRAG Gemini embeddings: {e}. "
                "Make sure LightRAG is properly installed."
            )


def create_embedding(provider: str, model_name: str, api_key: str = None,
                     embedding_dim: int = None, max_token_size: int = 8192, **kwargs) -> Callable:
    """Convenience function to create embedding function"""
    return EmbeddingFactory.create(provider, model_name, api_key, embedding_dim, max_token_size, **kwargs)

