"""
Basic tests for LightRAG Knowledge Graph QA System
Run with: pytest tests/test_basic.py
"""

import pytest
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.config_loader import ConfigLoader, Config
from src.utils.helpers import (
    format_file_size,
    sanitize_filename,
    count_tokens_estimate,
    truncate_text,
    get_file_extension,
    is_supported_format
)


class TestConfigLoader:
    """Test configuration loading"""
    
    def test_config_loader_exists(self):
        """Test that config loader can be imported"""
        assert ConfigLoader is not None
    
    def test_config_dataclass(self):
        """Test that Config dataclass is properly defined"""
        from dataclasses import fields
        field_names = [f.name for f in fields(Config)]
        assert 'llm' in field_names
        assert 'embedding' in field_names
        assert 'vector_store' in field_names
        assert 'graph_store' in field_names


        # assert hasattr(Config, 'llm')
        # assert hasattr(Config, 'embedding')
        # assert hasattr(Config, 'vector_store')
        # assert hasattr(Config, 'graph_store')


class TestHelpers:
    """Test utility helper functions"""
    
    def test_format_file_size(self):
        """Test file size formatting"""
        assert format_file_size(0) == "0.0 B"
        assert format_file_size(1024) == "1.0 KB"
        assert format_file_size(1048576) == "1.0 MB"
        assert format_file_size(1073741824) == "1.0 GB"
    
    def test_sanitize_filename(self):
        """Test filename sanitization"""
        assert sanitize_filename("test<file>.txt") == "test_file_.txt"
        assert sanitize_filename("file:name?.pdf") == "file_name_.pdf"
        assert sanitize_filename("  .test  ") == "test"
        assert sanitize_filename("") == "untitled"
    
    def test_count_tokens_estimate(self):
        """Test token counting estimation"""
        text = "This is a test sentence."
        tokens = count_tokens_estimate(text)
        assert tokens > 0
        assert tokens < len(text)  # Should be less than character count
    
    def test_truncate_text(self):
        """Test text truncation"""
        text = "This is a very long text that needs to be truncated"
        truncated = truncate_text(text, max_length=20)
        assert len(truncated) <= 20
        assert truncated.endswith("...")
    
    def test_get_file_extension(self):
        """Test file extension extraction"""
        assert get_file_extension("test.txt") == ".txt"
        assert get_file_extension("document.PDF") == ".pdf"
        assert get_file_extension("/path/to/file.md") == ".md"
        assert get_file_extension("noextension") == ""
    
    def test_is_supported_format(self):
        """Test format support checking"""
        supported = [".txt", ".md", ".pdf"]
        
        assert is_supported_format("test.txt", supported) == True
        assert is_supported_format("doc.PDF", supported) == True
        assert is_supported_format("file.docx", supported) == False


class TestFactories:
    """Test factory classes"""
    
    def test_llm_factory_import(self):
        """Test that LLM factory can be imported"""
        from src.factories.llm_factory import LLMFactory
        assert LLMFactory is not None
    
    def test_embedding_factory_import(self):
        """Test that embedding factory can be imported"""
        from src.factories.embedding_factory import EmbeddingFactory
        assert EmbeddingFactory is not None


class TestRAGEngine:
    """Test RAG engine core"""
    
    def test_rag_engine_import(self):
        """Test that RAG engine can be imported"""
        from src.core.rag_engine import RAGEngine
        assert RAGEngine is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

