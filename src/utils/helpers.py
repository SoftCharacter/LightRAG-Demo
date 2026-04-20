"""
Utility functions for the LightRAG system
"""

import logging
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", 
                  log_file: Optional[str] = None,
                  console: bool = True) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (optional)
        console: Whether to output to console
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger("LightRAG")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # Clear existing handlers
    logger.handlers = []
    
    # Formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_path, encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def validate_file_path(file_path: str, must_exist: bool = False) -> Path:
    """
    Validate and return Path object
    
    Args:
        file_path: File path string
        must_exist: Whether file must exist
    
    Returns:
        Path object
    
    Raises:
        FileNotFoundError: If must_exist=True and file doesn't exist
        ValueError: If path is invalid
    """
    try:
        path = Path(file_path).resolve()
        
        if must_exist and not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        return path
    
    except Exception as e:
        raise ValueError(f"Invalid file path: {file_path}. Error: {e}")


def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human-readable format
    
    Args:
        size_bytes: Size in bytes
    
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename by removing invalid characters
    
    Args:
        filename: Original filename
    
    Returns:
        Sanitized filename
    """
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Remove leading/trailing spaces and dots
    filename = filename.strip('. ')
    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')
    
    return filename or 'untitled'


def count_tokens_estimate(text: str) -> int:
    """
    Estimate token count (rough approximation)
    
    Args:
        text: Input text
    
    Returns:
        Estimated token count
    """
    # Rough estimate: ~4 characters per token for English
    return len(text) // 4


def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """
    Truncate text to maximum length
    
    Args:
        text: Input text
        max_length: Maximum length
        suffix: Suffix to add when truncated
    
    Returns:
        Truncated text
    """
    if len(text) <= max_length:
        return text
    return text[:max_length - len(suffix)] + suffix


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Directory path
    
    Returns:
        Path object
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_file_extension(filename: str) -> str:
    """
    Get file extension in lowercase
    
    Args:
        filename: Filename or path
    
    Returns:
        Extension with dot (e.g., '.txt')
    """
    return Path(filename).suffix.lower()


def is_supported_format(filename: str, supported_formats: list) -> bool:
    """
    Check if file format is supported
    
    Args:
        filename: Filename or path
        supported_formats: List of supported extensions (e.g., ['.txt', '.pdf'])
    
    Returns:
        True if supported
    """
    ext = get_file_extension(filename)
    return ext in [f.lower() for f in supported_formats]

