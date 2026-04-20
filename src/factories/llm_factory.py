"""
LLM Factory Module
Creates LLM model functions based on provider configuration
"""

import os
import asyncio
from typing import Callable, Any
from functools import partial


class LLMFactory:
    """Factory class for creating LLM model functions"""
    
    @staticmethod
    def create(provider: str, model_name: str, api_key: str = None, 
               base_url: str = None, **kwargs) -> Callable:
        """
        Create LLM model function based on provider
        
        Args:
            provider: LLM provider name (openai, ollama, anthropic, etc.)
            model_name: Model name/identifier
            api_key: API key for cloud providers
            base_url: Optional base URL for API
            **kwargs: Additional provider-specific parameters
        
        Returns:
            Callable LLM model function
        """
        provider = provider.lower()
        
        if provider == "openai":
            return LLMFactory._create_openai(model_name, api_key, base_url, **kwargs)
        elif provider == "ollama":
            return LLMFactory._create_ollama(model_name, base_url or "http://localhost:11434", **kwargs)
        elif provider == "anthropic":
            return LLMFactory._create_anthropic(model_name, api_key, **kwargs)
        elif provider == "gemini":
            return LLMFactory._create_gemini(model_name, api_key, **kwargs)
        elif provider == "bedrock":
            return LLMFactory._create_bedrock(model_name, **kwargs)
        elif provider == "zhipu":
            return LLMFactory._create_zhipu(model_name, api_key, **kwargs)
        else:
            raise ValueError(f"Unsupported LLM provider: {provider}")
    
    @staticmethod
    def _create_openai(model_name: str, api_key: str, base_url: str = None, **kwargs) -> Callable:
        """Create OpenAI LLM function"""
        try:
            from lightrag.llm.openai import gpt_4o_mini_complete
            
            # Set environment variables
            if api_key:
                os.environ["OPENAI_API_KEY"] = api_key
            if base_url:
                os.environ["OPENAI_BASE_URL"] = base_url
            
            # Return the LLM function (can be customized with partial if needed)
            return gpt_4o_mini_complete
            
        except ImportError as e:
            raise ImportError(f"Failed to import OpenAI LLM: {e}. Install with: pip install openai")
    
    @staticmethod
    def _create_ollama(model_name: str, base_url: str, **kwargs) -> Callable:
        """Create Ollama LLM function"""
        try:
            from lightrag.llm.ollama import ollama_model_complete
            
            # Ollama doesn't need API key, but needs host URL
            return ollama_model_complete
            
        except ImportError as e:
            raise ImportError(f"Failed to import Ollama LLM: {e}")
    
    @staticmethod
    def _create_anthropic(model_name: str, api_key: str, **kwargs) -> Callable:
        """Create Anthropic Claude LLM function"""
        try:
            from lightrag.llm.anthropic import anthropic_model_complete
            
            if api_key:
                os.environ["ANTHROPIC_API_KEY"] = api_key
            
            return anthropic_model_complete
            
        except ImportError as e:
            raise ImportError(f"Failed to import Anthropic LLM: {e}. Install with: pip install anthropic")
    
    @staticmethod
    def _create_gemini(model_name: str, api_key: str, **kwargs) -> Callable:
        """Create Google Gemini LLM function"""
        try:
            from google import genai

            if api_key:
                os.environ["GEMINI_API_KEY"] = api_key

            # Create a wrapper function that matches LightRAG's expected interface
            async def gemini_model_complete(
                prompt: str,
                system_prompt: str = None,
                **llm_kwargs
            ) -> str:
                """
                Gemini model completion function

                Args:
                    prompt: User prompt
                    system_prompt: System instruction (optional)
                    **llm_kwargs: Additional arguments

                Returns:
                    Generated text
                """
                # Initialize client (will use GEMINI_API_KEY from environment)
                client = genai.Client(api_key=api_key if api_key else None)

                # Prepare contents
                if system_prompt:
                    # Gemini doesn't have explicit system role, prepend to prompt
                    full_prompt = f"{system_prompt}\n\n{prompt}"
                else:
                    full_prompt = prompt

                # Extract parameters
                temperature = llm_kwargs.get("temperature", kwargs.get("temperature", 0.7))
                max_tokens = llm_kwargs.get("max_tokens", kwargs.get("max_tokens", 2048))

                # Generate content
                response = await asyncio.to_thread(
                    client.models.generate_content,
                    model=model_name,
                    contents=full_prompt,
                    config={
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                    }
                )

                # Extract text from response
                return response.text

            return gemini_model_complete

        except ImportError as e:
            raise ImportError(
                f"Failed to import Google Gemini: {e}. "
                "Install with: pip install google-genai"
            )
    
    @staticmethod
    def _create_bedrock(model_name: str, **kwargs) -> Callable:
        """Create AWS Bedrock LLM function"""
        try:
            from lightrag.llm.bedrock import bedrock_complete
            
            # Bedrock uses AWS credentials from environment or ~/.aws/credentials
            return bedrock_complete
            
        except ImportError as e:
            raise ImportError(f"Failed to import Bedrock LLM: {e}. Install with: pip install boto3")
    
    @staticmethod
    def _create_zhipu(model_name: str, api_key: str, **kwargs) -> Callable:
        """Create Zhipu AI LLM function"""
        try:
            from lightrag.llm.zhipu import zhipuai_complete
            
            if api_key:
                os.environ["ZHIPUAI_API_KEY"] = api_key
            
            return zhipuai_complete
            
        except ImportError as e:
            raise ImportError(f"Failed to import Zhipu AI LLM: {e}")


def create_llm(provider: str, model_name: str, api_key: str = None, 
               base_url: str = None, **kwargs) -> Callable:
    """Convenience function to create LLM model function"""
    return LLMFactory.create(provider, model_name, api_key, base_url, **kwargs)

