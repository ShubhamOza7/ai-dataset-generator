#!/usr/bin/env python3
"""
AWS LLM Core Services
Reusable LLM service abstraction for dynamic model selection and integration on AWS.
Provides unified interface for multiple LLM providers with AWS infrastructure support.
"""

import json
import hashlib
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
from datetime import datetime
import subprocess

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AWSConfig:
    """AWS configuration for LLM services"""
    region: str = "us-east-1"
    endpoint_url: Optional[str] = None
    bedrock_enabled: bool = True
    sagemaker_enabled: bool = False
    lambda_enabled: bool = True
    s3_bucket: str = "ai-dataset-generator"
    cloudwatch_enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "region": self.region,
            "endpoint_url": self.endpoint_url,
            "bedrock_enabled": self.bedrock_enabled,
            "sagemaker_enabled": self.sagemaker_enabled,
            "lambda_enabled": self.lambda_enabled,
            "s3_bucket": self.s3_bucket,
            "cloudwatch_enabled": self.cloudwatch_enabled
        }


@dataclass
class LLMResponse:
    """Standardized LLM response structure"""
    content: str
    model: str
    tokens_used: int
    latency_ms: float
    request_id: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "model": self.model,
            "tokens_used": self.tokens_used,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "metadata": self.metadata
        }


@dataclass
class ModelConfig:
    """Configuration for a specific LLM model"""
    name: str
    provider: str  # "bedrock", "sagemaker", "ollama", "openai"
    endpoint: Optional[str] = None
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    context_window: int = 8192
    cost_per_1k_tokens: float = 0.0
    capabilities: List[str] = field(default_factory=lambda: ["text-generation"])
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "provider": self.provider,
            "endpoint": self.endpoint,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "context_window": self.context_window,
            "cost_per_1k_tokens": self.cost_per_1k_tokens,
            "capabilities": self.capabilities
        }


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, config: ModelConfig) -> LLMResponse:
        """Generate response from LLM"""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if provider is available"""
        pass
    
    @abstractmethod
    def list_models(self) -> List[str]:
        """List available models"""
        pass


class BedrockProvider(LLMProvider):
    """AWS Bedrock LLM provider (mock implementation)"""
    
    def __init__(self, aws_config: AWSConfig):
        self.aws_config = aws_config
        self.available_models = [
            "anthropic.claude-3-sonnet",
            "anthropic.claude-3-haiku",
            "amazon.titan-text-express",
            "meta.llama3-8b-instruct"
        ]
    
    def generate(self, prompt: str, config: ModelConfig) -> LLMResponse:
        """Generate response using AWS Bedrock (mock)"""
        start_time = time.time()
        
        # Mock response generation
        response_content = self._mock_generate(prompt, config)
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response_content,
            model=config.name,
            tokens_used=len(prompt.split()) + len(response_content.split()),
            latency_ms=latency,
            request_id=hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:16],
            metadata={
                "provider": "bedrock",
                "region": self.aws_config.region
            }
        )
    
    def _mock_generate(self, prompt: str, config: ModelConfig) -> str:
        """Mock generation for demo purposes"""
        return f"[Bedrock {config.name}] Generated response for: {prompt[:100]}..."
    
    def is_available(self) -> bool:
        return self.aws_config.bedrock_enabled
    
    def list_models(self) -> List[str]:
        return self.available_models


class SageMakerProvider(LLMProvider):
    """AWS SageMaker LLM provider (mock implementation)"""
    
    def __init__(self, aws_config: AWSConfig):
        self.aws_config = aws_config
        self.available_models = [
            "huggingface-llama3-8b",
            "huggingface-mistral-7b",
            "custom-finetuned-model"
        ]
    
    def generate(self, prompt: str, config: ModelConfig) -> LLMResponse:
        """Generate response using SageMaker endpoint (mock)"""
        start_time = time.time()
        
        response_content = f"[SageMaker {config.name}] Endpoint response for prompt"
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response_content,
            model=config.name,
            tokens_used=len(prompt.split()) * 2,
            latency_ms=latency,
            request_id=hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:16],
            metadata={
                "provider": "sagemaker",
                "endpoint": config.endpoint
            }
        )
    
    def is_available(self) -> bool:
        return self.aws_config.sagemaker_enabled
    
    def list_models(self) -> List[str]:
        return self.available_models


class OllamaProvider(LLMProvider):
    """Local Ollama LLM provider"""
    
    def __init__(self):
        self.available_models = self._detect_models()
    
    def _detect_models(self) -> List[str]:
        """Detect locally available Ollama models"""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')[1:]  # Skip header
                return [line.split()[0] for line in lines if line.strip()]
        except Exception:
            pass
        return ["llama3.2:3b", "gpt-oss:20b"]  # Default fallbacks
    
    def generate(self, prompt: str, config: ModelConfig) -> LLMResponse:
        """Generate response using local Ollama"""
        start_time = time.time()
        
        try:
            result = subprocess.run(
                ["ollama", "run", config.name],
                input=prompt,
                capture_output=True,
                text=True,
                timeout=120
            )
            response_content = result.stdout.strip() if result.returncode == 0 else ""
        except Exception as e:
            logger.warning(f"Ollama generation failed: {e}")
            response_content = ""
        
        latency = (time.time() - start_time) * 1000
        
        return LLMResponse(
            content=response_content,
            model=config.name,
            tokens_used=len(prompt.split()) + len(response_content.split()),
            latency_ms=latency,
            request_id=hashlib.md5(f"{prompt}{time.time()}".encode()).hexdigest()[:16],
            metadata={"provider": "ollama", "local": True}
        )
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                timeout=5
            )
            return result.returncode == 0
        except Exception:
            return False
    
    def list_models(self) -> List[str]:
        return self.available_models


class LLMServiceRegistry:
    """Registry for managing multiple LLM providers and models"""
    
    def __init__(self, aws_config: Optional[AWSConfig] = None):
        self.aws_config = aws_config or AWSConfig()
        self.providers: Dict[str, LLMProvider] = {}
        self.model_configs: Dict[str, ModelConfig] = {}
        self._initialize_providers()
        self._register_default_models()
    
    def _initialize_providers(self):
        """Initialize all available providers"""
        # AWS Providers
        if self.aws_config.bedrock_enabled:
            self.providers["bedrock"] = BedrockProvider(self.aws_config)
        
        if self.aws_config.sagemaker_enabled:
            self.providers["sagemaker"] = SageMakerProvider(self.aws_config)
        
        # Local Providers
        ollama = OllamaProvider()
        if ollama.is_available():
            self.providers["ollama"] = ollama
    
    def _register_default_models(self):
        """Register default model configurations"""
        # Bedrock models
        self.register_model(ModelConfig(
            name="anthropic.claude-3-sonnet",
            provider="bedrock",
            max_tokens=4096,
            temperature=0.7,
            cost_per_1k_tokens=0.003,
            capabilities=["text-generation", "analysis", "code"]
        ))
        
        self.register_model(ModelConfig(
            name="amazon.titan-text-express",
            provider="bedrock",
            max_tokens=8192,
            temperature=0.7,
            cost_per_1k_tokens=0.0008,
            capabilities=["text-generation"]
        ))
        
        # Ollama models
        self.register_model(ModelConfig(
            name="llama3.2:3b",
            provider="ollama",
            max_tokens=4096,
            temperature=0.7,
            cost_per_1k_tokens=0.0,
            capabilities=["text-generation", "analysis"]
        ))
        
        self.register_model(ModelConfig(
            name="gpt-oss:20b",
            provider="ollama",
            max_tokens=4096,
            temperature=0.7,
            cost_per_1k_tokens=0.0,
            capabilities=["text-generation", "analysis", "comprehensive"]
        ))
    
    def register_model(self, config: ModelConfig):
        """Register a new model configuration"""
        self.model_configs[config.name] = config
        logger.info(f"Registered model: {config.name} ({config.provider})")
    
    def get_available_models(self) -> List[str]:
        """Get list of all available models"""
        available = []
        for name, config in self.model_configs.items():
            if config.provider in self.providers:
                if self.providers[config.provider].is_available():
                    available.append(name)
        return available
    
    def get_model_config(self, model_name: str) -> Optional[ModelConfig]:
        """Get configuration for a specific model"""
        return self.model_configs.get(model_name)


class LLMCoreService:
    """
    Core LLM Service with dynamic model selection and unified interface.
    Provides reusable service layer for all LLM interactions in the pipeline.
    """
    
    def __init__(self, aws_config: Optional[AWSConfig] = None, default_model: str = "llama3.2:3b"):
        self.registry = LLMServiceRegistry(aws_config)
        self.default_model = default_model
        self.request_history: List[Dict[str, Any]] = []
        self.metrics: Dict[str, Any] = {
            "total_requests": 0,
            "total_tokens": 0,
            "total_latency_ms": 0,
            "errors": 0
        }
    
    def generate(
        self,
        prompt: str,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> LLMResponse:
        """
        Generate LLM response with dynamic model selection.
        
        Args:
            prompt: Input prompt
            model: Model name (uses default if not specified)
            temperature: Override temperature
            max_tokens: Override max tokens
            **kwargs: Additional parameters
            
        Returns:
            LLMResponse with generated content and metadata
        """
        model_name = model or self.default_model
        config = self.registry.get_model_config(model_name)
        
        if not config:
            logger.warning(f"Model {model_name} not found, using default")
            config = self.registry.get_model_config(self.default_model)
            if not config:
                raise ValueError(f"No valid model configuration found")
        
        # Apply overrides
        if temperature is not None:
            config.temperature = temperature
        if max_tokens is not None:
            config.max_tokens = max_tokens
        
        # Get provider and generate
        provider = self.registry.providers.get(config.provider)
        if not provider or not provider.is_available():
            raise RuntimeError(f"Provider {config.provider} not available")
        
        try:
            response = provider.generate(prompt, config)
            self._update_metrics(response)
            self._log_request(prompt, response)
            return response
        except Exception as e:
            self.metrics["errors"] += 1
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def _update_metrics(self, response: LLMResponse):
        """Update service metrics"""
        self.metrics["total_requests"] += 1
        self.metrics["total_tokens"] += response.tokens_used
        self.metrics["total_latency_ms"] += response.latency_ms
    
    def _log_request(self, prompt: str, response: LLMResponse):
        """Log request for audit trail"""
        self.request_history.append({
            "timestamp": datetime.now().isoformat(),
            "prompt_hash": hashlib.md5(prompt.encode()).hexdigest(),
            "model": response.model,
            "tokens": response.tokens_used,
            "latency_ms": response.latency_ms,
            "request_id": response.request_id
        })
    
    def select_optimal_model(
        self,
        task_type: str = "text-generation",
        prefer_local: bool = True,
        max_cost: Optional[float] = None
    ) -> str:
        """
        Dynamically select optimal model based on criteria.
        
        Args:
            task_type: Type of task (text-generation, analysis, code)
            prefer_local: Prefer local models over cloud
            max_cost: Maximum cost per 1k tokens
            
        Returns:
            Name of selected model
        """
        available = self.registry.get_available_models()
        candidates = []
        
        for model_name in available:
            config = self.registry.get_model_config(model_name)
            if config and task_type in config.capabilities:
                if max_cost is None or config.cost_per_1k_tokens <= max_cost:
                    candidates.append(config)
        
        if not candidates:
            return self.default_model
        
        # Sort by preference
        if prefer_local:
            candidates.sort(key=lambda c: (c.provider != "ollama", c.cost_per_1k_tokens))
        else:
            candidates.sort(key=lambda c: c.cost_per_1k_tokens)
        
        return candidates[0].name
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get service metrics"""
        return {
            **self.metrics,
            "average_latency_ms": (
                self.metrics["total_latency_ms"] / self.metrics["total_requests"]
                if self.metrics["total_requests"] > 0 else 0
            )
        }
    
    def get_audit_trail(self) -> List[Dict[str, Any]]:
        """Get request audit trail"""
        return self.request_history


# Convenience function for quick access
def create_llm_service(
    aws_region: str = "us-east-1",
    default_model: str = "llama3.2:3b",
    enable_bedrock: bool = True,
    enable_sagemaker: bool = False
) -> LLMCoreService:
    """
    Create a configured LLM core service instance.
    
    Args:
        aws_region: AWS region for cloud services
        default_model: Default model to use
        enable_bedrock: Enable AWS Bedrock
        enable_sagemaker: Enable AWS SageMaker
        
    Returns:
        Configured LLMCoreService instance
    """
    aws_config = AWSConfig(
        region=aws_region,
        bedrock_enabled=enable_bedrock,
        sagemaker_enabled=enable_sagemaker
    )
    return LLMCoreService(aws_config, default_model)


if __name__ == "__main__":
    # Demo usage
    print("AWS LLM Core Services Demo")
    print("=" * 50)
    
    # Create service
    service = create_llm_service()
    
    # List available models
    print("\nAvailable Models:")
    for model in service.registry.get_available_models():
        config = service.registry.get_model_config(model)
        print(f"  - {model} ({config.provider})")
    
    # Dynamic model selection
    optimal = service.select_optimal_model(task_type="analysis", prefer_local=True)
    print(f"\nOptimal model for analysis: {optimal}")
    
    # Generate sample response
    print("\nGenerating sample response...")
    try:
        response = service.generate(
            prompt="Generate a test scenario for loan approval",
            model=optimal
        )
        print(f"Response: {response.content[:200]}...")
        print(f"Tokens used: {response.tokens_used}")
        print(f"Latency: {response.latency_ms:.2f}ms")
    except Exception as e:
        print(f"Generation failed (expected in demo mode): {e}")
    
    # Show metrics
    print("\nService Metrics:")
    print(json.dumps(service.get_metrics(), indent=2))
