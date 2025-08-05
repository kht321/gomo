"""Hugging Face LLM client implementation."""

from typing import Optional, Any, Union, List, Dict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    _has_transformers = True
except ImportError:
    _has_transformers = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    pipeline = None

from .interfaces import LLMClient


def _check_transformers():
    """Check if transformers is available."""
    if not _has_transformers:
        raise ImportError(
            "Transformers not installed. Install with: pip install 'gomoku-ai[huggingface]' " "or pip install transformers torch accelerate"
        )


def _get_dtype_for_device(device: str) -> Any:
    """Get optimal torch dtype for the given device."""
    if not _has_transformers:
        return None

    if device == "cuda":
        # Check if bfloat16 is supported on CUDA
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        else:
            return torch.float16
    elif device == "mps":
        # MPS works well with float16
        return torch.float16
    else:
        # CPU fallback
        return torch.float32


class HuggingFaceClient(LLMClient):
    """Generic Hugging Face client using transformers."""

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,  # Use Any instead of torch.dtype
        trust_remote_code: bool = False,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        max_length: int = 1024,
        padding: bool = True,
        truncation: bool = True,
        **model_kwargs,
    ):
        """
        Initialize Hugging Face client.

        Args:
            model_name: HuggingFace model identifier (e.g., "microsoft/DialoGPT-medium")
            device: Device to run model on ("cuda", "mps", "cpu", "auto")
                   - "auto": Automatically selects cuda > mps > cpu
                   - "cuda": NVIDIA GPU (auto-selects bfloat16 if supported, else float16)
                   - "mps": Apple Silicon GPU (M1/M2/M3) with float16
                   - "cpu": CPU fallback with float32
            torch_dtype: Torch data type (torch.float16, torch.bfloat16, etc.)
            trust_remote_code: Whether to trust remote code in model
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter (0.0-1.0)
            repetition_penalty: Repetition penalty (1.0 = no penalty)
            max_length: Maximum input length for tokenization
            padding: Whether to pad inputs
            truncation: Whether to truncate inputs
            **model_kwargs: Additional model arguments
        """
        _check_transformers()  # Ensure transformers is available

        self.model_name = model
        
        # Generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        # Tokenization parameters
        self.max_length = max_length
        self.padding = padding
        self.truncation = truncation

        # Determine device with MPS support
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Determine torch dtype with optimal selection
        if torch_dtype is None:
            self.torch_dtype = _get_dtype_for_device(self.device)
        else:
            self.torch_dtype = torch_dtype

        print(f"Loading HuggingFace model: {model}")
        print(f"Device: {self.device}, Dtype: {self.torch_dtype}")

        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code, **model_kwargs)

            # Add pad token if missing (use a different token than eos to avoid attention mask issues)
            if self.tokenizer.pad_token is None:
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                elif hasattr(self.tokenizer, "add_special_tokens"):
                    # Add a dedicated pad token
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                else:
                    # Fallback to eos token but we'll handle attention mask manually
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                model,
                torch_dtype=self.torch_dtype,
                device_map=self.device if self.device not in ["cpu", "mps"] else None,
                trust_remote_code=trust_remote_code,
                **model_kwargs,
            )

            # Move to device if CPU or MPS (device_map doesn't work well with MPS)
            if self.device in ["cpu", "mps"]:
                self.model = self.model.to(self.device)

            # Resize model embeddings if we added new tokens
            if len(self.tokenizer) > self.model.config.vocab_size:
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"📏 Resized model embeddings to {len(self.tokenizer)} tokens")

            print(f"✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model {model}: {e}")
            raise

    async def complete(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """Send messages to Hugging Face model and return response."""
        try:
            # Convert messages to prompt format
            if isinstance(messages, str):
                # Legacy string prompt
                prompt = messages
            else:
                # Convert messages list to prompt
                prompt = self._messages_to_prompt(messages)

            # Tokenize input with attention mask using stored parameters
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=self.padding, 
                truncation=self.truncation,
                max_length=self.max_length
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=self.repetition_penalty,
                )

            # Decode response (only the new tokens)
            input_length = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

            return response

        except Exception as e:
            # Generic fallback response
            print(f"HuggingFace model error: {e}")
            raise Exception(f"HuggingFace model error: {e}")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages list to a properly formatted prompt using the model's chat template."""
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            raise ValueError(
                f"Model {self.model_name} does not support chat templates. "
                "Use a chat-compatible model or pass a simple string prompt instead."
            )


class HuggingFacePipelineClient(LLMClient):
    """Generic Hugging Face client using pipelines."""

    def __init__(
        self, 
        model_name: str, 
        device: Optional[str] = None, 
        torch_dtype: Optional[Any] = None,
        max_new_tokens: int = 100,
        temperature: float = 0.7,
        do_sample: bool = True,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        **pipeline_kwargs
    ):
        """
        Initialize with HuggingFace pipeline.
        
        Args:
            model_name: HuggingFace model identifier
            device: Device to run on ("cuda", "mps", "cpu", "auto")
            torch_dtype: Torch data type
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0.0-2.0)
            do_sample: Whether to use sampling
            top_p: Top-p sampling parameter (0.0-1.0)
            repetition_penalty: Repetition penalty (1.0 = no penalty)
            **pipeline_kwargs: Additional pipeline arguments
        """
        _check_transformers()  # Ensure transformers is available

        # Determine device with MPS support
        if device is None or device == "auto":
            if torch.cuda.is_available():
                device_id = 0
                device_type = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device_id = "mps"  # For pipelines, MPS is passed as string
                device_type = "mps"
            else:
                device_id = -1
                device_type = "cpu"
        elif device == "cuda":
            device_id = 0
            device_type = "cuda"
        elif device == "mps":
            device_id = "mps"
            device_type = "mps"
        else:
            device_id = -1
            device_type = "cpu"

        # Determine torch dtype with optimal selection
        if torch_dtype is None:
            torch_dtype = _get_dtype_for_device(device_type)
        
        # Store generation parameters
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty

        print(f"Loading HuggingFace pipeline: {model_name}")
        print(f"Device: {device_type} ({device_id}), Dtype: {torch_dtype}")

        try:
            # Create text generation pipeline
            self.pipeline = pipeline(
                "text-generation", model=model_name, device=device_id, torch_dtype=torch_dtype, return_full_text=False, **pipeline_kwargs
            )

            # Fix pad token for pipeline tokenizer
            if self.pipeline.tokenizer.pad_token is None:
                if self.pipeline.tokenizer.unk_token is not None:
                    self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.unk_token
                elif hasattr(self.pipeline.tokenizer, "add_special_tokens"):
                    self.pipeline.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                    # Resize model embeddings if we added new tokens
                    if len(self.pipeline.tokenizer) > self.pipeline.model.config.vocab_size:
                        self.pipeline.model.resize_token_embeddings(len(self.pipeline.tokenizer))
                else:
                    self.pipeline.tokenizer.pad_token = self.pipeline.tokenizer.eos_token

            print(f"✅ Pipeline loaded successfully!")

        except Exception as e:
            print(f"❌ Error creating pipeline for {model_name}: {e}")
            raise

    async def complete(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        """Generate response using pipeline."""
        try:
            # Convert messages to prompt format
            if isinstance(messages, str):
                # Legacy string prompt
                prompt = messages
            else:
                # Convert messages list to prompt
                prompt = self._messages_to_prompt(messages)

            # Generate using stored parameters
            results = self.pipeline(
                prompt,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                do_sample=self.do_sample,
                top_p=self.top_p,
                repetition_penalty=self.repetition_penalty,
                eos_token_id=self.pipeline.tokenizer.eos_token_id,
            )

            # Extract response
            if results and len(results) > 0:
                response = results[0]["generated_text"]
                return response
            else:
                raise Exception("No response generated from pipeline")

        except Exception as e:
            print(f"Pipeline error: {e}")
            raise Exception(f"Pipeline error: {e}")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert messages list to a properly formatted prompt using the model's chat template."""
        if hasattr(self.pipeline.tokenizer, "apply_chat_template") and self.pipeline.tokenizer.chat_template:
            return self.pipeline.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            raise ValueError(
                f"Pipeline model does not support chat templates. " "Use a chat-compatible model or pass a simple string prompt instead."
            )


# Popular model configurations
POPULAR_MODELS = {
    "gpt2": {
        "model_name": "gpt2",
        "description": "OpenAI GPT-2 (small, fast)",
        "torch_dtype": "float32",  # Use string instead of torch type
    },
    "gpt2-medium": {
        "model_name": "gpt2-medium",
        "description": "OpenAI GPT-2 Medium",
        "torch_dtype": "float32",
    },
    "microsoft/DialoGPT-medium": {
        "model_name": "microsoft/DialoGPT-medium",
        "description": "Microsoft DialoGPT for conversations",
        "torch_dtype": "float16",
    },
    "microsoft/DialoGPT-large": {
        "model_name": "microsoft/DialoGPT-large",
        "description": "Microsoft DialoGPT Large",
        "torch_dtype": "float16",
    },
    "google/flan-t5-base": {
        "model_name": "google/flan-t5-base",
        "description": "Google FLAN-T5 Base",
        "torch_dtype": "float32",
    },
    "google/flan-t5-large": {
        "model_name": "google/flan-t5-large",
        "description": "Google FLAN-T5 Large",
        "torch_dtype": "float16",
    },
}


def create_huggingface_client(model_name: str, use_pipeline: bool = False, device: Optional[str] = None, **kwargs) -> LLMClient:
    """
    Convenience function to create HuggingFace client.

    Args:
        model_name: Model name or key from POPULAR_MODELS
        use_pipeline: Whether to use pipeline implementation
        device: Device to use
        **kwargs: Additional arguments

    Returns:
        Configured HuggingFace client
    """
    # Check if it's a popular model shortcut
    if model_name in POPULAR_MODELS:
        config = POPULAR_MODELS[model_name].copy()
        model_name = config.pop("model_name")

        # Remove non-model arguments
        config.pop("description", None)  # Remove description

        # Convert string dtype to torch dtype if transformers is available
        if _has_transformers and "torch_dtype" in config:
            dtype_str = config["torch_dtype"]
            if dtype_str == "float16":
                config["torch_dtype"] = torch.float16
            elif dtype_str == "float32":
                config["torch_dtype"] = torch.float32
            elif dtype_str == "bfloat16":
                config["torch_dtype"] = torch.bfloat16

        config.update(kwargs)  # Override with user kwargs
        kwargs = config

    # Create client
    if use_pipeline:
        return HuggingFacePipelineClient(model_name, device=device, **kwargs)
    else:
        return HuggingFaceClient(model_name, device=device, **kwargs)
