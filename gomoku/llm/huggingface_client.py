# --- BEGIN PATCH ---
"""Hugging Face LLM client implementation."""

from typing import Optional, Any, Union, List, Dict

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    try:
        from transformers import BitsAndBytesConfig  # optional
    except Exception:
        BitsAndBytesConfig = None
    _has_transformers = True
except ImportError:
    _has_transformers = False
    torch = None
    AutoTokenizer = None
    AutoModelForCausalLM = None
    BitsAndBytesConfig = None
    pipeline = None

from .interfaces import LLMClient


def _check_transformers():
    if not _has_transformers:
        raise ImportError(
            "Transformers not installed. Install with: "
            "pip install 'gomoku-ai[huggingface]' or "
            "pip install transformers torch accelerate bitsandbytes"
        )


def _get_dtype_for_device(device: str) -> Any:
    if not _has_transformers:
        return None
    if device == "cuda":
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    elif device == "mps":
        return torch.float16
    return torch.float32


def _maybe_build_bnb_config(kwargs: Dict[str, Any]):
    """
    Build a BitsAndBytesConfig from loose kwargs if user passed load_in_4bit=True
    and didn't supply an explicit quantization_config.
    """
    if not BitsAndBytesConfig:
        return None
    if "quantization_config" in kwargs and kwargs["quantization_config"] is not None:
        return kwargs["quantization_config"]
    if not kwargs.get("load_in_4bit", False):
        return None
    # allow string dtype from callers
    comp = kwargs.get("bnb_4bit_compute_dtype", "bfloat16")
    if isinstance(comp, str) and hasattr(torch, comp):
        comp = getattr(torch, comp)
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=kwargs.get("bnb_4bit_quant_type", "nf4"),
        bnb_4bit_compute_dtype=comp if comp is not None else torch.float16,
        bnb_4bit_use_double_quant=kwargs.get("bnb_4bit_use_double_quant", True),
    )


class HuggingFaceClient(LLMClient):
    """Generic Hugging Face client using transformers."""

    def __init__(
        self,
        model: str,
        device: Optional[str] = None,
        torch_dtype: Optional[Any] = None,
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
        _check_transformers()

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

        # Device
        if device is None or device == "auto":
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        # Dtype
        self.torch_dtype = _get_dtype_for_device(self.device) if torch_dtype is None else torch_dtype

        # Quantization / offload / device_map plumbing
        # - allow caller to pass device_map explicitly (recommended: "auto")
        # - if user asks for 4-bit (load_in_4bit=True) and no explicit device_map, default to "auto"
        quantization_config = _maybe_build_bnb_config(model_kwargs)
        device_map = model_kwargs.pop("device_map", None)
        offload_folder = model_kwargs.get("offload_folder")
        offload_state_dict = model_kwargs.get("offload_state_dict", False)
        max_memory = model_kwargs.get("max_memory")

        if device_map is None and quantization_config is not None:
            device_map = "auto"  # best default when quantized / tight VRAM

        # IMPORTANT: never pass device_map="cuda" etc.; only "auto" or a dict is valid.
        if isinstance(device_map, str) and device_map not in {"auto"}:
            device_map = None

        print(f"Loading HuggingFace model: {model}")
        print(f"Device: {self.device}, Dtype: {self.torch_dtype}")

        try:
            # Tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                **{k: v for k, v in model_kwargs.items() if k not in {
                    "low_cpu_mem_usage", "quantization_config", "max_memory",
                    "offload_folder", "offload_state_dict", "load_in_4bit",
                    "bnb_4bit_quant_type", "bnb_4bit_compute_dtype", "bnb_4bit_use_double_quant",
                    "device_map"
                }}
            )

            # Add pad token if missing
            if self.tokenizer.pad_token is None:
                if self.tokenizer.unk_token is not None:
                    self.tokenizer.pad_token = self.tokenizer.unk_token
                elif hasattr(self.tokenizer, "add_special_tokens"):
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                else:
                    self.tokenizer.pad_token = self.tokenizer.eos_token

            # Model
            from_kwargs = dict(
                trust_remote_code=trust_remote_code,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
            if quantization_config is not None:
                from_kwargs["quantization_config"] = quantization_config
            if device_map is not None:
                from_kwargs["device_map"] = device_map
            if max_memory is not None:
                from_kwargs["max_memory"] = max_memory
            if offload_folder is not None:
                from_kwargs["offload_folder"] = offload_folder
            if offload_state_dict:
                from_kwargs["offload_state_dict"] = True

            self.model = AutoModelForCausalLM.from_pretrained(model, **from_kwargs)

            # If we didn't use device_map, put the model on the requested single device
            if device_map is None:
                if self.device in ["cpu", "mps"]:
                    self.model = self.model.to(self.device)
                elif self.device == "cuda":
                    # single-GPU move
                    self.model = self.model.to("cuda")

            # Resize embeddings if we added a pad token
            if len(self.tokenizer) > getattr(self.model.config, "vocab_size", len(self.tokenizer)):
                self.model.resize_token_embeddings(len(self.tokenizer))
                print(f"📏 Resized model embeddings to {len(self.tokenizer)} tokens")

            print("✅ Model loaded successfully!")

        except Exception as e:
            print(f"❌ Error loading model {model}: {e}")
            raise

    def _pick_input_device(self):
        """
        Pick a device for inputs:
        - with device_map='auto', use the device of the first parameter we can find
        - otherwise, use the configured single device
        """
        try:
            return next(self.model.parameters()).device
        except Exception:
            # fallback to configured device
            if self.device == "cuda" and torch.cuda.is_available():
                return torch.device("cuda")
            if self.device == "mps":
                return torch.device("mps")
            return torch.device("cpu")

    async def complete(self, messages: Union[str, List[Dict[str, str]]]) -> str:
        try:
            if isinstance(messages, str):
                prompt = messages
            else:
                prompt = self._messages_to_prompt(messages)

            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=self.padding,
                truncation=self.truncation,
                max_length=self.max_length,
            )

            # Move inputs to a sensible device (works with device_map='auto')
            tgt = self._pick_input_device()
            inputs = {k: v.to(tgt) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs.get("attention_mask"),
                    max_new_tokens=self.max_new_tokens,
                    temperature=self.temperature,
                    do_sample=self.do_sample,
                    top_p=self.top_p,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=self.repetition_penalty,
                )

            input_len = inputs["input_ids"].shape[1]
            generated_tokens = outputs[0][input_len:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            return response

        except Exception as e:
            print(f"HuggingFace model error: {e}")
            raise Exception(f"HuggingFace model error: {e}")

    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        raise ValueError(
            f"Model {self.model_name} does not support chat templates. "
            "Use a chat-compatible model or pass a simple string prompt instead."
        )
# --- END PATCH ---
HuggingFacePipelineClient = HuggingFaceClient