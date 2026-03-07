#!/usr/bin/env python3
"""Inference script for running prompts through a fine-tuned LLM adapter.

Loads the base model with a LoRA adapter and generates responses for
given prompts.
"""

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def load_model(
    base_model: str,
    adapter_path: str,
    quantization: int = 0,
    device: str = "auto",
):
    # Purpose: build an inference-ready model stack by loading the tokenizer,
    # loading the base LLM, then attaching the trained LoRA adapter weights.
    # Beginner view: this is the setup step that prepares everything before
    # you can ask prompts and get responses.
    """Load the base model with a LoRA adapter merged in.

    Args:
        base_model: HuggingFace model ID or local path.
        adapter_path: Path to the trained LoRA adapter directory.
        quantization: 0 for none, 4 or 8 for bitsandbytes quantization.
        device: Device mapping strategy.

    Returns:
        Tuple of (model, tokenizer).
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
    import torch

    logger.info(f"Loading tokenizer from {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {
        "trust_remote_code": True,
        "device_map": device,
    }

    if quantization == 4:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
        )
        logger.info("Using 4-bit quantization")
    elif quantization == 8:
        load_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
        logger.info("Using 8-bit quantization")
    else:
        load_kwargs["torch_dtype"] = torch.float16

    logger.info(f"Loading base model from {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **load_kwargs)

    logger.info(f"Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    logger.info("Model loaded successfully")
    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    # Purpose: take one prompt, convert it into model input tokens, run text
    # generation, and decode only the newly generated part of the output.
    # Beginner view: this is the "ask question -> get model answer" function.
    """Generate a text response for a given prompt.

    Args:
        model: The loaded language model.
        tokenizer: The tokenizer.
        prompt: The input prompt string.
        max_new_tokens: Maximum number of new tokens to generate.
        temperature: Sampling temperature.
        top_p: Top-p (nucleus) sampling threshold.
        do_sample: Whether to use sampling or greedy decoding.

    Returns:
        The generated response string.
    """
    import torch

    messages = [{"role": "user", "content": prompt}]

    # Try chat template first, fall back to raw prompt
    try:
        input_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        input_text = f"### Instruction:\n{prompt}\n\n### Response:\n"

    inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only the generated tokens (skip the input)
    generated_ids = outputs[0][inputs["input_ids"].shape[1] :]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


def main():
    # Purpose: CLI entrypoint that reads command-line options, loads the model
    # once, gathers prompts (single/file/default), and prints responses.
    # Beginner view: this controls the full user workflow when you run the file.
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument(
        "--base-model",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Base model name or path",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        required=True,
        help="Path to LoRA adapter directory",
    )
    parser.add_argument("--prompt", type=str, help="Single prompt to run")
    parser.add_argument(
        "--prompts-file", type=str, help="File with prompts (one per line)"
    )
    parser.add_argument("--quantization", type=int, default=0, choices=[0, 4, 8])
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    model, tokenizer = load_model(
        base_model=args.base_model,
        adapter_path=args.adapter_path,
        quantization=args.quantization,
    )

    prompts = []
    if args.prompt:
        prompts.append(args.prompt)
    elif args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default demo prompts
        prompts = [
            "Explain the common symptoms of a heart attack.",
            "What are the risk factors for type 2 diabetes?",
            "Describe the management of acute pancreatitis.",
        ]

    print("\n" + "=" * 70)
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Prompt {i}] {prompt}")
        print("-" * 70)
        response = generate_response(
            model,
            tokenizer,
            prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
        )
        print(f"{response}")
        print("=" * 70)


if __name__ == "__main__":
    main()
