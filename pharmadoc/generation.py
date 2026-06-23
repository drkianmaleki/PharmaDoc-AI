"""
Answer generation, source formatting, and confidence scoring for PharmaDoc AI.

Formats retrieval sources for display, estimates answer confidence from
semantic scores and evidence-term matching, and routes generation requests
to either a local FLAN-T5 model or the OpenAI API (gpt-4o-mini).
"""

import os

from .config import LOCAL_LLM_MODEL_NAME, MODEL_CATALOG

# Module-level cache for the local FLAN-T5 generator; populated on first use.
_local_generator = None



def _get_retrieval_display_score(chunk):
    """
    Return the best available score for source display.

    Retrieval branches may expose:
        - score
        - rerank_score
        - semantic_score

    Plot-specific retrieval may begin with raw content items, so this
    function avoids KeyError when one score field is absent.
    """
    if not isinstance(chunk, dict):
        return 0.0

    for field_name in (
        "score",
        "rerank_score",
        "semantic_score",
    ):
        value = chunk.get(field_name)

        if value is None:
            continue

        try:
            return float(value)
        except (TypeError, ValueError):
            continue

    return 0.0


def _get_source_page(chunk):
    """
    Return the most appropriate page number from a content item.
    """
    if not isinstance(chunk, dict):
        return "Unknown"

    for field_name in (
        "page_start",
        "page",
        "page_number",
    ):
        value = chunk.get(field_name)

        if value is not None:
            return value

    return "Unknown"


def format_sources(retrieved_chunks):
    """
    Create a clean and defensive source summary for the UI.

    This version supports results from:
        - semantic retrieval
        - hybrid reranking
        - deterministic structured retrieval
        - plot-specific retrieval
    """
    if not retrieved_chunks:
        return "No sources retrieved."

    source_lines = []

    for index, chunk in enumerate(
        retrieved_chunks,
        start=1,
    ):
        if not isinstance(chunk, dict):
            continue

        filename = str(
            chunk.get("file", "Unknown file")
        )

        page = _get_source_page(chunk)

        content_type = str(
            chunk.get("content_type", "unknown")
        )

        score = _get_retrieval_display_score(
            chunk
        )

        source_lines.append(
            f"{index}. {filename} | "
            f"page {page} | "
            f"{content_type} | "
            f"score: {score:.4f}"
        )

    if not source_lines:
        return "No sources retrieved."

    return "\n".join(source_lines)



def estimate_retrieval_confidence(retrieved_chunks):
    """
    Estimate retrieval confidence from the original semantic score.

    Reranking changes result order, but confidence remains based on
    normalized embedding similarity rather than heuristic bonuses.
    """
    if not retrieved_chunks:
        return "Low", 0.0

    top_result = retrieved_chunks[0]

    top_score = float(
        top_result.get(
            "semantic_score",
            top_result.get("score", 0.0)
        )
    )

    if top_score >= 0.45:
        confidence = "High"
    elif top_score >= 0.30:
        confidence = "Medium"
    else:
        confidence = "Low"

    return confidence, top_score



def load_optional_api_keys():
    """Load optional API keys without failing outside Colab."""
    try:
        from google.colab import userdata
    except ImportError:
        print("Not running in Colab; optional API keys were not loaded.")
        return

    for key_name in ["OPENAI_API_KEY"]:
        try:
            value = userdata.get(key_name)
            if value:
                os.environ[key_name] = value
                print(f"Loaded {key_name}.")
            else:
                print(f"{key_name} is not configured.")
        except Exception:
            print(f"{key_name} is not configured.")


load_optional_api_keys()



def generate_with_local_model(
    prompt,
    model_name=LOCAL_LLM_MODEL_NAME,
    max_tokens=80,
):
    """
    Generate an answer using the assignment-compliant
    open-source FLAN-T5 model.
    """
    global _local_generator

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    if _local_generator is None:
        print(f"Loading local model: {model_name}")

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name,
            dtype=(
                torch.float16
                if torch.cuda.is_available()
                else torch.float32
            ),
        )

        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        model = model.to(device)
        model.eval()

        _local_generator = {
            "tokenizer": tokenizer,
            "model": model,
            "device": device,
            "model_name": model_name,
        }

        print(f"Local model loaded on: {device}")

    tokenizer = _local_generator["tokenizer"]
    model = _local_generator["model"]
    device = _local_generator["device"]

    encoded_inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=1024,
    )

    encoded_inputs = {
        key: value.to(device)
        for key, value in encoded_inputs.items()
    }

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded_inputs,
            max_new_tokens=int(max_tokens),
            do_sample=False,
            num_beams=4,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )

    answer = tokenizer.decode(
        generated_ids[0],
        skip_special_tokens=True,
    )

    return answer.strip()


def generate_with_openai(
    prompt,
    model_name="gpt-4o-mini",
    max_tokens=80,
):
    """
    Generate an answer using OpenAI when an API key is available.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "OpenAI API key was not found. Add OPENAI_API_KEY "
            "to Colab Secrets or select the open-source model."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    response = client.responses.create(
        model=model_name,
        input=prompt,
        max_output_tokens=int(max_tokens),
    )

    return response.output_text.strip()


def generate_answer_with_model(
    prompt,
    model_choice,
    max_tokens=80,
):
    """
    Route answer generation to the selected model provider.
    """
    if model_choice not in MODEL_CATALOG:
        raise ValueError(
            f"Unknown model choice: {model_choice}. "
            f"Available choices: {list(MODEL_CATALOG.keys())}"
        )

    model_info = MODEL_CATALOG[model_choice]
    provider = model_info["provider"]
    model_name = model_info["model_name"]

    if provider == "local_huggingface":
        return generate_with_local_model(
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
        )

    if provider == "openai":
        return generate_with_openai(
            prompt=prompt,
            model_name=model_name,
            max_tokens=max_tokens,
        )

    raise ValueError(
        f"Unsupported model provider: {provider}"
    )

