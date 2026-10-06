"""Explainable AI framework for automated medical diagnosis using VLMs."""

import os
import socket

# Must be set before torch initialises its MPS backend: lets unsupported ops fall
# back to CPU instead of hard-erroring mid-backward-pass.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")


def _hub_reachable(timeout: float = 1.5) -> bool:
    try:
        socket.create_connection(("huggingface.co", 443), timeout=timeout).close()
        return True
    except OSError:
        return False


# Offline, every optional-file lookup (chat_template.jinja, audio_tokenizer_config.json)
# retries 5x with backoff before falling back to cache, adding minutes to startup.
# Weights are cached locally, so go straight to the cache when the hub is unreachable.
if "HF_HUB_OFFLINE" not in os.environ and not _hub_reachable():
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

__version__ = "0.1.0"
