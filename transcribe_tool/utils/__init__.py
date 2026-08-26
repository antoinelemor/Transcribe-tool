"""Utility modules for Transcribe Tool."""

# Imported first: setting NoDefaultCurrentDirectoryInExePath must happen before
# any module below can spawn a subprocess.
from .platform_compat import (
    configure_stdio,
    resolve_binary,
    run_tool,
    child_utf8_env,
    write_csv_text,
    write_text_file,
    safe_output_path,
    restrict_permissions,
    load_saved_hf_token,
)
# Everything below is resolved lazily (PEP 562). The NLP submodules pull in
# torch/transformers/spacy, which costs several seconds; importing this package
# for a helper alone -- as __main__.py does for configure_stdio -- must stay cheap.
_LAZY_ATTRS = {
    "LanguageDetector": "language_detector",
    "DetectionMethod": "language_detector",
    "detect_language": "language_detector",
    "detect_language_full": "language_detector",
    "get_language_name": "language_detector",
    "get_supported_detection_languages": "language_detector",
    "get_detection_info": "language_detector",
    "LANGUAGE_CODES": "language_detector",
    "Tokenizer": "tokenizer",
    "segment_text": "tokenizer",
    "AudioProcessor": "audio_processor",
    "DocumentLoader": "document_loader",
    "Document": "document_loader",
    "load_document": "document_loader",
    "load_documents": "document_loader",
    "TextTokenizer": "text_tokenization",
    "NLPFeatures": "text_tokenization",
    "TokenizationResult": "text_tokenization",
    "TokenizedSegment": "text_tokenization",
    "tokenize_text": "text_tokenization",
    "tokenize_dataframe": "text_tokenization",
    "format_tokenization_csv": "text_tokenization",
    "format_tokenization_json": "text_tokenization",
    "format_tokenization_rds": "text_tokenization",
    "get_available_nlp_features": "text_tokenization",
    "get_supported_languages": "text_tokenization",
}

# language_detector exports this under its plain name; the package re-exports it
# under a disambiguated one so it does not collide with text_tokenization's.
_LAZY_RENAMES = {"get_supported_detection_languages": "get_supported_languages"}


def __getattr__(name):
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, _LAZY_RENAMES.get(name, name))
    globals()[name] = value
    return value


def __dir__():
    return sorted(__all__)

__all__ = [
    # Platform compatibility
    "configure_stdio",
    "resolve_binary",
    "run_tool",
    "child_utf8_env",
    "write_csv_text",
    "write_text_file",
    "safe_output_path",
    "restrict_permissions",
    "load_saved_hf_token",
    # Language detection (SOTA)
    "LanguageDetector",
    "DetectionMethod",
    "detect_language",
    "detect_language_full",
    "get_language_name",
    "get_supported_detection_languages",
    "get_detection_info",
    "LANGUAGE_CODES",
    # Legacy tokenizer
    "Tokenizer",
    "segment_text",
    # Audio processing
    "AudioProcessor",
    # Document loading
    "DocumentLoader",
    "Document",
    "load_document",
    "load_documents",
    # SOTA Text tokenization
    "TextTokenizer",
    "NLPFeatures",
    "TokenizationResult",
    "TokenizedSegment",
    "tokenize_text",
    "tokenize_dataframe",
    "format_tokenization_csv",
    "format_tokenization_json",
    "format_tokenization_rds",
    "get_available_nlp_features",
    "get_supported_languages",
]
