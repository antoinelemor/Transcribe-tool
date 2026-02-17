"""
SOTA Text Tokenization Module for Transcribe Tool.

Provides state-of-the-art text tokenization with configurable options:

SEGMENTATION LEVELS:
- 1, 2, 3, N sentences per segment
- Paragraph-based segmentation

NLP FEATURES (can be combined):
- Lemmatization (word base forms)
- POS tagging (part-of-speech)
- NER (named entity recognition)

OUTPUT FORMAT:
- CSV with JSON-encoded structured data for machine readability
- JSON for full data export

Model hierarchy:
1. wtpsplit (SaT-12l) - SOTA sentence segmentation, 85+ languages
2. spaCy transformer models (_trf) - Best NLP accuracy
3. spaCy large models - Good fallback
4. stanza - Academic SOTA alternative
"""

import json
import os
import re
import csv
import io
from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import logging
import queue
import threading

# ============================================================================
# Import NLP libraries with graceful fallbacks
# ============================================================================

# wtpsplit - SOTA sentence segmentation
try:
    from wtpsplit import SaT
    HAS_WTPSPLIT = True
except ImportError:
    HAS_WTPSPLIT = False
    SaT = None

# spaCy - NLP pipeline
try:
    import spacy
    HAS_SPACY = True
except ImportError:
    HAS_SPACY = False
    spacy = None

# stanza - Stanford NLP (fallback)
try:
    import stanza
    HAS_STANZA = True
except ImportError:
    HAS_STANZA = False
    stanza = None

# NLTK - fallback tokenization
try:
    import nltk
    from nltk.tokenize import sent_tokenize
    HAS_NLTK = True
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
except ImportError:
    HAS_NLTK = False


# ============================================================================
# Data structures
# ============================================================================

@dataclass
class NLPFeatures:
    """Configuration for which NLP features to extract."""
    lemmatization: bool = False
    pos_tagging: bool = False
    ner: bool = False

    def any_enabled(self) -> bool:
        return self.lemmatization or self.pos_tagging or self.ner

    @classmethod
    def all_features(cls) -> 'NLPFeatures':
        return cls(lemmatization=True, pos_tagging=True, ner=True)

    @classmethod
    def none(cls) -> 'NLPFeatures':
        return cls()


@dataclass
class TokenInfo:
    """Information about a single token."""
    text: str
    lemma: Optional[str] = None
    pos: Optional[str] = None
    tag: Optional[str] = None  # Fine-grained POS

    def to_dict(self) -> Dict[str, Any]:
        d = {'text': self.text}
        if self.lemma:
            d['lemma'] = self.lemma
        if self.pos:
            d['pos'] = self.pos
        if self.tag:
            d['tag'] = self.tag
        return d


@dataclass
class EntityInfo:
    """Information about a named entity."""
    text: str
    label: str
    start_char: int
    end_char: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'label': self.label,
            'start': self.start_char,
            'end': self.end_char
        }


@dataclass
class TokenizedSegment:
    """A single tokenized segment with all extracted features."""
    segment_id: int
    document_id: str
    text: str
    sentences: List[str] = field(default_factory=list)
    tokens: List[TokenInfo] = field(default_factory=list)
    entities: List[EntityInfo] = field(default_factory=list)

    @property
    def sentence_count(self) -> int:
        return len(self.sentences) if self.sentences else 1

    @property
    def word_count(self) -> int:
        return len(self.tokens) if self.tokens else len(self.text.split())

    def tokens_to_json(self) -> str:
        """Serialize tokens to JSON string."""
        if not self.tokens:
            return "[]"
        return json.dumps([t.to_dict() for t in self.tokens], ensure_ascii=False)

    def entities_to_json(self) -> str:
        """Serialize entities to JSON string."""
        if not self.entities:
            return "[]"
        return json.dumps([e.to_dict() for e in self.entities], ensure_ascii=False)

    def lemmas_to_json(self) -> str:
        """Get lemmas as JSON array."""
        if not self.tokens:
            return "[]"
        lemmas = [t.lemma for t in self.tokens if t.lemma]
        return json.dumps(lemmas, ensure_ascii=False)

    def pos_to_json(self) -> str:
        """Get POS tags as JSON array."""
        if not self.tokens:
            return "[]"
        pos_tags = [t.pos for t in self.tokens if t.pos]
        return json.dumps(pos_tags, ensure_ascii=False)


@dataclass
class TokenizationResult:
    """Result of tokenizing a document."""
    success: bool = True
    segments: List[TokenizedSegment] = field(default_factory=list)
    document_id: str = ""
    source_file: Optional[str] = None
    language: str = "unknown"
    segmentation_level: int = 1
    nlp_features: NLPFeatures = field(default_factory=NLPFeatures)
    total_sentences: int = 0
    total_words: int = 0
    total_entities: int = 0
    processing_info: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


# ============================================================================
# Language-specific model configurations
# ============================================================================

# spaCy SOTA models per language (transformer > large > medium > small)
SPACY_MODELS = {
    'en': ['en_core_web_trf', 'en_core_web_lg', 'en_core_web_md', 'en_core_web_sm'],
    'fr': ['fr_dep_news_trf', 'fr_core_news_lg', 'fr_core_news_md', 'fr_core_news_sm'],
    'de': ['de_dep_news_trf', 'de_core_news_lg', 'de_core_news_md', 'de_core_news_sm'],
    'es': ['es_dep_news_trf', 'es_core_news_lg', 'es_core_news_md', 'es_core_news_sm'],
    'it': ['it_core_news_lg', 'it_core_news_md', 'it_core_news_sm'],
    'pt': ['pt_core_news_lg', 'pt_core_news_md', 'pt_core_news_sm'],
    'nl': ['nl_core_news_lg', 'nl_core_news_md', 'nl_core_news_sm'],
    'pl': ['pl_core_news_lg', 'pl_core_news_md', 'pl_core_news_sm'],
    'ru': ['ru_core_news_lg', 'ru_core_news_md', 'ru_core_news_sm'],
    'ja': ['ja_core_news_trf', 'ja_core_news_lg', 'ja_core_news_md', 'ja_core_news_sm'],
    'zh': ['zh_core_web_trf', 'zh_core_web_lg', 'zh_core_web_md', 'zh_core_web_sm'],
    'da': ['da_core_news_lg', 'da_core_news_md', 'da_core_news_sm'],
    'el': ['el_core_news_lg', 'el_core_news_md', 'el_core_news_sm'],
    'lt': ['lt_core_news_lg', 'lt_core_news_md', 'lt_core_news_sm'],
    'nb': ['nb_core_news_lg', 'nb_core_news_md', 'nb_core_news_sm'],
    'ro': ['ro_core_news_lg', 'ro_core_news_md', 'ro_core_news_sm'],
    'ca': ['ca_core_news_lg', 'ca_core_news_md', 'ca_core_news_sm'],
    'hr': ['hr_core_news_lg', 'hr_core_news_md', 'hr_core_news_sm'],
    'fi': ['fi_core_news_lg', 'fi_core_news_md', 'fi_core_news_sm'],
    'ko': ['ko_core_news_lg', 'ko_core_news_md', 'ko_core_news_sm'],
    'mk': ['mk_core_news_lg', 'mk_core_news_md', 'mk_core_news_sm'],
    'sv': ['sv_core_news_lg', 'sv_core_news_md', 'sv_core_news_sm'],
    'uk': ['uk_core_news_lg', 'uk_core_news_md', 'uk_core_news_sm'],
}

# NLTK language mapping
NLTK_LANGUAGES = {
    'en': 'english', 'fr': 'french', 'de': 'german', 'es': 'spanish',
    'it': 'italian', 'pt': 'portuguese', 'nl': 'dutch', 'pl': 'polish',
    'ru': 'russian', 'cs': 'czech', 'da': 'danish', 'et': 'estonian',
    'fi': 'finnish', 'el': 'greek', 'no': 'norwegian', 'nb': 'norwegian',
    'sl': 'slovene', 'sv': 'swedish', 'tr': 'turkish'
}

# ============================================================================
# Model cache (singleton for performance)
# ============================================================================

_wtp_model = None
_spacy_cache: Dict[str, Any] = {}
_stanza_cache: Dict[str, Any] = {}


# ============================================================================
# Batch processing configuration
# ============================================================================

@dataclass
class ProcessingConfig:
    """Hardware-adaptive batch processing configuration."""
    spacy_batch_size: int = 256
    wtp_batch_size: int = 128
    json_threads: int = 4
    device: str = "cpu"
    pipeline_prefetch: int = 3      # queue depth for parallel pipeline
    pipeline_chunk_size: int = 1000  # rows per chunk


def _detect_optimal_config(use_gpu: bool = True, logger: Optional[logging.Logger] = None) -> ProcessingConfig:
    """Detect hardware and return optimal batch sizes."""
    config = ProcessingConfig()
    log = logger or logging.getLogger(__name__)

    if not use_gpu:
        config.device = "cpu"
        config.spacy_batch_size = 128
        config.wtp_batch_size = 64
        config.json_threads = min(os.cpu_count() or 4, 4)
        config.pipeline_prefetch = 0  # no benefit without GPU
        log.info(f"Processing config: CPU mode (spacy_batch={config.spacy_batch_size}, "
                 f"wtp_batch={config.wtp_batch_size}, threads={config.json_threads})")
        return config

    try:
        import torch
        if torch.cuda.is_available():
            config.device = "cuda"
            vram_gb = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            if vram_gb > 8:
                config.spacy_batch_size = 512
                config.wtp_batch_size = 256
            else:
                config.spacy_batch_size = 256
                config.wtp_batch_size = 128
            config.json_threads = min(os.cpu_count() or 4, 8)
            config.pipeline_prefetch = 3
            log.info(f"Processing config: CUDA ({vram_gb:.1f}GB VRAM, spacy_batch={config.spacy_batch_size}, "
                     f"wtp_batch={config.wtp_batch_size}, threads={config.json_threads})")
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            config.device = "mps"
            config.spacy_batch_size = 256
            config.wtp_batch_size = 128
            config.json_threads = min(os.cpu_count() or 4, 8)
            config.pipeline_prefetch = 3
            log.info(f"Processing config: MPS Apple Silicon (spacy_batch={config.spacy_batch_size}, "
                     f"wtp_batch={config.wtp_batch_size}, threads={config.json_threads})")
        else:
            config.device = "cpu"
            config.spacy_batch_size = 128
            config.wtp_batch_size = 64
            config.json_threads = min(os.cpu_count() or 4, 4)
            config.pipeline_prefetch = 0
            log.info(f"Processing config: CPU (spacy_batch={config.spacy_batch_size}, "
                     f"wtp_batch={config.wtp_batch_size}, threads={config.json_threads})")
    except ImportError:
        config.device = "cpu"
        config.spacy_batch_size = 128
        config.wtp_batch_size = 64
        config.json_threads = min(os.cpu_count() or 4, 4)
        config.pipeline_prefetch = 0
        log.info(f"Processing config: CPU fallback (no torch)")

    return config


# ============================================================================
# Batch processing helper functions
# ============================================================================

def _normalize_language(lang_value, default: str = 'fr') -> str:
    """Normalize language codes to ISO 639-1."""
    import pandas as pd
    if pd.isna(lang_value) or not isinstance(lang_value, str) or len(str(lang_value).strip()) == 0:
        return default
    lang = str(lang_value).strip().lower()
    lang_map = {
        'french': 'fr', 'english': 'en', 'spanish': 'es',
        'german': 'de', 'italian': 'it', 'portuguese': 'pt',
        'fra': 'fr', 'eng': 'en', 'spa': 'es', 'deu': 'de',
    }
    lang = lang_map.get(lang, lang)
    if len(lang) > 2 and lang not in lang_map:
        lang = lang[:2]
    return lang


def _format_sentences_with_ids(sentences: List[str], row_prefix, counter: int):
    """
    Format sentences as [{"id": ..., "text": "..."}] with sequential IDs.

    Args:
        sentences: List of sentence strings
        row_prefix: If not None, IDs are "{row_prefix}_{counter}"; else plain int
        counter: Starting counter value

    Returns:
        (json_string, updated_counter)
    """
    result = []
    c = counter
    for sent in sentences:
        sid = f"{row_prefix}_{c}" if row_prefix is not None else c
        result.append({"id": sid, "text": sent})
        c += 1
    return json.dumps(result, ensure_ascii=False), c


def _merge_lowercase_fragments_static(sentences: List[str]) -> List[str]:
    """Merge sentence fragments that start with lowercase (module-level version)."""
    if not sentences or len(sentences) < 2:
        return sentences

    merged = []
    i = 0
    while i < len(sentences):
        current = sentences[i]
        while i + 1 < len(sentences):
            next_sent = sentences[i + 1].strip()
            if next_sent and next_sent[0].islower():
                current = current.rstrip() + ' ' + next_sent
                i += 1
            else:
                break
        merged.append(current)
        i += 1
    return merged


def _batch_split_sentences(
    wtp_model,
    texts: List[str],
    language: str,
    nlp,
    batch_size: int = 128,
    logger: Optional[logging.Logger] = None,
) -> List[List[str]]:
    """
    Batch sentence splitting. Uses wtpsplit if available, else spaCy/NLTK/regex fallback.

    Returns list of sentence lists, one per input text.
    """
    log = logger or logging.getLogger(__name__)
    results: List[List[str]] = []

    if wtp_model is not None:
        try:
            # wtpsplit v2 batch API: pass list of texts for ~10x speedup
            raw_results = list(wtp_model.split(texts, batch_size=batch_size))
            for sents in raw_results:
                sents = [s.strip() for s in sents if s.strip()]
                sents = _merge_lowercase_fragments_static(sents)
                results.append(sents)
            return results
        except Exception as e:
            log.warning(f"wtpsplit batch failed, falling back: {e}")
            results = []

    # Fallback: spaCy sentencizer
    if nlp is not None:
        try:
            for text in texts:
                doc = nlp(text)
                sents = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
                sents = _merge_lowercase_fragments_static(sents)
                results.append(sents)
            return results
        except Exception:
            results = []

    # Fallback: NLTK
    if HAS_NLTK:
        try:
            nltk_lang = NLTK_LANGUAGES.get(language, 'english')
            for text in texts:
                sents = sent_tokenize(text, language=nltk_lang)
                sents = [s.strip() for s in sents if s.strip()]
                sents = _merge_lowercase_fragments_static(sents)
                results.append(sents)
            return results
        except Exception:
            results = []

    # Fallback: regex
    for text in texts:
        sents = re.split(r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])', text)
        sents = [s.strip() for s in sents if s.strip()]
        sents = _merge_lowercase_fragments_static(sents)
        results.append(sents)
    return results


def _batch_extract_nlp_features(
    nlp,
    texts: List[str],
    nlp_features: 'NLPFeatures',
    batch_size: int = 256,
    progress_callback=None,
) -> List[Tuple[List['TokenInfo'], List['EntityInfo']]]:
    """
    Batch NLP feature extraction using nlp.pipe().

    Returns list of (tokens, entities) tuples, one per input text.
    """
    results: List[Tuple[List[TokenInfo], List[EntityInfo]]] = []

    if nlp is None:
        # Fallback: whitespace tokenization
        for text in texts:
            tokens = [TokenInfo(text=w) for w in text.split()]
            results.append((tokens, []))
            if progress_callback:
                progress_callback(1)
        return results

    processed = 0
    for doc in nlp.pipe(texts, batch_size=batch_size, disable=["parser", "senter"]):
        tokens = []
        entities = []

        for token in doc:
            if not token.is_space:
                info = TokenInfo(text=token.text)
                if nlp_features.lemmatization:
                    info.lemma = token.lemma_
                if nlp_features.pos_tagging:
                    info.pos = token.pos_
                    info.tag = token.tag_
                tokens.append(info)

        if nlp_features.ner:
            for ent in doc.ents:
                entities.append(EntityInfo(
                    text=ent.text,
                    label=ent.label_,
                    start_char=ent.start_char,
                    end_char=ent.end_char,
                ))

        results.append((tokens, entities))
        processed += 1
        if progress_callback and processed % batch_size == 0:
            progress_callback(batch_size)

    # Report remaining progress
    remainder = processed % batch_size
    if progress_callback and remainder > 0:
        progress_callback(remainder)

    return results


def _sequential_fallback(
    indices: List[int],
    texts: List[str],
    language: str,
    segmentation_level: int,
    nlp_features: 'NLPFeatures',
    use_gpu: bool,
    logger: logging.Logger,
    tk_sentences, tk_sentence_count, tk_word_count,
    tk_tokens, tk_lemmas, tk_pos_tags, tk_entities,
    pbar=None,
    prefix_map: Optional[Dict[int, str]] = None,
    counter: Optional[list] = None,
):
    """Fallback: process rows sequentially using TextTokenizer (old path)."""
    tokenizer = TextTokenizer(
        language=language,
        segmentation_level=segmentation_level,
        nlp_features=nlp_features,
        use_gpu=use_gpu,
        logger=logger,
    )
    for pos, orig_idx in enumerate(indices):
        text = texts[pos]
        try:
            result = tokenizer.tokenize(text, document_id=f"row_{orig_idx}")
            if result.success and result.segments:
                all_sentences = []
                all_tokens_text = []
                all_lemmas_list = []
                all_pos_list = []
                all_entities_list = []
                for seg in result.segments:
                    all_sentences.extend(seg.sentences)
                    for t in seg.tokens:
                        all_tokens_text.append(t.text)
                        if t.lemma:
                            all_lemmas_list.append(t.lemma)
                        if t.pos:
                            all_pos_list.append(t.pos)
                    for e in seg.entities:
                        all_entities_list.append({'text': e.text, 'label': e.label})

                row_prefix = prefix_map.get(orig_idx) if prefix_map is not None else None
                current_counter = counter[0] if counter is not None else 1
                s_sentences, new_counter = _format_sentences_with_ids(all_sentences, row_prefix, current_counter)
                if counter is not None:
                    counter[0] = new_counter

                tk_sentences[orig_idx] = s_sentences
                tk_sentence_count[orig_idx] = len(all_sentences)
                tk_word_count[orig_idx] = result.total_words
                tk_tokens[orig_idx] = json.dumps(all_tokens_text, ensure_ascii=False)
                tk_lemmas[orig_idx] = json.dumps(all_lemmas_list, ensure_ascii=False)
                tk_pos_tags[orig_idx] = json.dumps(all_pos_list, ensure_ascii=False)
                tk_entities[orig_idx] = json.dumps(all_entities_list, ensure_ascii=False)
            else:
                tk_word_count[orig_idx] = len(text.split())
        except Exception as e:
            logger.warning(f"Row {orig_idx} sequential fallback error: {e}")
            tk_word_count[orig_idx] = len(text.split()) if text else 0

        if pbar is not None:
            pbar.update(1)


# ============================================================================
# Parallel pipeline infrastructure
# ============================================================================

_SENTINEL = object()  # termination signal for pipeline queues


@dataclass
class _ChunkPayload:
    """Container for data flowing between pipeline stages."""
    chunk_id: int
    chunk_indices: List[int]
    chunk_texts: Optional[List[str]]
    chunk_sentences: Optional[List[List[str]]] = None
    flat_segment_texts: Optional[List[str]] = None
    row_segment_ranges: Optional[List[tuple]] = None
    raw_row_data: Optional[list] = None
    serialized: Optional[list] = None
    error: Optional[str] = None


def _stage_gpu_split(
    payload: _ChunkPayload,
    wtp_model,
    nlp,
    language: str,
    segmentation_level: int,
    wtp_batch_size: int,
    logger: logging.Logger,
) -> _ChunkPayload:
    """Stage A+B: GPU sentence splitting + grouping into segments."""
    try:
        chunk_texts = payload.chunk_texts
        chunk_size = len(chunk_texts)

        # Step A: Batch sentence splitting (GPU-bound via wtpsplit)
        chunk_sentences = _batch_split_sentences(
            wtp_model=wtp_model,
            texts=chunk_texts,
            language=language,
            nlp=nlp,
            batch_size=wtp_batch_size,
            logger=logger,
        )
        payload.chunk_sentences = chunk_sentences

        # Step B: Group sentences into segments, build flat text list
        flat_segment_texts = []
        row_segment_ranges = []

        for row_pos in range(chunk_size):
            sentences = chunk_sentences[row_pos]
            if segmentation_level == 0:
                segments = [(' '.join(sentences), sentences)]
            elif segmentation_level > 0:
                segments = []
                for s_i in range(0, len(sentences), segmentation_level):
                    grp = sentences[s_i:s_i + segmentation_level]
                    segments.append((' '.join(grp), grp))
            else:
                segments = [(' '.join(sentences), sentences)]

            start = len(flat_segment_texts)
            for seg_text, _ in segments:
                flat_segment_texts.append(seg_text)
            row_segment_ranges.append((start, len(flat_segment_texts), segments))

        payload.flat_segment_texts = flat_segment_texts
        payload.row_segment_ranges = row_segment_ranges

        # Free raw texts to reduce memory pressure
        payload.chunk_texts = None

    except Exception as e:
        logger.warning(f"GPU stage error on chunk {payload.chunk_id}: {e}")
        payload.error = str(e)

    return payload


def _stage_spacy_nlp(
    payload: _ChunkPayload,
    nlp,
    nlp_features: 'NLPFeatures',
    spacy_batch_size: int,
    json_threads: int,
) -> _ChunkPayload:
    """Stage C+D+E: spaCy NLP extraction + aggregation + JSON serialization."""
    if payload.error is not None:
        return payload

    try:
        chunk_size = len(payload.chunk_indices)
        flat_segment_texts = payload.flat_segment_texts
        chunk_sentences = payload.chunk_sentences
        row_segment_ranges = payload.row_segment_ranges

        # Step C: Batch NLP feature extraction (CPU-bound via spaCy)
        nlp_results = None
        if nlp_features.any_enabled():
            nlp_results = _batch_extract_nlp_features(
                nlp=nlp,
                texts=flat_segment_texts,
                nlp_features=nlp_features,
                batch_size=spacy_batch_size,
            )

        # Step D: Assemble raw results per row (defer sentence ID serialization to writer)
        raw_row_data = []
        for row_pos in range(chunk_size):
            start, end, segments = row_segment_ranges[row_pos]
            sentences = chunk_sentences[row_pos]

            all_tok_text = []
            all_lem = []
            all_pos_list = []
            all_ent = []
            total_words = 0

            for seg_idx in range(start, end):
                if nlp_results is not None:
                    tokens, entities = nlp_results[seg_idx]
                    for t in tokens:
                        all_tok_text.append(t.text)
                        if t.lemma:
                            all_lem.append(t.lemma)
                        if t.pos:
                            all_pos_list.append(t.pos)
                    for e in entities:
                        all_ent.append({'text': e.text, 'label': e.label})
                    total_words += len(tokens)
                else:
                    seg_text = flat_segment_texts[seg_idx]
                    total_words += len(seg_text.split())

            raw_row_data.append((sentences, all_tok_text, all_lem, all_pos_list, all_ent, total_words))

        # Store raw data for writer thread to serialize with sequential IDs
        payload.raw_row_data = raw_row_data

        # Free intermediate data
        payload.flat_segment_texts = None
        payload.row_segment_ranges = None
        payload.chunk_sentences = None

    except Exception as e:
        payload.error = str(e)

    return payload


def _stage_write_results(
    payload: _ChunkPayload,
    tk_sentences, tk_sentence_count, tk_word_count,
    tk_tokens, tk_lemmas, tk_pos_tags, tk_entities,
    prefix_map: Optional[Dict[int, str]] = None,
    counter: Optional[list] = None,
) -> int:
    """Stage F: Serialize and write results into pre-allocated arrays. Returns rows written."""
    if payload.error is not None or payload.raw_row_data is None:
        return 0

    for row_pos, orig_idx in enumerate(payload.chunk_indices):
        sentences, tok_text, lemmas, pos_list, ent_list, word_count = payload.raw_row_data[row_pos]

        # Serialize sentences with IDs (sequential via counter)
        row_prefix = prefix_map.get(orig_idx) if prefix_map is not None else None
        current_counter = counter[0] if counter is not None else 1
        s_sentences, new_counter = _format_sentences_with_ids(sentences, row_prefix, current_counter)
        if counter is not None:
            counter[0] = new_counter

        tk_sentences[orig_idx] = s_sentences
        tk_sentence_count[orig_idx] = len(sentences)
        tk_word_count[orig_idx] = word_count
        tk_tokens[orig_idx] = json.dumps(tok_text, ensure_ascii=False)
        tk_lemmas[orig_idx] = json.dumps(lemmas, ensure_ascii=False)
        tk_pos_tags[orig_idx] = json.dumps(pos_list, ensure_ascii=False)
        tk_entities[orig_idx] = json.dumps(ent_list, ensure_ascii=False)

    return len(payload.chunk_indices)


def _run_parallel_pipeline(
    payloads: List[_ChunkPayload],
    wtp_model,
    nlp,
    language: str,
    segmentation_level: int,
    nlp_features: 'NLPFeatures',
    config: ProcessingConfig,
    tk_sentences, tk_sentence_count, tk_word_count,
    tk_tokens, tk_lemmas, tk_pos_tags, tk_entities,
    logger: logging.Logger,
    pbar=None,
    prefix_map: Optional[Dict[int, str]] = None,
    counter: Optional[list] = None,
) -> List[_ChunkPayload]:
    """
    Run the 3-thread parallel pipeline: GPU → spaCy → Writer.

    Returns list of failed payloads for sequential retry.
    """
    prefetch = config.pipeline_prefetch
    gpu_to_spacy_q: queue.Queue = queue.Queue(maxsize=prefetch)
    spacy_to_writer_q: queue.Queue = queue.Queue(maxsize=prefetch)

    failed_payloads: List[_ChunkPayload] = []
    failed_lock = threading.Lock()
    thread_errors: List[str] = []

    def gpu_thread():
        """Thread 1: GPU sentence splitting."""
        try:
            for payload in payloads:
                payload = _stage_gpu_split(
                    payload, wtp_model, nlp, language,
                    segmentation_level, config.wtp_batch_size, logger,
                )
                gpu_to_spacy_q.put(payload)
        except Exception as e:
            thread_errors.append(f"GPU thread error: {e}")
            logger.error(f"GPU thread error: {e}")
        finally:
            gpu_to_spacy_q.put(_SENTINEL)

    def spacy_thread():
        """Thread 2: spaCy NLP processing."""
        try:
            while True:
                item = gpu_to_spacy_q.get(timeout=300)
                if item is _SENTINEL:
                    break
                item = _stage_spacy_nlp(
                    item, nlp, nlp_features,
                    config.spacy_batch_size, config.json_threads,
                )
                spacy_to_writer_q.put(item)
        except queue.Empty:
            thread_errors.append("spaCy thread timed out waiting for GPU")
            logger.error("spaCy thread timed out waiting for GPU output")
        except Exception as e:
            thread_errors.append(f"spaCy thread error: {e}")
            logger.error(f"spaCy thread error: {e}")
        finally:
            spacy_to_writer_q.put(_SENTINEL)

    def writer_thread():
        """Thread 3: Write results to arrays."""
        try:
            while True:
                item = spacy_to_writer_q.get(timeout=300)
                if item is _SENTINEL:
                    break
                if item.error is not None:
                    with failed_lock:
                        failed_payloads.append(item)
                    if pbar is not None:
                        pbar.update(len(item.chunk_indices))
                    continue
                written = _stage_write_results(
                    item,
                    tk_sentences, tk_sentence_count, tk_word_count,
                    tk_tokens, tk_lemmas, tk_pos_tags, tk_entities,
                    prefix_map=prefix_map,
                    counter=counter,
                )
                if pbar is not None:
                    pbar.update(written)
        except queue.Empty:
            thread_errors.append("Writer thread timed out waiting for spaCy")
            logger.error("Writer thread timed out waiting for spaCy output")
        except Exception as e:
            thread_errors.append(f"Writer thread error: {e}")
            logger.error(f"Writer thread error: {e}")

    t_gpu = threading.Thread(target=gpu_thread, name="pipeline-gpu", daemon=True)
    t_spacy = threading.Thread(target=spacy_thread, name="pipeline-spacy", daemon=True)
    t_writer = threading.Thread(target=writer_thread, name="pipeline-writer", daemon=True)

    t_gpu.start()
    t_spacy.start()
    t_writer.start()

    t_gpu.join()
    t_spacy.join()
    t_writer.join()

    if thread_errors:
        logger.warning(f"Pipeline completed with {len(thread_errors)} thread error(s)")

    return failed_payloads


class TextTokenizer:
    """
    SOTA multilingual text tokenizer with configurable segmentation and NLP features.

    Usage:
        tokenizer = TextTokenizer(
            language='fr',
            segmentation_level=1,  # 1 sentence per segment
            nlp_features=NLPFeatures(lemmatization=True, ner=True)
        )
        result = tokenizer.tokenize(text, document_id='doc1')
    """

    def __init__(
        self,
        language: str = 'en',
        segmentation_level: int = 1,  # 1, 2, 3, ... sentences per segment (0 = paragraph)
        nlp_features: Optional[NLPFeatures] = None,
        use_gpu: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize tokenizer.

        Args:
            language: ISO 639-1 language code
            segmentation_level: Number of sentences per segment (0 = paragraph mode)
            nlp_features: Which NLP features to extract
            use_gpu: Use GPU if available
            logger: Optional logger
        """
        self.language = language
        self.segmentation_level = segmentation_level
        self.nlp_features = nlp_features or NLPFeatures()
        self.use_gpu = use_gpu
        self.logger = logger or logging.getLogger(__name__)

        # Models
        self.wtp = None
        self.nlp = None  # spaCy
        self.stanza_nlp = None

        self._init_models()

    def _init_models(self):
        """Initialize required models."""
        global _wtp_model, _spacy_cache, _stanza_cache

        # Initialize wtpsplit for sentence segmentation
        if HAS_WTPSPLIT:
            try:
                if _wtp_model is None:
                    self.logger.info("Loading wtpsplit SaT model...")
                    _wtp_model = SaT("sat-12l")

                    if self.use_gpu:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                _wtp_model.half().to("cuda")
                            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                _wtp_model.to("mps")
                        except Exception:
                            pass

                self.wtp = _wtp_model
            except Exception as e:
                self.logger.warning(f"Could not load wtpsplit: {e}")

        # Initialize spaCy for NLP features
        if self.nlp_features.any_enabled() and HAS_SPACY:
            self._load_spacy_model()

        # Fallback to stanza if spaCy not available
        if self.nlp_features.any_enabled() and self.nlp is None and HAS_STANZA:
            self._load_stanza_model()

    def _load_spacy_model(self):
        """Load best available spaCy model for language."""
        global _spacy_cache

        if self.language in _spacy_cache:
            self.nlp = _spacy_cache[self.language]
            return

        models_to_try = SPACY_MODELS.get(
            self.language,
            [f'{self.language}_core_news_sm']
        )

        for model_name in models_to_try:
            try:
                self.nlp = spacy.load(model_name, disable=["parser", "senter"])
                _spacy_cache[self.language] = self.nlp
                self.logger.info(f"Loaded spaCy model: {model_name}")
                return
            except OSError:
                continue

        # Fallback: blank model with sentencizer
        try:
            self.nlp = spacy.blank(self.language)
            self.nlp.add_pipe('sentencizer')
            _spacy_cache[self.language] = self.nlp
            self.logger.info(f"Using blank spaCy model for {self.language}")
        except Exception as e:
            self.logger.warning(f"Could not create spaCy model: {e}")

    def _load_stanza_model(self):
        """Load stanza model as fallback."""
        global _stanza_cache

        if self.language in _stanza_cache:
            self.stanza_nlp = _stanza_cache[self.language]
            return

        try:
            stanza.download(self.language, processors='tokenize,pos,lemma,ner', verbose=False)
            self.stanza_nlp = stanza.Pipeline(
                self.language,
                processors='tokenize,pos,lemma,ner',
                use_gpu=self.use_gpu,
                verbose=False
            )
            _stanza_cache[self.language] = self.stanza_nlp
            self.logger.info(f"Loaded stanza model for {self.language}")
        except Exception as e:
            self.logger.warning(f"Could not load stanza: {e}")

    def tokenize(
        self,
        text: str,
        document_id: str = "doc_1",
        source_file: Optional[str] = None
    ) -> TokenizationResult:
        """
        Tokenize text with configured segmentation and NLP features.

        Args:
            text: Input text
            document_id: Identifier for the document
            source_file: Optional source filename

        Returns:
            TokenizationResult with segments
        """
        if not text or not text.strip():
            return TokenizationResult(
                success=False,
                document_id=document_id,
                error="Empty text provided"
            )

        text = text.strip()

        try:
            # Step 1: Split into sentences
            sentences = self._split_sentences(text)

            # Step 2: Group sentences based on segmentation level
            if self.segmentation_level == 0:
                # Paragraph mode
                grouped = self._split_paragraphs(text)
            else:
                grouped = self._group_sentences(sentences, self.segmentation_level)

            # Step 3: Process each group with NLP features
            segments = []
            total_entities = 0

            for idx, (segment_text, segment_sentences) in enumerate(grouped, start=1):
                segment = TokenizedSegment(
                    segment_id=idx,
                    document_id=document_id,
                    text=segment_text,
                    sentences=segment_sentences
                )

                # Extract NLP features if enabled
                if self.nlp_features.any_enabled():
                    tokens, entities = self._extract_nlp_features(segment_text)
                    segment.tokens = tokens
                    segment.entities = entities
                    total_entities += len(entities)

                segments.append(segment)

            total_words = sum(seg.word_count for seg in segments)

            return TokenizationResult(
                success=True,
                segments=segments,
                document_id=document_id,
                source_file=source_file,
                language=self.language,
                segmentation_level=self.segmentation_level,
                nlp_features=self.nlp_features,
                total_sentences=len(sentences),
                total_words=total_words,
                total_entities=total_entities,
                processing_info={
                    'sentence_segmenter': 'wtpsplit' if self.wtp else 'spacy' if self.nlp else 'nltk',
                    'nlp_model': self.nlp.meta['name'] if self.nlp and hasattr(self.nlp, 'meta') else 'stanza' if self.stanza_nlp else 'none'
                }
            )

        except Exception as e:
            self.logger.error(f"Tokenization error: {e}")
            return TokenizationResult(
                success=False,
                document_id=document_id,
                source_file=source_file,
                error=str(e)
            )

    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using best available method."""
        sentences = None

        # Method 1: wtpsplit (SOTA)
        if self.wtp is not None:
            try:
                sentences = self.wtp.split(text)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception:
                sentences = None

        # Method 2: spaCy
        if not sentences and self.nlp:
            try:
                doc = self.nlp(text)
                sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
            except Exception:
                sentences = None

        # Method 3: stanza
        if not sentences and self.stanza_nlp:
            try:
                doc = self.stanza_nlp(text)
                sentences = [sent.text.strip() for sent in doc.sentences if sent.text.strip()]
            except Exception:
                sentences = None

        # Method 4: NLTK
        if not sentences and HAS_NLTK:
            try:
                nltk_lang = NLTK_LANGUAGES.get(self.language, 'english')
                sentences = sent_tokenize(text, language=nltk_lang)
                sentences = [s.strip() for s in sentences if s.strip()]
            except Exception:
                sentences = None

        # Method 5: Regex fallback
        if not sentences:
            sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z\u00C0-\u024F])', text)
            sentences = [s.strip() for s in sentences if s.strip()]

        # Post-process: merge fragments starting with lowercase
        return self._merge_lowercase_fragments(sentences)

    def _merge_lowercase_fragments(self, sentences: List[str]) -> List[str]:
        """Merge sentence fragments that start with lowercase."""
        return _merge_lowercase_fragments_static(sentences)

    def _split_paragraphs(self, text: str) -> List[tuple]:
        """Split text by paragraphs (double newlines)."""
        paragraphs = re.split(r'\n\s*\n', text)
        result = []

        for para in paragraphs:
            para = para.strip()
            if para:
                sentences = self._split_sentences(para)
                result.append((para, sentences))

        return result

    def _group_sentences(self, sentences: List[str], group_size: int) -> List[tuple]:
        """Group sentences into segments of specified size."""
        if group_size <= 0:
            group_size = 1

        result = []
        for i in range(0, len(sentences), group_size):
            group = sentences[i:i + group_size]
            text = ' '.join(group)
            result.append((text, group))

        return result

    def _extract_nlp_features(self, text: str) -> tuple:
        """Extract tokens and entities using spaCy or stanza."""
        tokens = []
        entities = []

        if self.nlp:
            doc = self.nlp(text)

            # Extract tokens
            for token in doc:
                if not token.is_space:
                    info = TokenInfo(text=token.text)
                    if self.nlp_features.lemmatization:
                        info.lemma = token.lemma_
                    if self.nlp_features.pos_tagging:
                        info.pos = token.pos_
                        info.tag = token.tag_
                    tokens.append(info)

            # Extract entities
            if self.nlp_features.ner:
                for ent in doc.ents:
                    entities.append(EntityInfo(
                        text=ent.text,
                        label=ent.label_,
                        start_char=ent.start_char,
                        end_char=ent.end_char
                    ))

        elif self.stanza_nlp:
            doc = self.stanza_nlp(text)

            for sent in doc.sentences:
                for word in sent.words:
                    info = TokenInfo(text=word.text)
                    if self.nlp_features.lemmatization:
                        info.lemma = word.lemma
                    if self.nlp_features.pos_tagging:
                        info.pos = word.upos
                        info.tag = word.xpos
                    tokens.append(info)

                # Stanza NER
                if self.nlp_features.ner and hasattr(sent, 'ents'):
                    for ent in sent.ents:
                        entities.append(EntityInfo(
                            text=ent.text,
                            label=ent.type,
                            start_char=ent.start_char,
                            end_char=ent.end_char
                        ))

        else:
            # Fallback: just word tokenization
            for word in text.split():
                tokens.append(TokenInfo(text=word))

        return tokens, entities


# ============================================================================
# Output formatters
# ============================================================================

def format_tokenization_csv(
    results: List[TokenizationResult],
    include_source: bool = True
) -> str:
    """
    Format tokenization results as CSV with JSON-encoded structured data.

    Columns:
    - segment_id: Segment number within document
    - document_id: Document identifier
    - text: Raw text of the segment
    - sentences: JSON array of sentences
    - tokens: JSON array of token objects (if NLP features enabled)
    - entities: JSON array of entity objects (if NER enabled)
    - sentence_count: Number of sentences
    - word_count: Number of words
    - source_file: Source filename (optional)
    - language: Detected/specified language
    """
    output = io.StringIO()

    # Determine columns based on features used
    base_columns = ['segment_id', 'document_id', 'text', 'sentences', 'sentence_count', 'word_count']

    # Check for NLP features
    has_tokens = any(
        any(seg.tokens for seg in r.segments)
        for r in results if r.success
    )
    has_entities = any(
        any(seg.entities for seg in r.segments)
        for r in results if r.success
    )

    columns = base_columns.copy()
    if has_tokens:
        columns.extend(['tokens', 'lemmas', 'pos_tags'])
    if has_entities:
        columns.append('entities')
    if include_source:
        columns.extend(['source_file', 'language'])

    writer = csv.DictWriter(output, fieldnames=columns, quoting=csv.QUOTE_ALL)
    writer.writeheader()

    global_counter = [1]

    for result in results:
        if not result.success:
            continue

        for segment in result.segments:
            sentences_json, new_counter = _format_sentences_with_ids(
                segment.sentences, None, global_counter[0]
            )
            global_counter[0] = new_counter

            row = {
                'segment_id': segment.segment_id,
                'document_id': segment.document_id,
                'text': segment.text,
                'sentences': sentences_json,
                'sentence_count': segment.sentence_count,
                'word_count': segment.word_count,
            }

            if has_tokens:
                row['tokens'] = segment.tokens_to_json()
                row['lemmas'] = segment.lemmas_to_json()
                row['pos_tags'] = segment.pos_to_json()

            if has_entities:
                row['entities'] = segment.entities_to_json()

            if include_source:
                row['source_file'] = result.source_file or ''
                row['language'] = result.language

            writer.writerow(row)

    return output.getvalue()


def format_tokenization_json(results: List[TokenizationResult]) -> str:
    """Format tokenization results as JSON."""
    output = []
    global_counter = [1]

    for result in results:
        if not result.success:
            continue

        doc_data = {
            'document_id': result.document_id,
            'source_file': result.source_file,
            'language': result.language,
            'segmentation_level': result.segmentation_level,
            'nlp_features': {
                'lemmatization': result.nlp_features.lemmatization,
                'pos_tagging': result.nlp_features.pos_tagging,
                'ner': result.nlp_features.ner
            },
            'statistics': {
                'total_segments': len(result.segments),
                'total_sentences': result.total_sentences,
                'total_words': result.total_words,
                'total_entities': result.total_entities
            },
            'processing_info': result.processing_info,
            'segments': []
        }

        for seg in result.segments:
            # Build sentence objects with IDs
            sentences_with_ids = []
            for sent in seg.sentences:
                sentences_with_ids.append({"id": global_counter[0], "text": sent})
                global_counter[0] += 1

            seg_data = {
                'segment_id': seg.segment_id,
                'text': seg.text,
                'sentences': sentences_with_ids,
                'sentence_count': seg.sentence_count,
                'word_count': seg.word_count,
            }

            if seg.tokens:
                seg_data['tokens'] = [t.to_dict() for t in seg.tokens]

            if seg.entities:
                seg_data['entities'] = [e.to_dict() for e in seg.entities]

            doc_data['segments'].append(seg_data)

        output.append(doc_data)

    return json.dumps(output, ensure_ascii=False, indent=2)


# ============================================================================
# DataFrame tokenization (for RDS / tabular data)
# ============================================================================

def tokenize_dataframe(
    df,
    text_column: str,
    lang_column: Optional[str] = None,
    default_language: str = 'fr',
    nlp_features: Optional[NLPFeatures] = None,
    segmentation_level: int = 1,
    use_gpu: bool = True,
    logger: Optional[logging.Logger] = None,
    show_progress: bool = True,
    sentence_id_prefix: Optional[Dict[int, str]] = None,
):
    """
    Tokenize a DataFrame with batched processing grouped by language.

    Uses nlp.pipe() for batched spaCy processing, language grouping to minimize
    model reloads, and ThreadPoolExecutor for parallel JSON serialization.

    Args:
        df: pandas DataFrame
        text_column: Name of the column containing text
        lang_column: Optional column with per-row language codes
        default_language: Fallback language if no lang_column or value is missing
        nlp_features: NLP features to extract
        segmentation_level: Sentences per segment (1 = sentence-level)
        use_gpu: Use GPU if available
        logger: Optional logger
        show_progress: Show tqdm progress bar
        sentence_id_prefix: Optional dict mapping row index → prefix string for sentence IDs.
            If None, sentences get auto-incremented integer IDs (1, 2, 3...).
            If provided, sentences get "{prefix}_{seq}" IDs for rows in the map.

    Returns:
        DataFrame with added tk_* columns
    """
    import pandas as pd
    from tqdm import tqdm

    if nlp_features is None:
        nlp_features = NLPFeatures.all_features()

    if logger is None:
        logger = logging.getLogger(__name__)

    n_rows = len(df)

    # 1. Detect optimal hardware config
    config = _detect_optimal_config(use_gpu=use_gpu, logger=logger)

    # 2. Normalize languages (vectorized)
    texts_series = df[text_column]
    if lang_column and lang_column in df.columns:
        languages = df[lang_column].apply(lambda x: _normalize_language(x, default_language))
    else:
        languages = pd.Series([default_language] * n_rows, index=df.index)

    # 3. Identify valid (non-empty) rows
    valid_mask = texts_series.notna() & texts_series.astype(str).str.strip().astype(bool)

    # 4. Pre-allocate result arrays with defaults
    tk_sentences = ['[]'] * n_rows
    tk_sentence_count = [0] * n_rows
    tk_word_count = [0] * n_rows
    tk_tokens = ['[]'] * n_rows
    tk_lemmas = ['[]'] * n_rows
    tk_pos_tags = ['[]'] * n_rows
    tk_entities = ['[]'] * n_rows

    # 4b. Global sentence ID counter (mutable list for thread-safe pass-by-reference)
    global_counter = [1]

    # 5. Group valid rows by language
    lang_groups: Dict[str, List[Tuple[int, int]]] = defaultdict(list)
    # (position_in_group, original_df_index)
    valid_indices = valid_mask[valid_mask].index.tolist()
    for orig_idx in valid_indices:
        lang = languages.iloc[orig_idx] if orig_idx < len(languages) else default_language
        lang_groups[lang].append(orig_idx)

    n_valid = len(valid_indices)
    logger.info(f"Batched tokenization: {n_valid:,} valid rows, {n_rows - n_valid:,} empty/NaN, "
                f"{len(lang_groups)} language(s)")

    # 6. Single progress bar
    pbar = None
    if show_progress:
        pbar = tqdm(
            total=n_valid,
            desc="Tokenizing",
            unit="row",
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )

    # 7. Process each language group
    global _wtp_model, _spacy_cache

    # Chunk size for incremental processing within a language group
    CHUNK_SIZE = 1000

    for lang, orig_indices in lang_groups.items():
        group_size = len(orig_indices)
        logger.info(f"Processing language '{lang}': {group_size:,} rows")

        # --- Load models once for this language ---
        wtp = None
        if HAS_WTPSPLIT:
            try:
                if _wtp_model is None:
                    logger.info("Loading wtpsplit SaT model...")
                    _wtp_model = SaT("sat-12l")
                    if use_gpu:
                        try:
                            import torch
                            if torch.cuda.is_available():
                                _wtp_model.half().to("cuda")
                            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                                _wtp_model.to("mps")
                        except Exception:
                            pass
                wtp = _wtp_model
            except Exception as e:
                logger.warning(f"Could not load wtpsplit: {e}")

        nlp = None
        if nlp_features.any_enabled() and HAS_SPACY:
            if lang in _spacy_cache:
                nlp = _spacy_cache[lang]
            else:
                models_to_try = SPACY_MODELS.get(lang, [f'{lang}_core_news_sm'])
                for model_name in models_to_try:
                    try:
                        nlp = spacy.load(model_name, disable=["parser", "senter"])
                        _spacy_cache[lang] = nlp
                        logger.info(f"Loaded spaCy model: {model_name}")
                        break
                    except OSError:
                        continue
                if nlp is None:
                    try:
                        nlp = spacy.blank(lang)
                        nlp.add_pipe('sentencizer')
                        _spacy_cache[lang] = nlp
                        logger.info(f"Using blank spaCy model for {lang}")
                    except Exception as e:
                        logger.warning(f"Could not create spaCy model for {lang}: {e}")

        # --- Build chunk payloads ---
        chunk_payloads = []
        for chunk_id, chunk_start in enumerate(range(0, group_size, CHUNK_SIZE)):
            chunk_end = min(chunk_start + CHUNK_SIZE, group_size)
            chunk_indices = orig_indices[chunk_start:chunk_end]
            chunk_texts = [str(texts_series.iloc[idx]).strip() for idx in chunk_indices]
            chunk_payloads.append(_ChunkPayload(
                chunk_id=chunk_id,
                chunk_indices=chunk_indices,
                chunk_texts=chunk_texts,
            ))

        # --- Parallel pipeline (GPU+CPU overlap) or sequential ---
        use_parallel = config.pipeline_prefetch > 0 and wtp is not None
        if use_parallel:
            logger.info(f"Using parallel GPU+CPU pipeline (prefetch={config.pipeline_prefetch}, "
                        f"{len(chunk_payloads)} chunks)")
            failed = _run_parallel_pipeline(
                payloads=chunk_payloads,
                wtp_model=wtp,
                nlp=nlp,
                language=lang,
                segmentation_level=segmentation_level,
                nlp_features=nlp_features,
                config=config,
                tk_sentences=tk_sentences,
                tk_sentence_count=tk_sentence_count,
                tk_word_count=tk_word_count,
                tk_tokens=tk_tokens,
                tk_lemmas=tk_lemmas,
                tk_pos_tags=tk_pos_tags,
                tk_entities=tk_entities,
                logger=logger,
                pbar=pbar,
                prefix_map=sentence_id_prefix,
                counter=global_counter,
            )

            # Retry failed chunks sequentially
            if failed:
                logger.warning(f"{len(failed)} chunk(s) failed in pipeline, retrying sequentially")
                for fp in failed:
                    retry_texts = [str(texts_series.iloc[idx]).strip() for idx in fp.chunk_indices]
                    _sequential_fallback(
                        indices=fp.chunk_indices,
                        texts=retry_texts,
                        language=lang,
                        segmentation_level=segmentation_level,
                        nlp_features=nlp_features,
                        use_gpu=use_gpu,
                        logger=logger,
                        tk_sentences=tk_sentences,
                        tk_sentence_count=tk_sentence_count,
                        tk_word_count=tk_word_count,
                        tk_tokens=tk_tokens,
                        tk_lemmas=tk_lemmas,
                        tk_pos_tags=tk_pos_tags,
                        tk_entities=tk_entities,
                        prefix_map=sentence_id_prefix,
                        counter=global_counter,
                    )
        else:
            # Sequential path (CPU-only or no wtpsplit)
            for payload in chunk_payloads:
                chunk_indices = payload.chunk_indices
                chunk_texts = payload.chunk_texts
                chunk_size = len(chunk_indices)

                try:
                    # Step A: Batch sentence splitting
                    chunk_sentences = _batch_split_sentences(
                        wtp_model=wtp,
                        texts=chunk_texts,
                        language=lang,
                        nlp=nlp,
                        batch_size=config.wtp_batch_size,
                        logger=logger,
                    )

                    # Step B: Group sentences into segments
                    flat_segment_texts = []
                    row_segment_ranges = []

                    for row_pos in range(chunk_size):
                        sentences = chunk_sentences[row_pos]
                        if segmentation_level == 0:
                            segments = [(' '.join(sentences), sentences)]
                        elif segmentation_level > 0:
                            segments = []
                            for s_i in range(0, len(sentences), segmentation_level):
                                grp = sentences[s_i:s_i + segmentation_level]
                                segments.append((' '.join(grp), grp))
                        else:
                            segments = [(' '.join(sentences), sentences)]

                        start = len(flat_segment_texts)
                        for seg_text, _ in segments:
                            flat_segment_texts.append(seg_text)
                        row_segment_ranges.append((start, len(flat_segment_texts), segments))

                    # Step C: Batch NLP feature extraction
                    nlp_results = None
                    if nlp_features.any_enabled():
                        nlp_results = _batch_extract_nlp_features(
                            nlp=nlp,
                            texts=flat_segment_texts,
                            nlp_features=nlp_features,
                            batch_size=config.spacy_batch_size,
                        )

                    # Step D: Assemble raw results per row
                    raw_row_data = []
                    for row_pos in range(chunk_size):
                        start, end, segments = row_segment_ranges[row_pos]
                        sentences = chunk_sentences[row_pos]

                        all_tok_text = []
                        all_lem = []
                        all_pos_list = []
                        all_ent = []
                        total_words = 0

                        for seg_idx in range(start, end):
                            if nlp_results is not None:
                                tokens, entities = nlp_results[seg_idx]
                                for t in tokens:
                                    all_tok_text.append(t.text)
                                    if t.lemma:
                                        all_lem.append(t.lemma)
                                    if t.pos:
                                        all_pos_list.append(t.pos)
                                for e in entities:
                                    all_ent.append({'text': e.text, 'label': e.label})
                                total_words += len(tokens)
                            else:
                                seg_text = flat_segment_texts[seg_idx]
                                total_words += len(seg_text.split())

                        raw_row_data.append((sentences, all_tok_text, all_lem, all_pos_list, all_ent, total_words))

                    # Step E+F: Serialize with sentence IDs and write into arrays
                    for row_pos, orig_idx in enumerate(chunk_indices):
                        sentences, tok_text, lemmas, pos_list, ent_list, word_count = raw_row_data[row_pos]

                        row_prefix = sentence_id_prefix.get(orig_idx) if sentence_id_prefix is not None else None
                        s_sentences, new_counter = _format_sentences_with_ids(
                            sentences, row_prefix, global_counter[0]
                        )
                        global_counter[0] = new_counter

                        tk_sentences[orig_idx] = s_sentences
                        tk_sentence_count[orig_idx] = len(sentences)
                        tk_word_count[orig_idx] = word_count
                        tk_tokens[orig_idx] = json.dumps(tok_text, ensure_ascii=False)
                        tk_lemmas[orig_idx] = json.dumps(lemmas, ensure_ascii=False)
                        tk_pos_tags[orig_idx] = json.dumps(pos_list, ensure_ascii=False)
                        tk_entities[orig_idx] = json.dumps(ent_list, ensure_ascii=False)

                    if pbar is not None:
                        pbar.update(chunk_size)

                except Exception as e:
                    logger.warning(f"Batch chunk failed for '{lang}' (chunk {payload.chunk_id}): {e}. "
                                   f"Falling back to sequential.")
                    _sequential_fallback(
                        indices=chunk_indices,
                        texts=chunk_texts,
                        language=lang,
                        segmentation_level=segmentation_level,
                        nlp_features=nlp_features,
                        use_gpu=use_gpu,
                        logger=logger,
                        tk_sentences=tk_sentences,
                        tk_sentence_count=tk_sentence_count,
                        tk_word_count=tk_word_count,
                        tk_tokens=tk_tokens,
                        tk_lemmas=tk_lemmas,
                        tk_pos_tags=tk_pos_tags,
                        tk_entities=tk_entities,
                        pbar=pbar,
                        prefix_map=sentence_id_prefix,
                        counter=global_counter,
                    )

    if pbar is not None:
        pbar.close()

    # 8. Build result DataFrame
    result_df = df.copy()
    result_df['tk_sentences'] = tk_sentences
    result_df['tk_sentence_count'] = tk_sentence_count
    result_df['tk_word_count'] = tk_word_count
    result_df['tk_tokens'] = tk_tokens
    result_df['tk_lemmas'] = tk_lemmas
    result_df['tk_pos_tags'] = tk_pos_tags
    result_df['tk_entities'] = tk_entities

    return result_df


def format_tokenization_rds(df, output_path: str):
    """
    Write an enriched DataFrame to RDS format.

    Args:
        df: pandas DataFrame with tk_* columns
        output_path: Path to write the .rds file
    """
    import pyreadr
    pyreadr.write_rds(str(output_path), df)


# ============================================================================
# Convenience functions
# ============================================================================

def tokenize_text(
    text: str,
    language: str = 'en',
    segmentation_level: int = 1,
    lemmatization: bool = False,
    pos_tagging: bool = False,
    ner: bool = False,
    document_id: str = 'doc_1'
) -> TokenizationResult:
    """
    Convenience function to tokenize text.

    Args:
        text: Input text
        language: ISO 639-1 language code
        segmentation_level: Sentences per segment (0 = paragraph)
        lemmatization: Extract lemmas
        pos_tagging: Extract POS tags
        ner: Extract named entities
        document_id: Document identifier

    Returns:
        TokenizationResult
    """
    features = NLPFeatures(
        lemmatization=lemmatization,
        pos_tagging=pos_tagging,
        ner=ner
    )

    tokenizer = TextTokenizer(
        language=language,
        segmentation_level=segmentation_level,
        nlp_features=features
    )

    return tokenizer.tokenize(text, document_id=document_id)


def get_available_nlp_features() -> List[Dict[str, str]]:
    """Get list of available NLP features."""
    return [
        {
            'id': 'lemmatization',
            'name': 'Lemmatization',
            'description': 'Extract word base forms (lemmas)'
        },
        {
            'id': 'pos_tagging',
            'name': 'POS Tagging',
            'description': 'Part-of-speech tags for each word'
        },
        {
            'id': 'ner',
            'name': 'Named Entity Recognition',
            'description': 'Extract named entities (people, places, organizations)'
        }
    ]


def get_supported_languages() -> List[Dict[str, str]]:
    """Get list of languages with good NLP support."""
    return [
        {'code': 'en', 'name': 'English', 'spacy_model': 'en_core_web_trf'},
        {'code': 'fr', 'name': 'French', 'spacy_model': 'fr_dep_news_trf'},
        {'code': 'de', 'name': 'German', 'spacy_model': 'de_dep_news_trf'},
        {'code': 'es', 'name': 'Spanish', 'spacy_model': 'es_dep_news_trf'},
        {'code': 'it', 'name': 'Italian', 'spacy_model': 'it_core_news_lg'},
        {'code': 'pt', 'name': 'Portuguese', 'spacy_model': 'pt_core_news_lg'},
        {'code': 'nl', 'name': 'Dutch', 'spacy_model': 'nl_core_news_lg'},
        {'code': 'pl', 'name': 'Polish', 'spacy_model': 'pl_core_news_lg'},
        {'code': 'ru', 'name': 'Russian', 'spacy_model': 'ru_core_news_lg'},
        {'code': 'ja', 'name': 'Japanese', 'spacy_model': 'ja_core_news_trf'},
        {'code': 'zh', 'name': 'Chinese', 'spacy_model': 'zh_core_web_trf'},
        {'code': 'ko', 'name': 'Korean', 'spacy_model': 'ko_core_news_lg'},
        {'code': 'da', 'name': 'Danish', 'spacy_model': 'da_core_news_lg'},
        {'code': 'sv', 'name': 'Swedish', 'spacy_model': 'sv_core_news_lg'},
        {'code': 'nb', 'name': 'Norwegian', 'spacy_model': 'nb_core_news_lg'},
        {'code': 'fi', 'name': 'Finnish', 'spacy_model': 'fi_core_news_lg'},
        {'code': 'el', 'name': 'Greek', 'spacy_model': 'el_core_news_lg'},
        {'code': 'ro', 'name': 'Romanian', 'spacy_model': 'ro_core_news_lg'},
        {'code': 'uk', 'name': 'Ukrainian', 'spacy_model': 'uk_core_news_lg'},
        {'code': 'ca', 'name': 'Catalan', 'spacy_model': 'ca_core_news_lg'},
        {'code': 'hr', 'name': 'Croatian', 'spacy_model': 'hr_core_news_lg'},
        {'code': 'lt', 'name': 'Lithuanian', 'spacy_model': 'lt_core_news_lg'},
        {'code': 'mk', 'name': 'Macedonian', 'spacy_model': 'mk_core_news_lg'},
    ]
