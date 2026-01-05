"""
Language detection module for Transcribe Tool.

Provides automatic language detection with multiple fallback methods
for accurate identification of transcription language.
"""

import logging
from typing import Dict, Any, List, Optional
from enum import Enum

# Primary: lingua-language-detector (most accurate)
try:
    from lingua import Language, LanguageDetectorBuilder
    HAS_LINGUA = True
except ImportError:
    HAS_LINGUA = False
    Language = LanguageDetectorBuilder = None

# Fallback: langid
try:
    import langid
    HAS_LANGID = True
except ImportError:
    HAS_LANGID = False


class DetectionMethod(Enum):
    """Available language detection methods."""
    LINGUA = "lingua"
    LANGID = "langid"
    ENSEMBLE = "ensemble"


# ISO 639-1 language codes with full names
LANGUAGE_CODES = {
    'en': 'English', 'fr': 'French', 'es': 'Spanish', 'de': 'German',
    'it': 'Italian', 'pt': 'Portuguese', 'nl': 'Dutch', 'pl': 'Polish',
    'ru': 'Russian', 'ja': 'Japanese', 'ko': 'Korean', 'zh': 'Chinese',
    'ar': 'Arabic', 'hi': 'Hindi', 'tr': 'Turkish', 'vi': 'Vietnamese',
    'id': 'Indonesian', 'th': 'Thai', 'sv': 'Swedish', 'no': 'Norwegian',
    'da': 'Danish', 'fi': 'Finnish', 'cs': 'Czech', 'el': 'Greek',
    'he': 'Hebrew', 'uk': 'Ukrainian', 'ro': 'Romanian', 'hu': 'Hungarian',
    'bg': 'Bulgarian', 'hr': 'Croatian', 'sk': 'Slovak', 'sl': 'Slovenian',
    'lt': 'Lithuanian', 'lv': 'Latvian', 'et': 'Estonian', 'ca': 'Catalan',
    'eu': 'Basque', 'gl': 'Galician', 'fa': 'Persian', 'ur': 'Urdu',
    'bn': 'Bengali', 'ta': 'Tamil', 'te': 'Telugu', 'ml': 'Malayalam',
    'ms': 'Malay', 'sr': 'Serbian'
}

# Lingua language enum mapping
LINGUA_TO_ISO = {
    'ENGLISH': 'en', 'FRENCH': 'fr', 'SPANISH': 'es', 'GERMAN': 'de',
    'ITALIAN': 'it', 'PORTUGUESE': 'pt', 'DUTCH': 'nl', 'RUSSIAN': 'ru',
    'CHINESE': 'zh', 'JAPANESE': 'ja', 'ARABIC': 'ar', 'POLISH': 'pl',
    'TURKISH': 'tr', 'KOREAN': 'ko', 'HINDI': 'hi', 'SWEDISH': 'sv',
    'BOKMAL': 'no', 'DANISH': 'da', 'FINNISH': 'fi', 'CZECH': 'cs',
    'GREEK': 'el', 'HEBREW': 'he', 'ROMANIAN': 'ro', 'UKRAINIAN': 'uk',
    'BULGARIAN': 'bg', 'CROATIAN': 'hr', 'VIETNAMESE': 'vi', 'THAI': 'th',
    'INDONESIAN': 'id', 'PERSIAN': 'fa', 'HUNGARIAN': 'hu', 'SLOVAK': 'sk',
    'SLOVENIAN': 'sl', 'LITHUANIAN': 'lt', 'LATVIAN': 'lv', 'ESTONIAN': 'et',
    'CATALAN': 'ca', 'BASQUE': 'eu', 'GALICIAN': 'gl', 'MALAY': 'ms',
    'SERBIAN': 'sr', 'BENGALI': 'bn', 'TAMIL': 'ta', 'TELUGU': 'te',
    'MARATHI': 'mr', 'GUJARATI': 'gu', 'PUNJABI': 'pa', 'URDU': 'ur'
}


class LanguageDetector:
    """Multi-method language detection with fallback support."""

    def __init__(
        self,
        method: DetectionMethod = DetectionMethod.LINGUA,
        confidence_threshold: float = 0.7,
        fallback_language: str = 'en'
    ):
        """
        Initialize the language detector.

        Args:
            method: Detection method to use
            confidence_threshold: Minimum confidence for detection
            fallback_language: Language to use when detection fails
        """
        self.method = method
        self.confidence_threshold = confidence_threshold
        self.fallback_language = fallback_language
        self.logger = logging.getLogger(__name__)

        # Initialize lingua detector
        self.lingua_detector = None
        if HAS_LINGUA:
            try:
                languages = [
                    Language.ENGLISH, Language.FRENCH, Language.SPANISH,
                    Language.GERMAN, Language.ITALIAN, Language.PORTUGUESE,
                    Language.DUTCH, Language.RUSSIAN, Language.CHINESE,
                    Language.JAPANESE, Language.ARABIC, Language.POLISH,
                    Language.TURKISH, Language.KOREAN, Language.HINDI,
                    Language.SWEDISH, Language.BOKMAL, Language.DANISH,
                    Language.FINNISH, Language.CZECH, Language.GREEK,
                    Language.HEBREW, Language.ROMANIAN, Language.UKRAINIAN,
                    Language.BULGARIAN, Language.CROATIAN, Language.VIETNAMESE,
                    Language.THAI, Language.INDONESIAN, Language.PERSIAN
                ]
                self.lingua_detector = LanguageDetectorBuilder.from_languages(*languages).build()
            except Exception as e:
                self.logger.warning(f"Could not initialize Lingua detector: {e}")

        # Check availability
        self._check_availability()

    def _check_availability(self):
        """Check which detection methods are available."""
        available = []
        if HAS_LINGUA and self.lingua_detector:
            available.append("lingua")
        if HAS_LANGID:
            available.append("langid")

        if not available:
            self.logger.warning(
                "No language detection libraries found. "
                "Install with: pip install lingua-language-detector"
            )
            self.method = None
        else:
            # Fallback if preferred method not available
            if self.method == DetectionMethod.LINGUA and not HAS_LINGUA:
                self.method = DetectionMethod.LANGID if HAS_LANGID else None

    def detect(self, text: str, return_all_scores: bool = False) -> Dict[str, Any]:
        """
        Detect language of text.

        Args:
            text: Text to analyze
            return_all_scores: Return scores for all detected languages

        Returns:
            Dictionary with language code, confidence, and method used
        """
        if not text or not text.strip():
            return {
                'language': self.fallback_language,
                'language_name': LANGUAGE_CODES.get(self.fallback_language, 'Unknown'),
                'confidence': 0.0,
                'method': 'fallback',
                'all_scores': []
            }

        text = text.strip()

        # Use specified method
        if self.method == DetectionMethod.LINGUA:
            result = self._detect_lingua(text, return_all_scores)
        elif self.method == DetectionMethod.LANGID:
            result = self._detect_langid(text, return_all_scores)
        elif self.method == DetectionMethod.ENSEMBLE:
            result = self._detect_ensemble(text, return_all_scores)
        else:
            result = {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'fallback',
                'all_scores': []
            }

        # Apply confidence threshold
        if result['confidence'] < self.confidence_threshold:
            result['language'] = self.fallback_language
            result['method'] = f"{result['method']}+fallback"

        # Add language name
        result['language_name'] = LANGUAGE_CODES.get(result['language'], 'Unknown')

        return result

    def _detect_lingua(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Detect language using lingua library."""
        if not self.lingua_detector:
            return {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'lingua_not_loaded',
                'all_scores': []
            }

        try:
            if return_all_scores:
                confidence_values = self.lingua_detector.compute_language_confidence_values(text)
                all_scores = []
                for conf in confidence_values:
                    lang_name = conf.language.name
                    iso_code = LINGUA_TO_ISO.get(lang_name, lang_name.lower()[:2])
                    all_scores.append({
                        'language': iso_code,
                        'confidence': conf.value
                    })

                if all_scores:
                    best = all_scores[0]
                    return {
                        'language': best['language'],
                        'confidence': best['confidence'],
                        'method': 'lingua',
                        'all_scores': all_scores[:5]
                    }
            else:
                result = self.lingua_detector.detect_language_of(text)
                if result:
                    lang_name = result.name
                    iso_code = LINGUA_TO_ISO.get(lang_name, lang_name.lower()[:2])

                    # Get confidence
                    confidence = self.lingua_detector.compute_language_confidence(text, result)

                    return {
                        'language': iso_code,
                        'confidence': confidence,
                        'method': 'lingua',
                        'all_scores': []
                    }

        except Exception as e:
            self.logger.warning(f"Lingua detection failed: {e}")

        return {
            'language': self.fallback_language,
            'confidence': 0.0,
            'method': 'lingua_error',
            'all_scores': []
        }

    def _detect_langid(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Detect language using langid library."""
        if not HAS_LANGID:
            return {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'langid_not_available',
                'all_scores': []
            }

        try:
            if return_all_scores:
                ranks = langid.rank(text)
                all_scores = [
                    {'language': lang, 'confidence': max(0, min(1, (score + 100) / 100))}
                    for lang, score in ranks[:5]
                ]
                if all_scores:
                    return {
                        'language': all_scores[0]['language'],
                        'confidence': all_scores[0]['confidence'],
                        'method': 'langid',
                        'all_scores': all_scores
                    }
            else:
                lang, confidence = langid.classify(text)
                # Normalize confidence to 0-1 range
                normalized_conf = max(0, min(1, (confidence + 100) / 100))
                return {
                    'language': lang,
                    'confidence': normalized_conf,
                    'method': 'langid',
                    'all_scores': []
                }

        except Exception as e:
            self.logger.warning(f"Langid detection failed: {e}")

        return {
            'language': self.fallback_language,
            'confidence': 0.0,
            'method': 'langid_error',
            'all_scores': []
        }

    def _detect_ensemble(self, text: str, return_all_scores: bool) -> Dict[str, Any]:
        """Use multiple methods and combine results."""
        results = []

        if HAS_LINGUA and self.lingua_detector:
            lingua_result = self._detect_lingua(text, False)
            if lingua_result['confidence'] > 0:
                results.append(lingua_result)

        if HAS_LANGID:
            langid_result = self._detect_langid(text, False)
            if langid_result['confidence'] > 0:
                results.append(langid_result)

        if not results:
            return {
                'language': self.fallback_language,
                'confidence': 0.0,
                'method': 'ensemble_empty',
                'all_scores': []
            }

        # Vote on best language
        lang_votes = {}
        for r in results:
            lang = r['language']
            lang_votes[lang] = lang_votes.get(lang, 0) + r['confidence']

        best_lang = max(lang_votes, key=lang_votes.get)
        avg_confidence = lang_votes[best_lang] / len([r for r in results if r['language'] == best_lang])

        return {
            'language': best_lang,
            'confidence': avg_confidence,
            'method': 'ensemble',
            'all_scores': []
        }

    def detect_batch(
        self,
        texts: List[str],
        show_progress: bool = False
    ) -> List[Dict[str, Any]]:
        """
        Detect language for multiple texts.

        Args:
            texts: List of texts to analyze
            show_progress: Show progress bar

        Returns:
            List of detection results
        """
        results = []

        iterator = texts
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(texts, desc="Detecting languages")
            except ImportError:
                pass

        for text in iterator:
            results.append(self.detect(text))

        return results


def detect_language(text: str, fallback: str = 'en') -> Dict[str, Any]:
    """
    Convenience function to detect language of text.

    Args:
        text: Text to analyze
        fallback: Fallback language code

    Returns:
        Detection result dictionary
    """
    detector = LanguageDetector(fallback_language=fallback)
    return detector.detect(text)


def get_language_name(code: str) -> str:
    """Get full language name from ISO code."""
    return LANGUAGE_CODES.get(code, code.upper())
