#!/usr/bin/env python3
"""
Transcribe Tool - Main Entry Point

Unified audio extraction and transcription tool for YouTube, TikTok, and local files.

Usage:
    python -m transcribe_tool           # Interactive mode
    transcribe-tool                     # If installed via pip
    tt                                  # Short alias

Author: Antoine Lemor
"""

# Suppress warnings before any other imports
import warnings
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", category=UserWarning, module="pyannote")
warnings.filterwarnings("ignore", category=FutureWarning)

import sys
import argparse
from pathlib import Path


def main():
    """Main entry point for Transcribe Tool."""
    parser = argparse.ArgumentParser(
        prog='transcribe-tool',
        description='Unified audio extraction and transcription tool',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    transcribe-tool                     # Interactive mode
    transcribe-tool --youtube URL       # Download YouTube video
    transcribe-tool --transcribe FILE   # Transcribe audio file
    transcribe-tool --scan DIR          # Scan directory for audio

For more information, visit: https://github.com/antoine-lemor/Transcribe-tool
        """
    )

    # Mode selection
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        '-i', '--interactive',
        action='store_true',
        help='Run in interactive mode (default)'
    )
    mode_group.add_argument(
        '-y', '--youtube',
        metavar='URL',
        help='Download audio from YouTube URL'
    )
    mode_group.add_argument(
        '-t', '--tiktok',
        metavar='URL',
        help='Download audio from TikTok URL'
    )
    mode_group.add_argument(
        '--transcribe',
        metavar='FILE',
        help='Transcribe an audio file'
    )
    mode_group.add_argument(
        '--scan',
        metavar='DIR',
        help='Scan directory for audio files'
    )

    # Options
    parser.add_argument(
        '-l', '--language',
        default=None,
        help='Language code for transcription (auto-detect if not specified)'
    )
    parser.add_argument(
        '-o', '--output',
        metavar='DIR',
        help='Output directory'
    )
    parser.add_argument(
        '-m', '--model',
        default='large-v3',
        choices=['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        help='Whisper model to use (default: large-v3)'
    )
    parser.add_argument(
        '-f', '--format',
        default='txt',
        choices=['txt', 'csv', 'json', 'srt', 'vtt'],
        help='Output format for transcription (default: txt)'
    )
    parser.add_argument(
        '--diarize',
        action='store_true',
        help='Enable speaker diarization'
    )
    parser.add_argument(
        '--cookies',
        choices=['chrome', 'firefox', 'safari', 'edge'],
        help='Browser to use for cookies'
    )
    parser.add_argument(
        '-v', '--verbose',
        action='store_true',
        help='Verbose output'
    )
    parser.add_argument(
        '--version',
        action='version',
        version='%(prog)s 1.0.0'
    )

    args = parser.parse_args()

    # Setup logging
    import logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    logger = logging.getLogger('transcribe_tool')

    # Handle different modes
    if args.youtube:
        _handle_youtube(args, logger)
    elif args.tiktok:
        _handle_tiktok(args, logger)
    elif args.transcribe:
        _handle_transcribe(args, logger)
    elif args.scan:
        _handle_scan(args, logger)
    else:
        # Default: interactive mode
        _run_interactive()


def _run_interactive():
    """Run interactive CLI mode."""
    try:
        from .cli import run_cli
        run_cli()
    except ImportError as e:
        print(f"Error: Missing dependencies - {e}")
        print("Install with: pip install transcribe-tool[full]")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.")
        sys.exit(0)


def _handle_youtube(args, logger):
    """Handle YouTube download."""
    from .extractors import YouTubeExtractor
    from .extractors.base import ExtractorConfig
    from .config.settings import get_config

    config = get_config()
    if args.output:
        config.audio_dir = Path(args.output)

    extractor_config = ExtractorConfig(
        output_dir=config.audio_dir,
        cookies_from_browser=args.cookies
    )

    try:
        extractor = YouTubeExtractor(config=extractor_config, logger=logger)
        print(f"Downloading from: {args.youtube}")

        result = extractor.extract_audio(args.youtube)

        if result.success:
            print(f"Downloaded: {result.audio_path}")

            # Auto-transcribe if requested
            if args.transcribe:
                _transcribe_file(result.audio_path, args, logger)
        else:
            print(f"Failed: {result.error}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def _handle_tiktok(args, logger):
    """Handle TikTok download."""
    from .extractors import TikTokExtractor
    from .extractors.base import ExtractorConfig
    from .config.settings import get_config

    config = get_config()
    if args.output:
        config.audio_dir = Path(args.output)

    extractor_config = ExtractorConfig(
        output_dir=config.audio_dir,
        cookies_from_browser=args.cookies
    )

    try:
        extractor = TikTokExtractor(config=extractor_config, logger=logger)
        print(f"Downloading from: {args.tiktok}")

        result = extractor.extract_audio(args.tiktok)

        if result.success:
            print(f"Downloaded: {result.audio_path}")
        else:
            print(f"Failed: {result.error}")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def _handle_transcribe(args, logger):
    """Handle transcription."""
    audio_path = Path(args.transcribe)

    if not audio_path.exists():
        print(f"File not found: {audio_path}")
        sys.exit(1)

    _transcribe_file(audio_path, args, logger)


def _transcribe_file(audio_path: Path, args, logger):
    """Transcribe a single file."""
    from .transcriber import WhisperTranscriber
    from .transcriber.whisper_transcriber import TranscriptionConfig
    from .transcriber.text_processor import TextProcessor
    from .config.settings import get_config

    config = get_config()

    transcriber_config = TranscriptionConfig(
        model_name=args.model,
        language=args.language
    )

    try:
        transcriber = WhisperTranscriber(config=transcriber_config, logger=logger)

        print(f"Loading model: {args.model}")
        transcriber.load_model()

        print(f"Transcribing: {audio_path.name}")
        result = transcriber.transcribe(audio_path, args.language)

        if not result.success:
            print(f"Transcription failed: {result.error}")
            sys.exit(1)

        print(f"Language: {result.language}")
        print(f"Words: {len(result.words)}")
        print(f"Duration: {result.duration:.1f}s")

        # Handle diarization
        if args.diarize:
            try:
                import os
                from .transcriber.diarization import SpeakerDiarizer, DiarizationConfig

                diar_config = DiarizationConfig(hf_token=os.getenv("HF_TOKEN"))
                diarizer = SpeakerDiarizer(config=diar_config, logger=logger)

                print("Running speaker diarization...")
                diar_result = diarizer.diarize(audio_path)

                if diar_result.success:
                    print(f"Found {len(diar_result.speakers)} speakers")

                    import pandas as pd
                    words_df = pd.DataFrame(result.words)
                    words_df['speaker'] = None
                    words_df = diarizer.align_words_with_speakers(words_df, diar_result)
                    result.words = words_df.to_dict('records')

                diarizer.cleanup()

            except ImportError:
                print("Pyannote not available, skipping diarization")

        # Save output
        output_dir = Path(args.output) if args.output else config.transcripts_dir
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{audio_path.stem}.{args.format}"

        # Extract metadata from filename
        import re
        from datetime import datetime
        metadata = {
            'audio_file': audio_path.name,
            'title': audio_path.stem,
            'date': None,
            'video_id': None,
            'source': 'local',
            'transcribed_at': datetime.now().isoformat()
        }
        date_match = re.search(r'(\d{8})', audio_path.stem)
        if date_match:
            metadata['date'] = date_match.group(1)
        parts = audio_path.stem.split('_')
        if len(parts) >= 2:
            potential_id = parts[-1]
            if 8 <= len(potential_id) <= 15:
                metadata['video_id'] = potential_id

        if args.format == "txt":
            content = TextProcessor.format_transcript_with_metadata(
                result.words,
                metadata=metadata,
                include_speakers=args.diarize,
                language=result.language
            )
            output_path.write_text(content, encoding='utf-8')

        elif args.format == "csv":
            content = TextProcessor.format_csv(
                result.words,
                metadata=metadata,
                segmentation_level='sentence',
                language=result.language
            )
            output_path.write_text(content, encoding='utf-8')

        elif args.format == "json":
            import json
            data = {
                'metadata': metadata,
                'audio_file': str(audio_path),
                'language': result.language,
                'duration': result.duration,
                'text': result.text,
                'words': result.words
            }
            output_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding='utf-8')

        elif args.format == "srt":
            content = TextProcessor.format_srt(result.words, result.language)
            output_path.write_text(content, encoding='utf-8')

        elif args.format == "vtt":
            content = TextProcessor.format_vtt(result.words, result.language)
            output_path.write_text(content, encoding='utf-8')

        print(f"Saved: {output_path}")

        transcriber.cleanup()

    except Exception as e:
        logger.error(f"Error: {e}")
        sys.exit(1)


def _handle_scan(args, logger):
    """Handle directory scan."""
    from .extractors import LocalExtractor
    from .extractors.base import ExtractorConfig

    directory = Path(args.scan)

    if not directory.exists():
        print(f"Directory not found: {directory}")
        sys.exit(1)

    extractor = LocalExtractor(logger=logger)
    files = extractor.scan_directory(str(directory))

    print(f"Found {len(files)} audio/video files:")
    for f in files:
        info = extractor.get_file_info(str(f))
        duration = f"{info['duration']:.1f}s" if info.get('duration') else "N/A"
        print(f"  {f.name} ({info['size_mb']:.1f} MB, {duration})")


if __name__ == "__main__":
    main()
