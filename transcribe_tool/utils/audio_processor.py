"""
Audio processing module for Transcribe Tool.

Handles vocal separation, audio filtering, and silence removal
for optimal transcription quality.
"""

import os
import re
import shutil
import subprocess
import logging
import time
from pathlib import Path
from typing import Optional, List, Callable
from dataclasses import dataclass
from tqdm import tqdm

from .platform_compat import resolve_binary, run_tool, child_utf8_env


def _detect_audio_device() -> str:
    """Detect the best device for audio processing (Demucs)."""
    import platform
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif platform.system() == "Darwin" and hasattr(torch.backends, 'mps'):
            if torch.backends.mps.is_available():
                return "mps"
    except ImportError:
        pass
    return "cpu"


@dataclass
class AudioProcessingConfig:
    """Configuration for audio processing."""
    low_pass_freq: int = 3200  # Hz
    high_pass_freq: int = 65   # Hz
    min_silence_len: int = 3000  # ms
    silence_thresh: int = -60   # dB
    enable_vocal_separation: bool = True
    device: str = "auto"  # auto, cuda, mps, or cpu
    show_progress: bool = True  # Show tqdm progress bars

    def __post_init__(self):
        """Resolve auto device."""
        if self.device == "auto":
            self.device = _detect_audio_device()


class AudioProcessor:
    """Handle audio separation, filtering, and preprocessing."""

    def __init__(
        self,
        config: Optional[AudioProcessingConfig] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize audio processor.

        Args:
            config: Audio processing configuration
            logger: Logger instance
        """
        self.config = config or AudioProcessingConfig()
        self.logger = logger or logging.getLogger(__name__)

        # Characters that are invalid or problematic inside a filename
        self.unsafe_filename_chars = [
            ' ', "'", '"', '(', ')', '&', ';', '|',
            '$', '`', '\\', '!', '#', '*', '?',
            '[', ']', '{', '}', '<', '>',
            '\n', '\r', '\t'
        ]

    def extract_audio_from_video(
        self,
        video_path: Path,
        output_path: Optional[Path] = None,
        format: str = "wav"
    ) -> Optional[Path]:
        """
        Extract audio track from video file.

        Args:
            video_path: Path to video file
            output_path: Output audio path (optional)
            format: Output audio format

        Returns:
            Path to extracted audio or None if failed
        """
        if not video_path.exists():
            self.logger.error(f"Video file not found: {video_path}")
            return None

        if output_path is None:
            output_path = video_path.with_suffix(f".{format}")

        ffmpeg = resolve_binary("ffmpeg")
        if ffmpeg is None:
            self.logger.error("FFmpeg not found. Install it and make sure it is on PATH")
            return None

        try:
            cmd = [
                ffmpeg,
                "-hide_banner", "-loglevel", "error",
                "-i", str(video_path),
                "-vn",  # No video
                "-acodec", "pcm_s16le" if format == "wav" else "libmp3lame",
                "-ar", "16000",  # 16kHz sample rate (good for speech)
                "-ac", "1",  # Mono
                "-y",  # Overwrite
                str(output_path)
            ]

            result = run_tool(cmd)

            if result.returncode != 0:
                self.logger.error(f"FFmpeg failed: {result.stderr}")
                return None

            self.logger.info(f"Extracted audio to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Audio extraction failed: {e}")
            return None

    def separate_vocals(
        self,
        audio_path: Path,
        output_dir: Path
    ) -> Optional[Path]:
        """
        Separate vocals from audio using Demucs.

        Args:
            audio_path: Input audio path
            output_dir: Output directory

        Returns:
            Path to separated vocals or None if failed
        """
        if not self.config.enable_vocal_separation:
            self.logger.info("Vocal separation disabled, using original audio")
            return audio_path

        try:
            # Create safe filename
            safe_stem = self._sanitize_filename(audio_path.stem)

            # Create temp directory for Demucs output
            demucs_output = output_dir / "demucs_temp"
            demucs_output.mkdir(parents=True, exist_ok=True)

            # Demucs names its output directory after the input stem, so feed it
            # a sanitized copy when the original stem would not survive
            audio_for_demucs = audio_path
            using_temp = False
            if safe_stem != audio_path.stem:
                temp_audio = output_dir / f"{safe_stem}{audio_path.suffix}"
                if not (temp_audio.exists() and temp_audio.samefile(audio_path)):
                    shutil.copy2(audio_path, temp_audio)
                    audio_for_demucs = temp_audio
                    using_temp = True

            # Run Demucs
            cmd = [
                resolve_binary("demucs") or "demucs",
                "--two-stems=vocals",
                "-n", "htdemucs",
                "--out", str(demucs_output),
                "--device", self.config.device,
                "--jobs", "1",
                str(audio_for_demucs)
            ]

            self.logger.info(f"Running Demucs on: {audio_path.name}")

            # Get audio duration for progress estimation
            audio_duration = self.get_audio_duration(audio_path) or 60.0

            # Demucs is a child CPython and would otherwise encode its piped
            # output with the locale code page
            demucs_env = child_utf8_env()

            if self.config.show_progress:
                # Run demucs with progress bar (estimate based on audio duration)
                # Demucs typically processes at ~2-5x realtime on CPU
                estimated_time = audio_duration / 3.0  # Rough estimate

                pbar = tqdm(
                    total=100,
                    desc="Vocal separation",
                    unit="%",
                    leave=False,
                    bar_format='{l_bar}{bar}| {n:.0f}% [{elapsed}<{remaining}]'
                )

                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    env=demucs_env,
                    # Matches run_tool(): without it the progress-bar path
                    # flashes a console window on Windows while the non-progress
                    # and CPU-fallback paths do not.
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)
                )

                start_time = time.time()
                while True:
                    try:
                        stdout, stderr = process.communicate(timeout=0.5)
                        break
                    except subprocess.TimeoutExpired:
                        # communicate() keeps draining both pipes across
                        # timeouts; polling alone deadlocks as soon as Demucs'
                        # progress bar fills the OS pipe buffer
                        elapsed = time.time() - start_time
                        progress = min(95, (elapsed / estimated_time) * 100)
                        pbar.n = progress
                        pbar.refresh()

                pbar.n = 100
                pbar.refresh()
                pbar.close()

                returncode = process.returncode
            else:
                result = run_tool(cmd, env=demucs_env)
                returncode = result.returncode
                stderr = result.stderr

            if returncode != 0:
                self.logger.error(f"Demucs failed: {stderr}")
                # Try CPU fallback
                if self.config.device != "cpu":
                    self.logger.info("Retrying with CPU...")
                    cmd[cmd.index("--device") + 1] = "cpu"
                    result = run_tool(cmd, env=demucs_env)
                    returncode = result.returncode
                    stderr = result.stderr
                    if returncode != 0:
                        self.logger.error(f"Demucs failed on CPU: {stderr}")
                        return None
                else:
                    return None

            # Find vocals file
            vocals_path = self._find_vocals_file(demucs_output, safe_stem, audio_path.stem)

            if not vocals_path:
                self.logger.error("Could not find separated vocals")
                return None

            # Copy to output
            output_path = output_dir / f"{safe_stem}_vocals.wav"
            shutil.copy2(vocals_path, output_path)

            # Cleanup
            try:
                shutil.rmtree(demucs_output)
                if using_temp:
                    audio_for_demucs.unlink()
            except Exception:
                pass

            self.logger.info(f"Separated vocals saved to: {output_path}")
            return output_path

        except FileNotFoundError:
            self.logger.warning("Demucs not installed, skipping vocal separation")
            return audio_path
        except Exception as e:
            self.logger.error(f"Vocal separation failed: {e}")
            return audio_path

    def _find_vocals_file(
        self,
        demucs_dir: Path,
        safe_stem: str,
        original_stem: str
    ) -> Optional[Path]:
        """Find the vocals.wav file in Demucs output."""
        # Try expected paths
        candidates = [
            demucs_dir / "htdemucs" / safe_stem / "vocals.wav",
            demucs_dir / safe_stem / "vocals.wav",
            demucs_dir / "htdemucs" / original_stem / "vocals.wav",
            demucs_dir / original_stem / "vocals.wav",
        ]

        for path in candidates:
            if path.exists():
                return path

        # Search recursively
        for vocals_file in demucs_dir.rglob("vocals.wav"):
            return vocals_file

        return None

    def apply_filters(
        self,
        audio_path: Path,
        output_path: Optional[Path] = None
    ) -> Optional[Path]:
        """
        Apply audio filters (frequency filtering and silence removal).

        Args:
            audio_path: Input audio path
            output_path: Output path (optional)

        Returns:
            Path to filtered audio or None if failed
        """
        try:
            from pydub import AudioSegment
            from pydub.silence import detect_nonsilent
        except ImportError:
            self.logger.warning("pydub not installed, skipping filters")
            return audio_path

        if output_path is None:
            output_path = audio_path.with_stem(f"{audio_path.stem}_filtered")

        try:
            # Create progress bar for filtering steps
            steps = ["Loading audio", "Low-pass filter", "High-pass filter",
                     "Detecting silence", "Removing silence", "Exporting"]

            if self.config.show_progress:
                pbar = tqdm(
                    total=len(steps),
                    desc="Audio filtering",
                    unit="step",
                    leave=False,
                    bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}]'
                )
            else:
                pbar = None

            def update_progress(step_name: str):
                if pbar:
                    pbar.set_postfix_str(step_name)
                    pbar.update(1)

            # Load audio
            update_progress("Loading")
            audio = AudioSegment.from_file(str(audio_path))

            # Apply frequency filters
            update_progress("Low-pass")
            audio = audio.low_pass_filter(self.config.low_pass_freq)

            update_progress("High-pass")
            audio = audio.high_pass_filter(self.config.high_pass_freq)

            self.logger.info(
                f"Applied filters: LP={self.config.low_pass_freq}Hz, "
                f"HP={self.config.high_pass_freq}Hz"
            )

            # Detect non-silent segments
            update_progress("Detecting silence")
            nonsilent = detect_nonsilent(
                audio,
                min_silence_len=self.config.min_silence_len,
                silence_thresh=self.config.silence_thresh
            )

            self.logger.info(f"Detected {len(nonsilent)} non-silent segments")

            # Create audio with only non-silent parts
            update_progress("Removing silence")
            if nonsilent:
                processed = AudioSegment.silent(duration=0)
                for start, end in nonsilent:
                    processed += audio[start:end]
                audio = processed

            # Export
            update_progress("Exporting")
            audio.export(str(output_path), format="wav")

            if pbar:
                pbar.close()

            self.logger.info(f"Filtered audio saved to: {output_path}")
            return output_path

        except Exception as e:
            self.logger.error(f"Audio filtering failed: {e}")
            return audio_path

    def process_audio(
        self,
        audio_path: Path,
        output_dir: Path,
        apply_vocal_separation: bool = True,
        apply_filters: bool = True
    ) -> Optional[Path]:
        """
        Full audio processing pipeline.

        Args:
            audio_path: Input audio path
            output_dir: Output directory
            apply_vocal_separation: Whether to separate vocals
            apply_filters: Whether to apply frequency filters

        Returns:
            Path to processed audio or None if failed
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        current_audio = audio_path

        # Step 1: Vocal separation
        if apply_vocal_separation and self.config.enable_vocal_separation:
            separated = self.separate_vocals(current_audio, output_dir)
            if separated:
                current_audio = separated

        # Step 2: Apply filters
        if apply_filters:
            filtered = self.apply_filters(current_audio, output_dir / f"{audio_path.stem}_processed.wav")
            if filtered:
                current_audio = filtered

        return current_audio

    def _sanitize_filename(self, filename: str) -> str:
        """Remove problematic characters from filename."""
        for char in self.unsafe_filename_chars:
            filename = filename.replace(char, '_')
        # Remove multiple underscores
        filename = re.sub(r'_+', '_', filename)
        return filename.strip('_')

    def get_audio_duration(self, audio_path: Path) -> Optional[float]:
        """Get audio duration in seconds."""
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(str(audio_path))
            return len(audio) / 1000.0
        except Exception:
            # Fallback to ffprobe
            try:
                result = run_tool(
                    [
                        resolve_binary("ffprobe") or "ffprobe", "-v", "error",
                        "-show_entries", "format=duration",
                        "-of", "default=noprint_wrappers=1:nokey=1",
                        str(audio_path)
                    ],
                    check=True
                )
                return float(result.stdout.strip())
            except Exception:
                return None

    def get_audio_info(self, audio_path: Path) -> dict:
        """Get audio file information."""
        info = {
            'path': str(audio_path),
            'name': audio_path.name,
            'size_mb': audio_path.stat().st_size / (1024 * 1024) if audio_path.exists() else 0,
            'duration': self.get_audio_duration(audio_path),
            'format': audio_path.suffix.lstrip('.')
        }
        return info
