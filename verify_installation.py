#!/usr/bin/env python3
"""
PROJECT:
-------
Transcribe Tool

TITLE:
------
verify_installation.py

MAIN OBJECTIVE:
---------------
Verification script to ensure Transcribe Tool is correctly installed with all dependencies
available and CLI commands accessible.

Dependencies:
-------------
- sys
- platform
- importlib.util
- subprocess
- shutil

MAIN FEATURES:
--------------
1) Python version verification (3.10+)
2) Transcribe Tool package verification
3) Core dependencies check (whisper, yt-dlp, rich, pandas, etc.)
4) Audio dependencies check (pydub, FFmpeg binary on PATH)
5) Optional dependencies check (pyannote, demucs)
6) GPU support detection (CUDA, MPS)
7) CLI commands verification
8) Comprehensive summary report

Author:
-------
Antoine Lemor
"""

import sys
import platform
import importlib.util
import subprocess
import shutil

try:
    from importlib import metadata as importlib_metadata
except ImportError:
    import importlib_metadata


if sys.platform == "win32":
    # Only 'errors' is forced. Forcing encoding='utf-8' would mojibake this
    # script's output whenever a caller reads it back with the console codepage
    # (redirection, install.ps1, subprocess.run(text=True)).
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


OK = "[OK]"
BAD = "[X]"
SKIP = "-"
# Width of the widest marker, so the columns after it stay aligned.
MARK_W = 4


def check_python_version():
    """Check if Python version meets requirements."""
    print("Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 10:
        print(f"  {OK} Python {version.major}.{version.minor}.{version.micro} (OK)")
        return True
    else:
        print(f"  {BAD} Python {version.major}.{version.minor}.{version.micro} (FAILED)")
        print("    Required: Python 3.10 or higher")
        return False


def _candidate_distribution_names(module):
    """Yield possible distribution names for a module (best-effort)."""
    names = [
        getattr(module, "__package__", None),
        getattr(module, "__name__", None),
    ]

    module_name = getattr(module, "__name__", "")
    if module_name:
        names.append(module_name.split(".")[0])

    seen = set()
    for name in filter(None, names):
        variants = {name}
        if "." in name:
            variants.add(name.replace(".", "-"))
        if "_" in name:
            variants.add(name.replace("_", "-"))
        for variant in variants:
            if variant and variant not in seen:
                seen.add(variant)
                yield variant


def _resolve_module_version(module):
    """Return the best-effort version string for a module."""
    for name in _candidate_distribution_names(module):
        try:
            return importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            continue

    version = getattr(module, "__version__", None)
    if version and version != "unknown":
        return version

    return "unknown"


def check_module(module_name, display_name=None, optional=False):
    """Check if a module is installed."""
    if display_name is None:
        display_name = module_name

    try:
        spec = importlib.util.find_spec(module_name)
        if spec is not None:
            module = importlib.import_module(module_name)
            version = _resolve_module_version(module)
            status = OK
            print(f"  {status:{MARK_W}s} {display_name:30s} version {version}")
            return True
        else:
            if optional:
                print(f"  {SKIP:{MARK_W}s} {display_name:30s} (optional, not installed)")
                return True
            else:
                print(f"  {BAD:{MARK_W}s} {display_name:30s} (MISSING)")
                return False
    except Exception as e:
        # An installed-but-unimportable module is broken, not absent: pydub on
        # Python 3.13+ reports "No module named 'pyaudioop'" (needs audioop-lts).
        if optional:
            print(f"  {SKIP:{MARK_W}s} {display_name:30s} (optional, unavailable: {e})")
            return True
        else:
            print(f"  {BAD:{MARK_W}s} {display_name:30s} (FAILED: {e})")
            return False


def check_transcribe_tool():
    """Check if transcribe_tool package is installed."""
    print("\nChecking Transcribe Tool installation...")
    try:
        import transcribe_tool
        version = getattr(transcribe_tool, "__version__", "1.0.0")
        print(f"  {OK} transcribe-tool version {version}")
        return True
    except ImportError as e:
        print(f"  {BAD} transcribe-tool not found: {e}")
        print("    Run: pip install -e .")
        return False


def check_core_dependencies():
    """Check core dependencies."""
    print("\nChecking core dependencies...")
    deps = [
        ("rich", "Rich (CLI interface)"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("pydub", "PyDub (audio processing)"),
    ]
    return all(check_module(mod, name) for mod, name in deps)


def check_extraction_dependencies():
    """Check extraction-related dependencies."""
    print("\nChecking extraction dependencies...")
    deps = [
        ("yt_dlp", "yt-dlp (YouTube/TikTok)"),
    ]
    result = all(check_module(mod, name) for mod, name in deps)

    # Check FFmpeg separately
    if shutil.which("ffmpeg"):
        try:
            proc = subprocess.run(
                ["ffmpeg", "-version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace'
            )
            version = proc.stdout.split('\n')[0].split(' ')[2] if proc.returncode == 0 else "unknown"
            print(f"  {OK:{MARK_W}s} {'FFmpeg':30s} version {version}")
        except Exception:
            print(f"  {OK:{MARK_W}s} {'FFmpeg':30s} (installed)")
    else:
        print(f"  {BAD:{MARK_W}s} {'FFmpeg':30s} (MISSING - required for audio extraction)")
        if sys.platform == "win32":
            print("    Install: winget install --id Gyan.FFmpeg -e")
            print("    Or download https://www.gyan.dev/ffmpeg/builds/ and add its bin\\ folder to PATH")
            print("    Extracting the zip is not enough - the bin\\ folder must be ON PATH")
            print("    Then open a NEW terminal so PATH is refreshed")
        elif sys.platform == "darwin":
            print("    Install: brew install ffmpeg")
        else:
            print("    Install: sudo apt install ffmpeg")
        result = False

    return result


def check_transcription_dependencies():
    """Check transcription-related dependencies."""
    print("\nChecking transcription dependencies...")
    deps = [
        ("whisper", "OpenAI Whisper"),
        ("torch", "PyTorch"),
    ]
    return all(check_module(mod, name) for mod, name in deps)


def check_nlp_dependencies():
    """Check NLP/tokenization dependencies."""
    print("\nChecking NLP dependencies...")
    deps = [
        ("lingua", "Lingua (language detection)"),
    ]
    result = all(check_module(mod, name) for mod, name in deps)

    # Check spaCy (optional but recommended)
    check_module("spacy", "spaCy (sentence segmentation)", optional=True)
    check_module("nltk", "NLTK (fallback tokenizer)", optional=True)

    return result


def check_optional_dependencies():
    """Check optional dependencies."""
    print("\nChecking optional dependencies...")
    deps = [
        ("pyannote.audio", "Pyannote (speaker diarization)", True),
        ("demucs", "Demucs (voice separation)", True),
        ("psutil", "psutil (system monitoring)", True),
    ]
    all(check_module(mod, name, optional) for mod, name, optional in deps)
    return True  # Optional deps don't fail verification


def check_gpu_support():
    """Check GPU availability."""
    print("\nChecking GPU support...")
    try:
        import torch

        # Check CUDA
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count > 0 else "Unknown"
            print(f"  {OK} CUDA available: {device_count} device(s)")
            print(f"    Primary GPU: {device_name}")
            return True

        # Check MPS (Apple Silicon)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            print(f"  {OK} MPS (Apple Silicon) available")
            return True

        else:
            print(f"  {SKIP} No GPU detected (CPU only)")
            print(f"    Transcription will use CPU (slower but works)")
            # An NVIDIA driver next to a CUDA-less torch means the CPU-only PyPI
            # wheel was installed, which is the Windows default.
            if (torch.version.cuda is None
                    and platform.system() == "Windows"
                    and shutil.which("nvidia-smi")):
                print("    An NVIDIA GPU was detected but PyTorch has no CUDA support")
                print("    (PyPI ships a CPU-only torch wheel on Windows). To enable it:")
                print("      pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128")
            return True

    except Exception as e:
        print(f"  {BAD} Error checking GPU: {e}")
        return False


def check_cli_commands():
    """Check if CLI commands are available."""
    print("\nChecking CLI commands...")

    commands = ["transcribe-tool", "tt"]
    success = True

    for cmd in commands:
        try:
            result = subprocess.run(
                [cmd, "--version"],
                capture_output=True,
                text=True,
                encoding='utf-8',
                errors='replace',
                timeout=10
            )
            if result.returncode == 0 or "transcribe" in result.stdout.lower() or "1.0.0" in result.stdout:
                print(f"  {OK} '{cmd}' command available")
            else:
                print(f"  {BAD} '{cmd}' command not working")
                success = False
        except FileNotFoundError:
            print(f"  {BAD} '{cmd}' command not found")
            success = False
        except Exception as e:
            print(f"  {BAD} '{cmd}' error: {e}")
            success = False

    return success


def main():
    print("=" * 70)
    print("TRANSCRIBE TOOL - Installation Verification")
    print("=" * 70)

    checks = [
        ("Python Version", check_python_version),
        ("Transcribe Tool Package", check_transcribe_tool),
        ("Core Dependencies", check_core_dependencies),
        ("Extraction Dependencies", check_extraction_dependencies),
        ("Transcription Dependencies", check_transcription_dependencies),
        ("NLP Dependencies", check_nlp_dependencies),
        ("Optional Dependencies", check_optional_dependencies),
        ("GPU Support", check_gpu_support),
        ("CLI Commands", check_cli_commands),
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  {BAD} Unexpected error during {name}: {e}")
            results.append((name, False))

    # Summary
    print("\n" + "=" * 70)
    print("VERIFICATION SUMMARY")
    print("=" * 70)

    non_optional_results = [r for r in results if r[0] != "Optional Dependencies"]
    passed = sum(1 for _, result in non_optional_results if result)
    total = len(non_optional_results)

    for name, result in non_optional_results:
        status = f"{OK} PASS" if result else f"{BAD} FAIL"
        print(f"  {status:10s} {name}")

    print("=" * 70)

    if passed == total:
        print(f"{OK} ALL CHECKS PASSED")
        print()
        print("Transcribe Tool is correctly installed and ready to use!")
        print()
        print("Next steps:")
        print("  1. Run the CLI: transcribe-tool (or tt)")
        print("  2. Transcribe a file: transcribe-tool --transcribe audio.mp3")
        print("  3. Download from YouTube: transcribe-tool --youtube URL")
        print()
        return 0
    else:
        print(f"{BAD} {total - passed} CHECK(S) FAILED")
        print()
        print("Please fix the failed checks before using Transcribe Tool.")
        print()
        print("Common fixes:")
        print("  - Missing dependencies: pip install -e .[full]")
        print("  - CLI not found: pip install -e .")
        if sys.platform == "win32":
            print("  - FFmpeg missing: winget install --id Gyan.FFmpeg -e (then reopen the terminal)")
            print("  - Import errors: Check virtual environment is activated (.venv\\Scripts\\activate.bat)")
            print("  - torch [WinError 126]: install https://aka.ms/vs/17/release/vc_redist.x64.exe")
        elif sys.platform == "darwin":
            print("  - FFmpeg missing: brew install ffmpeg")
            print("  - Import errors: Check virtual environment is activated (source .venv/bin/activate)")
        else:
            print("  - FFmpeg missing: sudo apt install ffmpeg")
            print("  - Import errors: Check virtual environment is activated (source .venv/bin/activate)")
        print()
        return 1


if __name__ == "__main__":
    sys.exit(main())
