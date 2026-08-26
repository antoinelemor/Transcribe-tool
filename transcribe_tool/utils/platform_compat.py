"""
Cross-platform compatibility helpers for Transcribe Tool.

Centralises the small platform differences that would otherwise be repeated
(and re-broken) in every extractor, transcriber and CLI module: console
encoding, external binary lookup, subprocess decoding, file writing and
Windows path/ACL rules.

Kept deliberately dependency-free (standard library only) because it is
imported at the top of modules that run before torch/pandas/spacy are loaded.
"""

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional, Sequence, Union

# Windows CreateProcess searches the current working directory *before* PATH,
# and shutil.which deliberately prepends the CWD on Windows too, so a stray
# ffmpeg.exe dropped next to an audio file would win over the installed one.
if os.name == "nt":
    os.environ.setdefault("NoDefaultCurrentDirectoryInExePath", "1")

# Characters Windows forbids inside a path component. The two separators are
# included: a stem is a single component, so a "/" or "\" would silently push
# the output into a directory that does not exist (or, with "..", outside the
# destination directory altogether).
_FORBIDDEN_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

# Reserved DOS device names: "CON.txt" is just as unusable as "CON".
_RESERVED_NAMES = frozenset(
    ["CON", "PRN", "AUX", "NUL"]
    + [f"COM{i}" for i in range(1, 10)]
    + [f"LPT{i}" for i in range(1, 10)]
)


def configure_stdio() -> None:
    """Force UTF-8 on the Windows console so non-ASCII output cannot crash.

    The CLI prints box-drawing glyphs, check marks and arrows, which raise
    UnicodeEncodeError on a cp1252 console or as soon as output is redirected
    to a file. Every step is guarded: pythonw leaves sys.stdout as None and
    pytest replaces it with a non-TextIOWrapper object. Safe to call repeatedly.
    """
    if sys.platform != "win32":
        return

    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is None:
                continue
            if (stream.encoding or "").lower().replace("-", "") == "utf8":
                continue
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError, OSError):
            pass


def _which(name: str, path: Optional[str] = None) -> Optional[str]:
    """shutil.which without the implicit current-directory hit on Windows.

    CPython below 3.12 prepends os.curdir to the Windows search path
    unconditionally - it does not honour NoDefaultCurrentDirectoryInExePath -
    so a stray ffmpeg.exe sitting next to the audio files would be found here
    and then launched by its absolute path, which is exactly the hijack the
    environment variable above is meant to prevent. Since requires-python is
    >=3.10, drop such a hit unless the CWD is genuinely on the search path.
    """
    found = shutil.which(name, path=path) if path else shutil.which(name)
    if found is None or os.name != "nt":
        return found

    try:
        hit_dir = os.path.normcase(os.path.dirname(os.path.abspath(found)))
        cwd = os.path.normcase(os.getcwd())
        if hit_dir != cwd:
            return found
        searched = path if path is not None else os.environ.get("PATH", "")
        for entry in searched.split(os.pathsep):
            if entry and os.path.normcase(os.path.abspath(entry)) == cwd:
                return found
    except OSError:
        return found

    return None


def resolve_binary(name: str) -> Optional[str]:
    """Locate an external executable, honouring an explicit override.

    Args:
        name: Bare command name, e.g. "ffmpeg"

    Returns:
        Absolute path to the executable, or None if it cannot be found
    """
    override = os.environ.get(f"TRANSCRIBE_TOOL_{name.upper()}")
    if override:
        candidate = Path(override)
        if candidate.is_file():
            return str(candidate)
        if candidate.is_dir():
            found = _which(name, path=str(candidate))
            if found:
                return found

    found = _which(name)
    if found:
        return found

    if os.name == "nt":
        found = _which(f"{name}.exe")
        if found:
            return found

    return None


def run_tool(cmd: Union[str, Sequence[str]], **kwargs) -> subprocess.CompletedProcess:
    """Run an external tool with decoding defaults that work on every platform.

    A bare ``text=True`` decodes the child's output with the Windows ANSI code
    page (cp1252); ffmpeg and demucs emit UTF-8 and U+2588 progress glyphs, so
    subprocess.run raises UnicodeDecodeError before the caller ever sees the
    result. Explicit utf-8 plus errors="replace" turns that into log noise.

    Args:
        cmd: Command to run, as a list of arguments or a string
        **kwargs: Passed to subprocess.run; any default below can be overridden

    Returns:
        The completed process

    Raises:
        FileNotFoundError: If the first list element cannot be resolved on PATH
    """
    # capture_output cannot be combined with an explicit stdout/stderr:
    # subprocess.run raises ValueError rather than honouring the caller.
    if "stdout" not in kwargs and "stderr" not in kwargs:
        kwargs.setdefault("capture_output", True)
    kwargs.setdefault("text", True)
    kwargs.setdefault("encoding", "utf-8")
    kwargs.setdefault("errors", "replace")
    kwargs.setdefault("check", False)

    # A flashing console window on every ffmpeg call is unacceptable in a GUI
    # launch, but creationflags and startupinfo are mutually exclusive.
    if os.name == "nt" and "creationflags" not in kwargs and "startupinfo" not in kwargs:
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)

    if isinstance(cmd, (list, tuple)) and cmd:
        program = str(cmd[0])
        if program == os.path.basename(program) and not os.path.isabs(program):
            resolved = resolve_binary(program)
            if resolved is None:
                raise FileNotFoundError(
                    f"'{program}' was not found. Install it and make sure it is on "
                    f"your PATH, or set TRANSCRIBE_TOOL_{program.upper()} to its "
                    f"full path."
                )
            cmd = [resolved, *[str(a) for a in cmd[1:]]]

    return subprocess.run(cmd, **kwargs)


def child_utf8_env(extra: Optional[dict] = None) -> dict:
    """Build an environment that forces UTF-8 on a child Python process.

    Demucs is a child CPython that encodes its piped output with the locale
    code page, so forcing UTF-8 on the parent alone still yields mojibake.

    Args:
        extra: Additional environment variables to merge in

    Returns:
        A copy of the current environment with the UTF-8 flags set
    """
    env = os.environ.copy()
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    if extra:
        env.update(extra)
    return env


def write_csv_text(path: Union[str, Path], content: str) -> None:
    """Write pre-formatted CSV text without corrupting its line endings.

    csv.DictWriter's excel dialect already emits ``\\r\\n``; the default
    newline=None would translate the ``\\n`` half again into os.linesep and
    ship every row as ``\\r\\r\\n`` on Windows - blank rows in Excel and a
    stray ``\\r`` glued to the last column in R's read.csv. No BOM is written:
    the encoding is plain utf-8 so read.csv() keeps working.

    Args:
        path: Destination file
        content: CSV text to write
    """
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(content)


def write_text_file(path: Union[str, Path], content: str) -> None:
    """Write UTF-8 text, translating newlines to the platform convention.

    Convenience twin of write_csv_text for txt/srt/vtt output, where the
    default ``\\n`` -> os.linesep translation is what we want.

    Args:
        path: Destination file
        content: Text to write
    """
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def safe_output_path(
    directory: Union[str, Path],
    stem: str,
    suffix: str,
    limit: int = 200
) -> Path:
    """Build an output path that is valid and short enough on Windows.

    Only the stem is sanitised and truncated - the directory is never touched.
    When truncation happens a short hash of the original stem is appended so
    two different long titles cannot collide onto the same file. Short, clean
    names are returned unchanged, so existing macOS output keeps its names.

    Args:
        directory: Destination directory
        stem: File name without extension
        suffix: Extension including its leading dot, e.g. ".csv"
        limit: Maximum length of the resulting absolute path

    Returns:
        Path to write to (the directory is not created)
    """
    directory = Path(directory)

    safe_stem = _FORBIDDEN_NAME_CHARS.sub("", stem).rstrip(". ")
    if not safe_stem:
        safe_stem = "output"
    if safe_stem.split(".")[0].upper() in _RESERVED_NAMES:
        safe_stem = f"{safe_stem}_"

    # abspath rather than resolve: a UNC path or a disconnected network drive
    # can raise on Windows.
    dir_len = len(os.path.abspath(str(directory)))
    budget = limit - dir_len - len(os.sep) - len(suffix)

    if len(safe_stem) > budget:
        digest = hashlib.sha1(stem.encode("utf-8", "replace")).hexdigest()[:8]
        keep = max(budget - len(digest) - 1, 1)
        safe_stem = f"{safe_stem[:keep]}_{digest}"

    return directory / f"{safe_stem}{suffix}"


def restrict_permissions(path: Union[str, Path]) -> None:
    """Restrict a secret file or directory to the current user, best effort.

    Path.chmod(0o600) on Windows only toggles the read-only attribute, so a
    token file would keep its inherited ACLs - and be cloud-synced along with
    the rest of a OneDrive-backed profile. icacls is the only way to actually
    drop inheritance. Never raises.

    Args:
        path: File or directory to protect
    """
    path = Path(path)
    try:
        if os.name == "nt":
            user = os.environ.get("USERNAME")
            if user:
                domain = os.environ.get("USERDOMAIN")
                if domain:
                    user = f"{domain}\\{user}"
            else:
                import getpass
                user = getpass.getuser()

            grant = f"{user}:(OI)(CI)F" if path.is_dir() else f"{user}:F"
            run_tool(
                ["icacls", str(path), "/inheritance:r", "/grant:r", grant],
                check=False
            )
        else:
            os.chmod(path, 0o700 if path.is_dir() else 0o600)
    except Exception:
        pass


def load_saved_hf_token() -> Optional[str]:
    """Read the HuggingFace token saved by the interactive CLI.

    Checks the system keychain first, then the JSON fallback file. Read-only:
    the config directory is never created here.

    Returns:
        The saved token, or None if there is none
    """
    try:
        import keyring
        token = keyring.get_password("transcribe-tool", "hf_token")
        if token:
            return token
    except Exception:
        pass

    try:
        config_path = Path.home() / ".transcribe-tool" / "config.json"
        if config_path.exists():
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config.get('hf_token')
    except Exception:
        pass

    return None
