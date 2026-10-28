"""Check source distribution hygiene and local Markdown links without network access.

Credential patterns are deliberately narrow and the private-content patterns are
heuristic. Passing is not a complete privacy or security audit; inspect the
candidate files manually as well.
"""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = (
    "LICENSE", "NOTICE", "README.md", "AGENTS.md", "CONTRIBUTING.md", "SECURITY.md",
    "docs/TRAINING.md", "docs/COMPATIBILITY.md",
    "tests/data/interpolate_reference.npz", "tests/data/nano_state_dict_shapes.json",
)
FORBIDDEN = {".git", "upstream", "weights", "artifacts", ".jax_cache", ".jax_cache_tpu", "__pycache__"}
INTERNAL_REPORTS = {"tpu.md", "gpu.md", "comparison.md"}
# Split the literal prefixes so the scanner does not flag its own source.
SECRETS = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE" + r" KEY-----"
    + r"|(?:gh" + r"p_|github_pat_)[A-Za-z0-9_]{30,}"
    + r"|AK" + r"IA[0-9A-Z]{16}"
)
# Personal, cloud and cost context stays in ignored artifacts. Literals are split so the
# scanner does not flag its own source; reserved example domains and noreply are allowed.
ALLOWED_EMAILS = re.compile(
    r"[A-Za-z0-9._%+-]+@(?:example\.(?:com|org|net)|(?:users\.)?noreply\.github\.com|github\.com)\b"
)
PRIVATE_CONTENT = {
    "email address": re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}"),
    "home directory path": re.compile(r"(?:/ho" + r"me/|/Us" + r"ers/)[A-Za-z]"),
    "cloud storage URI": re.compile(r"\bgs:" + r"//"),
    "cloud zone or region": re.compile(
        r"\b(?:us|europe|asia|southamerica|northamerica|australia|me|africa)-[a-z]+\d(?:-[a-z])?\b", re.I
    ),
    "cloud project or instance name": re.compile(
        r"\b[a-z][a-z0-9-]*-\d{6}\b|\binstance-\d{8}-\d{6}\b|\bproject-[0-9a-f]{8}-[0-9a-f]{4}\b", re.I
    ),
    "accelerator node name": re.compile(r"\bno" + r"de-\d+\b", re.I),
    "assistant or vendor name": re.compile(
        r"\b(?:[Cc]la" + r"ude|[Cc]o" + r"dex|[Cc]hat" + r"GPT|Copi" + r"lot|Anthro" + r"pic|Open" + r"AI)\b"
    ),
    "currency amount": re.compile(
        r"\$\d+(?:[.,]\d+)+|US\$\s?\d|€\s?\d|\b\d+(?:\.\d+)?\s?(?:USD|EUR)\b"
    ),
}
LINK = re.compile(r"(?<!!)\[[^\]\n]+\]\(([^\s)]+)(?:\s+[^)]*)?\)")


def is_private_path(relative: Path) -> bool:
    """Recognize private output locations, including files force-added to Git."""
    parts = relative.parts
    return bool(
        set(parts) & FORBIDDEN
        or relative.as_posix() == "scripts/tpu.local.sh"
        or relative.suffix in {".pem", ".key"}
        or any(part.startswith((".venv", ".env", ".jax_cache")) for part in parts)
        or (parts and parts[0] in {"data", "runs"})
        or (len(parts) > 1 and parts[0] == "docs" and (
            parts[1] in INTERNAL_REPORTS or parts[1].endswith("-measurements")
        ))
    )


def source_files() -> list[Path]:
    """Use the candidate Git file set, or the extracted sdist's source files."""
    top = subprocess.run(
        ["git", "-C", str(ROOT), "rev-parse", "--show-toplevel"], capture_output=True, text=True,
    )
    if top.returncode == 0 and Path(top.stdout.strip()).resolve() == ROOT:
        result = subprocess.run(
            ["git", "-C", str(ROOT), "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
            check=True, capture_output=True,
        )
        return sorted({ROOT / path.decode() for path in result.stdout.split(b"\0") if path})
    generated = {"__pycache__", ".pytest_cache", ".ruff_cache", "dist", "build"}
    return sorted(
        path for path in ROOT.rglob("*")
        if path.is_file() and not any(
            part in generated or part.endswith(".egg-info")
            for part in path.relative_to(ROOT).parts
        )
    )


def main() -> None:
    """Fail with filenames only, never print matched credential contents."""
    errors = [f"missing required file: {name}" for name in REQUIRED if not (ROOT / name).is_file()]
    files = source_files()
    for path in files:
        relative = path.relative_to(ROOT)
        if not path.exists():  # A tracked deletion is not part of the candidate.
            continue
        if path.is_symlink():
            errors.append(f"symlink needs release review: {relative}")
            continue
        if is_private_path(relative):
            errors.append(f"private/generated file in source: {relative}")
        try:
            content = path.read_text()
        except (UnicodeDecodeError, OSError):
            continue
        if SECRETS.search(content):
            errors.append(f"possible credential marker: {relative}")
        scrubbed = ALLOWED_EMAILS.sub("", content)
        for kind, pattern in PRIVATE_CONTENT.items():
            if pattern.search(scrubbed):
                errors.append(f"private content marker: {relative} ({kind})")
        if path.suffix == ".md":
            # Ignore code blocks: example Markdown paths need not be real links.
            prose = re.sub(r"```.*?```", "", content, flags=re.S)
            for destination in LINK.findall(prose):
                if ":" in destination or destination.startswith("#"):
                    continue
                target = destination.split("#", 1)[0]
                if target and not (path.parent / target).exists():
                    errors.append(f"broken local link in {relative}: {target}")
    if errors:
        raise SystemExit("\n".join(errors))
    print(f"Repository checks passed ({len(files)} candidate files; heuristic secret scan only).")


if __name__ == "__main__":
    main()
