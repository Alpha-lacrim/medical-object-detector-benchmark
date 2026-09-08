"""Check repository BibTeX syntax, identifiers, and Markdown citation resolution.

This offline check accepts the repository's literal-field BibTeX format (braced,
quoted, numeric, or month-name values). It deliberately rejects macros and
concatenation rather than pretending to parse unrestricted BibTeX. Source
existence and claim support still require the separate citation audit.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path


def parse_bibliography(text: str) -> dict[str, dict[str, str]]:
    """Parse literal-field entries, rejecting malformed or duplicate records."""
    entries: dict[str, dict[str, str]] = {}
    offset = 0

    def token(pattern: str) -> str:
        nonlocal offset
        match = re.compile(r"\s*" + pattern).match(text, offset)
        if match is None:
            raise ValueError(f"invalid BibTeX near offset {offset}: {text[offset : offset + 40]!r}")
        offset = match.end()
        return match[0].strip()

    def literal() -> str:
        nonlocal offset
        token(r"(?=\S)")
        start = offset
        if text[offset] not in '{"':
            return token(r"(?:\d+|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b")
        opening = text[offset]
        offset += 1
        depth = 1 if opening == "{" else 0
        while offset < len(text):
            char = text[offset]
            offset += 1
            if char == "\\":
                offset += 1
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth < 0:
                    raise ValueError("unbalanced braces in quoted BibTeX value")
                if opening == "{" and depth == 0:
                    return text[start + 1 : offset - 1]
            elif char == '"' and opening == '"' and depth == 0:
                return text[start + 1 : offset - 1]
        raise ValueError("unterminated BibTeX value")

    while text[offset:].strip():
        token(r"@[A-Za-z]+\s*\{")
        key = token(r"[\w:.-]+")
        token(",")
        fields: dict[str, str] = {}
        while True:
            if re.match(r"\s*\}", text[offset:]):
                token(r"\}")
                break
            field = token(r"[A-Za-z][\w-]*").lower()
            token("=")
            value = literal()
            if field in fields:
                raise ValueError(f"duplicate field {field} in {key}")
            fields[field] = value
            if re.match(r"\s*,", text[offset:]):
                token(",")
            elif not re.match(r"\s*\}", text[offset:]):
                raise ValueError(f"missing field separator in {key}")
        if key in entries:
            raise ValueError(f"duplicate citation key: {key}")
        for field in ("author", "title", "year"):
            if not fields.get(field, "").strip():
                raise ValueError(f"missing {field} in {key}")
        if not re.fullmatch(r"\d{4}", fields["year"]):
            raise ValueError(f"invalid year in {key}")
        if "doi" in fields and not re.fullmatch(r"10\.\d{4,9}/\S+", fields["doi"]):
            raise ValueError(f"invalid DOI syntax in {key}")
        entries[key] = fields
    if not entries:
        raise ValueError("empty bibliography")
    dois = [fields["doi"].lower() for fields in entries.values() if "doi" in fields]
    if len(dois) != len(set(dois)):
        raise ValueError("duplicate DOI in bibliography")
    return entries


def citation_keys(markdown: str) -> set[str]:
    """Extract Pandoc-style citation identifiers, excluding metric notation."""
    return set(re.findall(r"(?<![\w@])@([A-Za-z](?:[\w:.-]*\w)?)", markdown))


def verify(bibliography: Path, manuscripts: list[Path]) -> tuple[int, dict[str, int]]:
    """Check syntax and resolve every used key without pruning shared entries."""
    entries = parse_bibliography(bibliography.read_text(encoding="utf-8"))
    counts = {}
    for path in manuscripts:
        keys = citation_keys(path.read_text(encoding="utf-8"))
        missing = keys - entries.keys()
        if missing:
            raise ValueError(f"{path}: unresolved citations: {', '.join(sorted(missing))}")
        counts[path.as_posix()] = len(keys)
    return len(entries), counts


def main() -> int:
    """Run the explicit-path bibliography check."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bibliography", type=Path, required=True)
    parser.add_argument("--manuscripts", type=Path, nargs="+", required=True)
    args = parser.parse_args()
    try:
        count, cited = verify(args.bibliography, args.manuscripts)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Bibliography check failed: {error}\n")
    print(f"Bibliography check passed: {count} unique entries; cited keys {cited}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
