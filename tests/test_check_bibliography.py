"""Checks for bibliography corruption and unresolved manuscript citations."""

from pathlib import Path

import pytest

from scripts.check_bibliography import citation_keys, parse_bibliography, verify

ENTRY = "@article{one, author={A. Author}, title={A {nested} title}, year={2024}}"


def test_literal_fields_and_metric_notation() -> None:
    text = ENTRY.replace("year={2024}", 'year=2024, month=feb, note="A quoted value"')
    assert parse_bibliography(text)["one"]["title"] == "A {nested} title"
    assert citation_keys("AP@0.5 [@one; @two] and @three.") == {"one", "two", "three"}


@pytest.mark.parametrize(
    ("text", "message"),
    [
        (ENTRY + ENTRY, "duplicate citation key"),
        (ENTRY[:-1], "missing field separator"),
        (ENTRY[:-2], "unterminated BibTeX"),
        (ENTRY.replace("year={2024}", "year={2024}, year={2025}"), "duplicate field"),
        (ENTRY.replace("year={2024}", "year={unknown}"), "invalid year"),
        (ENTRY.replace("year={2024}", "note={unknown}"), "missing year"),
        (ENTRY.replace("year={2024}", "year={2024}, doi={invented}"), "invalid DOI"),
    ],
)
def test_malformed_records_fail(text: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_bibliography(text)


def test_shared_bibliography_resolves_and_rejects_missing_key(tmp_path: Path) -> None:
    bib = tmp_path / "references.bib"
    bib.write_text(ENTRY)
    paper = tmp_path / "paper.md"
    paper.write_text("One supported citation [@one].")
    assert verify(bib, [paper])[0] == 1
    paper.write_text("An unresolved citation [@missing].")
    with pytest.raises(ValueError, match="unresolved citations: missing"):
        verify(bib, [paper])


def test_duplicate_doi_rejected_even_with_different_keys() -> None:
    entry = ENTRY.replace("year={2024}", "year={2024}, doi={10.1234/example}")
    with pytest.raises(ValueError, match="duplicate DOI"):
        parse_bibliography(entry + entry.replace("{one,", "{two,"))
