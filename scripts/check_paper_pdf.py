"""Check PDF extraction, links and page bounds; visual review remains mandatory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pymupdf


def main() -> None:
    """Write an exact-PDF QA receipt without modifying the document."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--text", type=Path, required=True)
    args = parser.parse_args()
    document = pymupdf.open(args.pdf)
    pages, errors, texts = [], [], []
    uris: set[str] = set()
    internal_links = 0
    for number, page in enumerate(document, 1):
        text = page.get_text()
        texts.append(text)
        bounds = pymupdf.Rect(-1, -1, page.rect.width + 1, page.rect.height + 1)
        if "\ufffd" in text or not text.strip():
            errors.append(f"Page {number}: empty text or replacement glyph")
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    box = pymupdf.Rect(span["bbox"])
                    if not bounds.contains(box):
                        errors.append(f"Page {number}: text outside page: {span['text']}")
        for link in page.get_links():
            if link["kind"] in {pymupdf.LINK_GOTO, pymupdf.LINK_NAMED} and "page" in link:
                internal_links += 1
                if not 0 <= link["page"] < len(document):
                    errors.append(f"Page {number}: invalid internal link")
            elif link["kind"] == pymupdf.LINK_URI:
                uris.add(link["uri"])
            else:
                errors.append(f"Page {number}: unexpected local/unresolved link: {link}")
        pages.append(
            {
                "page": number,
                "text_characters": len(text),
                "embedded_images": len(page.get_images()),
                "links": len(page.get_links()),
                "rotation": page.rotation,
            }
        )
    receipt = {
        "pdf": args.pdf.as_posix(),
        "sha256": hashlib.sha256(args.pdf.read_bytes()).hexdigest(),
        "page_count": len(document),
        "pages": pages,
        "internal_links": internal_links,
        "external_uris": sorted(uris),
        "errors": errors,
        "status": "PASS" if not errors else "FAIL",
        "scope": "Automated checks only; see FINAL_SUBMISSION_AUDIT.md for visual inspection.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.text.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    args.text.write_text("\n\f\n".join(texts), encoding="utf-8")
    print(f"{receipt['status']}: {len(document)} pages; {len(errors)} automated errors")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
