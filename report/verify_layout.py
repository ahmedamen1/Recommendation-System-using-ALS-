"""Check PDF pagination, caption order, subsection placement, and rendering assets.

Usage: python verify_layout.py [path/to/report.pdf]
Requires PyMuPDF. This does not rerun model training or modify the report.
"""
import json
from pathlib import Path
import sys
import unicodedata

import pymupdf


PLACEMENTS = {
    "Figure 1.": ("2. System Overview and Architecture", "3. Dataset and Exploratory Analysis"),
    "Figure 2.": ("3.2. Distributions", "4. Methodology"),
    "Figure 3.": ("3.2. Distributions", "4. Methodology"),
    "Figure 4.": ("5.3. Bias-History Correction", "6. Results and Discussion"),
    "Figure 5.": ("6.1. Biases and Factors", "6.2. Capacity and Regularization"),
    "Figure 6.": ("6.2. Capacity and Regularization", "6.3. Genre Prior and Cold Start"),
    "Figure 7.": ("6.3. Genre Prior and Cold Start", "6.4. Two-Dimensional Fits"),
    "Figure 8.": ("6.4. Two-Dimensional Fits", "7. Recommendation Application"),
    "Figure 9.": ("6.4. Two-Dimensional Fits", "7. Recommendation Application"),
    "Table 1.": ("3.1. Data and Sparse Storage", "3.2. Distributions"),
    "Table 2.": ("5.1. Compiled Updates", "5.2. Evaluation Protocol"),
    "Table 3.": ("6.1. Biases and Factors", "6.2. Capacity and Regularization"),
    "Table 4.": ("6.2. Capacity and Regularization", "6.3. Genre Prior and Cold Start"),
    "Table 5.": ("6.2. Capacity and Regularization", "6.3. Genre Prior and Cold Start"),
    "Table 6.": ("6.3. Genre Prior and Cold Start", "6.4. Two-Dimensional Fits"),
    "Table 7.": ("7.1. New-User Recommendations", "7.2. Movie-Vector Norms"),
    "Table 8.": ("7.2. Movie-Vector Norms", "7.3. Independent Checks"),
}


def reading_position(doc, page_index, rect):
    """Order a two-column PDF by page, then left/right column, then height."""
    return page_index, int(rect.x0 >= doc[page_index].rect.width / 2), rect.y0


def heading_position(doc, title):
    found = []
    for i, page in enumerate(doc):
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"])
                text = " ".join(unicodedata.normalize("NFKC", text).split())
                if text == title or text.startswith(title + " "):
                    found.append((i, pymupdf.Rect(line["bbox"])))
    assert len(found) == 1, f"Missing or duplicated section heading: {title}"
    return reading_position(doc, *found[0])


def verify(path):
    with pymupdf.open(path) as doc:
        assert len(doc) <= 8, "Delivered report must not exceed eight pages"
        references = [(i, line["bbox"][1], line["bbox"][0]) for i, page in enumerate(doc)
                      for block in page.get_text("dict")["blocks"]
                      for line in block.get("lines", [])
                      if "".join(span["text"] for span in line["spans"]).strip() == "References"]
        assert len(references) == 1, "Expected one References heading"
        ref_page, ref_y, ref_x = references[0]
        ref_position = ref_page, int(ref_x >= doc[ref_page].rect.width / 2), ref_y
        locations = {}
        section_placements = {}
        for kind, count in (("Figure", 9), ("Table", 8)):
            previous_position = None
            for number in range(1, count + 1):
                label = f"{kind} {number}."
                found = [(i, rect) for i, page in enumerate(doc)
                         for rect in page.search_for(label)]
                assert len(found) == 1, f"Missing or duplicated caption: {label}"
                page_index, rect = found[0]
                position = reading_position(doc, page_index, rect)
                assert position < ref_position, f"{label} appears after the references"
                if previous_position is not None:
                    assert previous_position < position, f"{label} is out of numerical order"
                previous_position = position
                section, next_section = PLACEMENTS[label]
                start = heading_position(doc, section)
                end = heading_position(doc, next_section)
                assert start < position < end, \
                    f"{label} is outside {section}; rendered on page {page_index + 1}"
                locations[label] = page_index + 1
                section_placements[label] = section
        for page in doc:
            assert not page.get_images(), "Report figures must remain vector graphics"
            assert "??" not in page.get_text(), "Unresolved cross-reference"
            for font in page.get_fonts(full=True):
                assert font[2] != "Type3", "Unexpected bitmap font"
            # Rasterize every page as an additional malformed-resource check.
            assert page.get_pixmap(matrix=pymupdf.Matrix(.5, .5)).width > 0
        return {"status": "passed", "pages": len(doc), "references_page": ref_page + 1,
                "all_captions_before_references": True,
                "all_captions_in_correct_sections": True,
                "caption_pages": locations, "caption_sections": section_placements}


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).with_name("aamin_AMLAS_draft.pdf")
    print(json.dumps(verify(target), indent=2))
