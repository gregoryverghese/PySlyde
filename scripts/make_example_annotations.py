#!/usr/bin/env python
"""
Regenerate the synthetic example annotations shipped with PySlyde.

The example slide is ``CMU-1-Small-Region.svs`` from the OpenSlide test-data
collection (CC0-1.0, https://openslide.cs.cmu.edu/download/openslide-testdata/).
It carries no annotations of its own, so this script derives two classes of
*synthetic, computer-generated* regions from simple image processing:

* ``tissue``     - the largest foreground (non-background) component.
* ``epithelium`` - the largest haematoxylin-dense components inside the tissue.

These polygons are NOT pathologist annotations. They only exist so that every
annotation reader in PySlyde (GeoJSON, CSV, QuPath, ASAP, ImageJ and the custom
JSON format) can be exercised end-to-end on an openly licensed slide.

The same polygons are written in all six formats, so the formats are
interchangeable in tests, notebooks and documentation examples.

Usage
-----
    python scripts/make_example_annotations.py [--slide PATH] [--out DIR]

Without ``--slide`` the slide is downloaded (and checksum-verified) through
:func:`pyslyde.datasets.fetch_example_slide`.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import openslide
from skimage.color import rgb2hed

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "pyslyde" / "datasets" / "annotations"

# Polygon simplification tolerance, in level-0 pixels.
SIMPLIFY_EPS = 6.0
N_EPITHELIUM = 3


def _contours_to_polygons(contours, scale: float) -> list[list[list[int]]]:
    polys = []
    for c in contours:
        approx = cv2.approxPolyDP(c, SIMPLIFY_EPS / scale, closed=True)
        pts = (approx.reshape(-1, 2).astype(float) * scale).round().astype(int)
        if len(pts) >= 3:
            polys.append(pts.tolist())
    return polys


def derive_polygons(slide_path: Path) -> dict[str, list[list[list[int]]]]:
    """Derive the synthetic ``tissue`` and ``epithelium`` polygons."""
    slide = openslide.OpenSlide(str(slide_path))
    w, h = slide.dimensions
    scale = 2.0  # analyse at half resolution
    img = np.asarray(slide.get_thumbnail((int(w / scale), int(h / scale))).convert("RGB"))
    scale = w / img.shape[1]

    # --- tissue: Otsu on saturation, largest component -----------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    sat = cv2.GaussianBlur(hsv[:, :, 1], (5, 5), 0)
    _, tissue = cv2.threshold(sat, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    tissue = cv2.morphologyEx(tissue, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    contours, _ = cv2.findContours(tissue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
    tissue_mask = np.zeros_like(tissue)
    cv2.drawContours(tissue_mask, contours, -1, 255, -1)
    tissue_polys = _contours_to_polygons(contours, scale)

    # --- epithelium: haematoxylin-rich regions inside the tissue ------------
    # Colour deconvolution (Ruifrok & Johnston) -> haematoxylin channel.
    hed = rgb2hed(img)
    haem = hed[:, :, 0]
    haem = cv2.GaussianBlur(haem.astype(np.float32), (0, 0), 6)
    vals = haem[tissue_mask > 0]
    thr = np.percentile(vals, 65)
    dark = ((haem > thr) & (tissue_mask > 0)).astype(np.uint8) * 255
    dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:N_EPITHELIUM]
    epi_polys = _contours_to_polygons(contours, scale)

    return {"tissue": tissue_polys, "epithelium": epi_polys}


def _closed(poly):
    return poly + [poly[0]] if poly[0] != poly[-1] else poly


def write_all(polys: dict[str, list[list[list[int]]]], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)

    # GeoJSON FeatureCollection (label in properties.label)
    features = []
    for label, items in polys.items():
        for p in items:
            features.append({
                "type": "Feature",
                "properties": {"label": label},
                "geometry": {"type": "Polygon", "coordinates": [_closed(p)]},
            })
    (out / "example.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=1)
    )

    # QuPath export: bare list of features with properties.classification.name
    qp = []
    for label, items in polys.items():
        for p in items:
            qp.append({
                "type": "Feature",
                "geometry": {"type": "Polygon", "coordinates": [_closed(p)]},
                "properties": {"objectType": "annotation",
                               "classification": {"name": label}},
            })
    (out / "example_qupath.json").write_text(json.dumps(qp, indent=1))

    # Custom PySlyde JSON: {label: {polygon_id: [{x, y}, ...]}}
    custom = {
        label: {str(i): [{"x": x, "y": y} for x, y in p] for i, p in enumerate(items)}
        for label, items in polys.items()
    }
    (out / "example_custom.json").write_text(json.dumps(custom, indent=1))

    # CSV: label, polygon_id, vertex_id, x, y
    with (out / "example.csv").open("w", newline="") as f:
        wr = csv.writer(f, lineterminator="\n")
        wr.writerow(["label", "polygon_id", "vertex_id", "x", "y"])
        for label, items in polys.items():
            for i, p in enumerate(items):
                for j, (x, y) in enumerate(p):
                    wr.writerow([label, f"{label}_{i}", j, x, y])

    # ASAP XML
    root = ET.Element("ASAP_Annotations")
    anns = ET.SubElement(root, "Annotations")
    n = 0
    for label, items in polys.items():
        for p in items:
            a = ET.SubElement(anns, "Annotation", Name=f"Annotation {n}",
                              Type="Polygon", PartOfGroup=label, Color="#F4FA58")
            coords = ET.SubElement(a, "Coordinates")
            for k, (x, y) in enumerate(p):
                ET.SubElement(coords, "Coordinate", Order=str(k), X=str(x), Y=str(y))
            n += 1
    groups = ET.SubElement(root, "AnnotationGroups")
    for label in polys:
        ET.SubElement(groups, "Group", Name=label, PartOfGroup="None", Color="#F4FA58")
    ET.indent(root)
    ET.ElementTree(root).write(out / "example_asap.xml", encoding="utf-8",
                               xml_declaration=True)

    # ImageJ-style XML: Annotation Name=label / Regions / Region / Vertices / Vertex
    root = ET.Element("Annotations")
    for a_idx, (label, items) in enumerate(polys.items()):
        a = ET.SubElement(root, "Annotation", Id=str(a_idx + 1), Name=label)
        regions = ET.SubElement(a, "Regions")
        for r_idx, p in enumerate(items):
            r = ET.SubElement(regions, "Region", Id=str(r_idx + 1))
            verts = ET.SubElement(r, "Vertices")
            for x, y in p:
                ET.SubElement(verts, "Vertex", X=str(x), Y=str(y), Z="0")
    ET.indent(root)
    ET.ElementTree(root).write(out / "example_imagej.xml", encoding="utf-8",
                               xml_declaration=True)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--slide", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)

    slide_path = args.slide
    if slide_path is None:
        sys.path.insert(0, str(REPO_ROOT))
        from pyslyde.datasets import fetch_example_slide
        slide_path = fetch_example_slide()

    polys = derive_polygons(slide_path)
    write_all(polys, args.out)
    for k, v in polys.items():
        print(f"{k}: {len(v)} polygon(s), {sum(len(p) for p in v)} vertices")
    print(f"Wrote 6 annotation files to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
