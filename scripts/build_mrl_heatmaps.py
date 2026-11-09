"""Build layer-by-head MRL views and a complete prompt/step PDF appendix.

The input records retain each 2D pair's all-token MRL. Band cells average
these pair MRL values within frequency thirds of one RoPE axis.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / "tmp/matplotlib-mrl"))


def frequency_groups(root):
    import numpy as np
    import torch

    rope = torch.load(root / "rope.pt", map_location="cpu", weights_only=True)
    grid = rope["grid"]
    groups = {}
    for dimension, axis in enumerate(("t", "h", "w")):
        pairs = [pair for pair, name in enumerate(rope["pair_axes"]) if name == axis]
        unit_position = int(np.prod(grid[dimension + 1:]))
        frequencies = torch.atan2(rope["sin"][unit_position, pairs], rope["cos"][unit_position, pairs]).double().numpy()
        assert np.all((frequencies > 0) & (frequencies < np.pi))
        order = np.argsort(-frequencies, kind="stable")
        ordered = np.asarray(pairs)[order]
        bands = {}
        for name, band in zip(("high", "middle", "low"), np.array_split(ordered, 3)):
            angular = [float(frequencies[pairs.index(int(pair))]) for pair in band]
            bands[name] = {"pairs": band.tolist(), "frequency_min": min(angular), "frequency_max": max(angular)}
        groups[axis] = {
            "pairs": ordered.tolist(), "bands": bands,
            "radians_per_token": {str(pair): float(frequencies[pairs.index(pair)]) for pair in pairs}
        }
    return groups


def band_matrix(record, side, stage, pairs):
    import numpy as np
    values = np.asarray(record["mrl"][f"{side}_pair_mrl_{stage}"], dtype=np.float64)
    assert values.shape == (30, 12, 64)
    return values[:, :, pairs].mean(axis=2)


def draw_condition(record, axis, stage, groups, destination):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.family": "DejaVu Serif", "font.size": 10, "axes.titlesize": 11,
                         "axes.labelsize": 10, "xtick.labelsize": 8, "ytick.labelsize": 8,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    figure, axes = plt.subplots(2, 3, figsize=(10.8, 9.0), layout="constrained")
    for row, side in enumerate(("q", "k")):
        for column, (band, definition) in enumerate(groups[axis]["bands"].items()):
            ax = axes[row, column]
            pairs = definition["pairs"]
            matrix = band_matrix(record, side, stage, pairs)
            im = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto", interpolation="nearest", origin="upper")
            ax.set_title(f"{side.upper()} | {band} frequency | pairs {min(pairs)}-{max(pairs)}")
            ax.set_xlabel("Head (zero-based)")
            ax.set_ylabel("Layer (zero-based)")
            ax.set_xticks(range(12))
            ax.set_yticks([0, 5, 10, 15, 20, 25, 29])
    colorbar = figure.colorbar(im, ax=axes, shrink=.80, pad=.02, ticks=[0, .25, .5, .75, 1])
    colorbar.set_label("Mean of 2D-pair MRL values")
    figure.suptitle(f"{record['prompt_id']} | denoising step {record['step']:02d} | {axis} axis | {stage}-RoPE", fontsize=14)
    figure.get_layout_engine().set(rect=(0, .065, 1, .94))
    figure.text(.5, .025, "One prompt and step. Each cell is one layer/head, averaged only over the listed frequency pairs.\nEach pair MRL uses all 10,800 token vectors. Q/K and pre/post remain separate. Shared color scale 0-1.",
                ha="center", va="center", fontsize=9)
    target = destination / f"{record['prompt_id']}-step-{record['step']:03d}-{axis}-{stage}.png"
    figure.savefig(target, dpi=150, facecolor="white")
    plt.close(figure)


def appendix(records, groups, output):
    """One landscape A3 page per prompt/step/stage, with all three axes."""
    from io import BytesIO
    import numpy as np
    from PIL import Image
    from reportlab.lib.colors import HexColor
    from reportlab.lib.pagesizes import A3, landscape
    from reportlab.lib.utils import ImageReader
    from reportlab.pdfgen.canvas import Canvas
    from matplotlib import colormaps

    width, height = landscape(A3)
    canvas = Canvas(str(output), pagesize=(width, height), pageCompression=1)
    canvas.setTitle("Complete layer-head MRL heatmaps for every prompt and denoising step")
    palette = colormaps["viridis"]
    left, bottom, cell_width, cell_height = 66, 102, 168, 188
    for record in records:
        for stage in ("pre", "post"):
            label = f"{record['prompt_id']} / step {record['step']:02d} / {stage}-RoPE"
            bookmark = f"{record['prompt_id']}-{record['step']}-{stage}"
            canvas.bookmarkPage(bookmark)
            canvas.addOutlineEntry(label, bookmark, level=0)
            canvas.setFillColor(HexColor("#202a36"))
            canvas.setFont("Times-Bold", 19)
            canvas.drawString(45, height - 39, f"Layer x head MRL | {label}")
            canvas.setFont("Times-Roman", 11)
            canvas.drawString(45, height - 58, f"Conditional Wan2.1 | scheduler timestep {record['model_timestep']} | all 30 layers x 12 heads | one prompt and denoising step")
            for axis_index, axis in enumerate(("t", "h", "w")):
                y = bottom + (2 - axis_index) * 228
                canvas.setFont("Times-Bold", 12)
                canvas.drawString(18, y + cell_height / 2, axis)
                for side_index, side in enumerate(("q", "k")):
                    for band_index, (band, definition) in enumerate(groups[axis]["bands"].items()):
                        column = side_index * 3 + band_index
                        x = left + column * 183
                        pairs = definition["pairs"]
                        matrix = band_matrix(record, side, stage, pairs)
                        rgb = np.round(palette(matrix)[:, :, :3] * 255).astype("uint8")
                        # Nearest-neighbor enlargement keeps every 30 x 12 cell distinct.
                        image = Image.fromarray(rgb).resize((120, 300), resample=Image.Resampling.NEAREST)
                        buffer = BytesIO()
                        image.save(buffer, format="PNG")
                        canvas.drawImage(ImageReader(buffer), x, y, width=cell_width, height=cell_height)
                        canvas.setFont("Times-Bold", 9)
                        canvas.drawString(x, y + cell_height + 14, f"{side.upper()} {band} | pairs {min(pairs)}-{max(pairs)}")
                        canvas.setFont("Times-Roman", 8)
                        for head in range(12):
                            canvas.drawCentredString(x + (head + .5) * cell_width / 12, y - 11, str(head))
                        if axis_index == 2:
                            canvas.drawCentredString(x + cell_width / 2, y - 23, "Head")
                        for layer in (0, 5, 10, 15, 20, 25, 29):
                            canvas.drawRightString(x - 5, y + cell_height * (1 - (layer + .5) / 30) - 3, str(layer))
                        if column == 0:
                            canvas.saveState()
                            canvas.translate(x - 25, y + cell_height / 2)
                            canvas.rotate(90)
                            canvas.drawCentredString(0, 0, "Layer")
                            canvas.restoreState()
            for color_index in range(256):
                color = palette(color_index / 255)
                canvas.setFillColorRGB(*color[:3])
                canvas.rect(410 + color_index * 1.3, 60, 1.31, 9, stroke=0, fill=1)
            canvas.setFillColor(HexColor("#202a36"))
            canvas.setFont("Times-Roman", 9)
            canvas.drawString(397, 60, "0")
            canvas.drawString(749, 60, "1")
            canvas.drawCentredString(width / 2, 42, "Cell = mean of the listed 2D-pair MRL values. Each pair uses all 10,800 tokens at this one prompt/step/layer/head.")
            canvas.drawCentredString(width / 2, 27, "Frequency thirds are defined separately within t/h/w by actual angular frequency. Fixed 0-1 color scale. Layer/head indices are zero-based.")
            canvas.drawRightString(width - 35, 17, str(canvas.getPageNumber()))
            canvas.showPage()
    canvas.save()


def report_section(metadata, prefix=""):
    text = [
        "## 5. Layer-by-head MRL across frequency bands",
        "Each heatmap has layer on the y-axis and head on the x-axis. Q and K have separate panels. "
        "Each underlying pair MRL uses all 10,800 tokens at one prompt, denoising step, layer and head. "
        "Band cells are arithmetic means of these 2D-pair MRL values over the explicitly listed pair IDs.",
        "High, middle and low frequencies are nearly equal-count thirds sorted by the actual saved RoPE angular frequency, "
        "separately within t/h/w. Temporal groups contain 8/7/7 pairs, and height/width groups contain 7/7/7 pairs. "
        "The color scale is fixed at 0-1 across all views. The website also exposes individual-pair values, pre/post stages "
        "and click-to-select layer/head cells for all 80 prompt/step conditions.",
        f"[Complete heatmap appendix]({prefix}wan-mrl-heatmaps-complete.pdf). All 160 landscape A3 pages are indexed "
        "by prompt, step and stage. Each page contains all three axes, both Q/K, all 30 layers and 12 heads, and the three frequency groups.",
        "The following figures retain the fixed p03, step 20 example. Each cell preserves layer/head identity and averages "
        "only its stated frequency pairs. The accompanying tables pool 4 prompts x 20 steps x 30 layers x 12 heads "
        "= 28,800 band-mean cells per entry, separately by axis, Q/K and stage."
    ]
    for axis in ("t", "h", "w"):
        text.append(f"![{axis}-axis layer-by-head pair MRL]({prefix}mrl-heatmaps/figures/p03-step-020-{axis}-pre.png)")
        rows = ["| Axis | Band | Pair IDs | Q pre | K pre | Q post | K post |",
                "|---|---|---|---:|---:|---:|---:|"]
        for band, definition in metadata["groups"][axis]["bands"].items():
            pairs = definition["pairs"]
            values = [metadata["summary"][axis][stage][side][band]["mean"]
                      for stage in ("pre", "post") for side in ("q", "k")]
            numbers = " | ".join(f"{value:.4f}" for value in values)
            rows.append(f"| {axis} | {band} | {min(pairs)}-{max(pairs)} | {numbers} |")
        text.append("\n".join(rows))
    text.append(f"[Exact frequency groups, summary definitions and source hashes]({prefix}mrl-heatmaps/manifest.json). "
                f"Every prompt/step's full 30 x 12 x 64 pair-MRL array is retained under `{prefix}mrl-heatmaps/data/`.")
    return "\n\n".join(text)


def update_report(path, metadata, prefix=""):
    if not path.exists():
        return
    text = path.read_text()
    end = text.index("## Interpretation limits")
    marker = "## 5. Layer-by-head MRL across frequency bands"
    start = text.index(marker) if marker in text else end
    path.write_text(text[:start] + report_section(metadata, prefix) + "\n\n" + text[end:])


def main():
    import numpy as np
    from matplotlib import colormaps
    from matplotlib.colors import to_hex

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="output/full-sweep-wan-20261101")
    args = parser.parse_args()
    root = ROOT / args.root
    destination = root / "mrl-heatmaps"
    data_directory = destination / "data"
    figures = destination / "figures"
    data_directory.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    inputs = json.loads((destination / "inputs/manifest.json").read_text())
    assert hashlib.sha256((root / "rope.pt").read_bytes()).hexdigest() == inputs["rope_sha256"]
    groups = frequency_groups(root)
    records = []
    for item in inputs["conditions"]:
        source = gzip.decompress((destination / "inputs" / item["path"]).read_bytes())
        assert hashlib.sha256(source).hexdigest() == item["sha256"]
        record = json.loads(source)
        for axis in groups:
            assert sorted(record["pair_axes"][axis]) == sorted(groups[axis]["pairs"])
        target = data_directory / item["path"].removesuffix(".gz")
        target.write_bytes(source)
        records.append(record)
    assert len(records) == 80
    summary = {}
    for axis in groups:
        summary[axis] = {}
        for stage in ("pre", "post"):
            summary[axis][stage] = {}
            for side in ("q", "k"):
                summary[axis][stage][side] = {}
                for band, definition in groups[axis]["bands"].items():
                    values = np.stack([band_matrix(record, side, stage, definition["pairs"]) for record in records])
                    summary[axis][stage][side][band] = {
                        "mean": float(values.mean()), "fraction_below_0_8": float((values < .8).mean()),
                        "minimum": float(values.min()), "maximum": float(values.max()), "cells": int(values.size)
                    }
    metadata = {
        "schema": 1, "conditions": 80, "layers": 30, "heads": 12, "tokens": 10800,
        "groups": groups, "color_scale": [0, 1],
        "palette": [to_hex(colormaps["viridis"](index / 255)) for index in range(256)],
        "cell_definition": "Arithmetic mean of individual 2D-pair MRLs within a frequency third. Each pair MRL uses all 10,800 tokens at one prompt/step/layer/head. Individual-pair view uses the saved pair value directly.",
        "band_definition": "Sort actual angular frequencies descending separately within each axis, then split into three nearly equal pair-count groups named high, middle and low.",
        "summary_pooling": "Each summary mean/fraction pools 4 prompts x 20 steps x 30 layers x 12 heads = 28,800 band-mean cells, separately by axis, stage and Q/K.",
        "summary": summary, "source_inputs": inputs,
        "appendix_pages": 160, "appendix_layout": "One landscape A3 page per prompt/step/stage, with all three axes and Q/K high/middle/low panels.",
        "static_example": {"prompt_id": "p03", "step": 20, "figures": 6}
    }
    (destination / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
    fixed = next(record for record in records if record["prompt_id"] == "p03" and record["step"] == 20)
    for stage in ("pre", "post"):
        for axis in groups:
            draw_condition(fixed, axis, stage, groups, figures)
            print("Report example", axis, stage, "rendered", flush=True)
    output = ROOT / "output/pdf/wan-mrl-heatmaps-complete.pdf"
    output.parent.mkdir(parents=True, exist_ok=True)
    appendix(records, groups, output)
    update_report(root / "REPORT.md", metadata)
    if (root / "mrl-heatmaps/pattern-analysis.json").exists():
        from mrl_pattern_summary import load_summary, update_markdown
        update_markdown(root / "REPORT.md", load_summary(root))
    print(output)


if __name__ == "__main__":
    main()
