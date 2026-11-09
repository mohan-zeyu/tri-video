"""Shared, scope-labelled interpretation of the saved MRL heatmaps."""
import hashlib
import html
import json
from pathlib import Path


def load_summary(root):
    source = root / "mrl-heatmaps/pattern-analysis.json"
    statistics = json.loads(source.read_text())
    manifest_path = root / "mrl-heatmaps/manifest.json"
    manifest = json.loads(manifest_path.read_text())
    assert statistics["source_heatmap_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert statistics["groups"] == manifest["groups"]
    for axis in ("t", "h", "w"):
        for band in ("high", "middle", "low"):
            for side in ("q", "k"):
                for stage in ("pre", "post"):
                    actual = statistics["axes"][axis][band][side][stage]["mean"]
                    expected = manifest["summary"][axis][stage][side][band]["mean"]
                    assert abs(actual - expected) < 1e-12
    return summarize(statistics)


def summarize(statistics):
    axes = statistics["axes"]
    def mean(axis, band, side, stage="pre"):
        return axes[axis][band][side][stage]["mean"]

    pair_rows = []
    for axis in ("t", "h", "w"):
        ranges = []
        for band in ("high", "middle", "low"):
            pairs = statistics["groups"][axis]["bands"][band]["pairs"]
            ranges.append(f"{min(pairs)}-{max(pairs)}")
        pair_rows.append([axis, *ranges])
    pre_high = ", ".join(f"{mean(axis,'high','q'):.3f}/{mean(axis,'high','k'):.3f} ({axis})" for axis in axes)
    post_high = ", ".join(f"{mean(axis,'high','q','post'):.3f}/{mean(axis,'high','k','post'):.3f} ({axis})" for axis in axes)
    views = [axes[axis][band][side]["pre"] for axis in axes for band in ("high", "middle", "low") for side in ("q", "k")]
    within_ranges = [view["mean_within_layer_head_range"] for view in views]
    head_ranges = [view["head_range"] for view in views]
    early = [axes[axis]["low"]["q"]["pre"]["layer_blocks"]["L1_9"] for axis in axes]
    late = [axes[axis]["low"]["q"]["pre"]["layer_blocks"]["L20_29"] for axis in axes]
    example = statistics["targeted_checks"]["w"]["layer0_head3_high_Q_K_mean"]
    h_high = axes["h"]["high"]["q"]["pre"]["layer_blocks"]
    temporal_k = axes["t"]["high"]["k"]["pre"]["step_means"]
    layer0_first = [statistics["targeted_checks"][axis]["layer0_low_Q_K_step_means"][1][0] for axis in axes]
    layer0_last = [statistics["targeted_checks"][axis]["layer0_low_Q_K_step_means"][1][-1] for axis in axes]
    neighbor = [view["consecutive_step_map_r_median"] for view in views]
    endpoint = [axes[axis]["low"]["q"]["pre"]["step1_vs_step20_map_r"] for axis in axes]
    for axis in axes:
        for side in ("q", "k"):
            assert axes[axis]["band_order"][side]["high_largest_prompt_step_means"] == 80
        for band in ("middle", "low"):
            assert all(value < 0 for value in axes[axis][band]["k"]["pre"]["step20_minus_step1_by_prompt"])
    step_rows = []
    for step in (1, 10, 20):
        numbers = [axes["h"][band][side]["pre"]["step_means"][step - 1]
                   for band, side in (("high", "q"), ("high", "k"), ("low", "q"), ("low", "k"))]
        step_rows.append([str(step), *[f"{number:.3f}" for number in numbers]])
    findings = [
        {
            "title": "Frequency selectivity",
            "text": f"High frequencies have the greatest pre-RoPE mean in all 80 prompt/step conditions for Q and K on every axis. Pooled high-band Q/K means are {pre_high}.",
            "scope": "Rank bands after averaging 30 layers x 12 heads within each prompt/step. Pooled means average 4 prompts x 20 steps x 30 layers x 12 heads = 28,800 cells per axis/side."
        },
        {
            "title": "Positional dispersion",
            "text": f"After RoPE, high-band Q/K means fall to {post_high}. Low-band mean absolute changes stay below 0.00043. The low-band phase span reaches only 0.021 radians temporally and 0.062 spatially, supporting weak positional dispersion there.",
            "scope": "Compare pre/post at matched cells, then average 28,800 cells per axis/side. Phase spans use the highest low-band frequency times 11 temporal or 29 spatial coordinate increments."
        },
        {
            "title": "Depth",
            "text": f"Low-band Q rises from {min(early):.2f}-{max(early):.2f} in layers 1-9 to {min(late):.3f}-{max(late):.3f} in layers 20-29 across axes. High bands show a middle-layer dip. Height high-Q means are {h_high['L1_9']:.3f}, {h_high['L10_19']:.3f}, {h_high['L20_29']:.3f} for layers 1-9, 10-19, 20-29.",
            "scope": "Average all 4 prompts, all 20 steps, all 12 heads and the stated layer block, separately by axis, band and Q/K, before RoPE."
        },
        {
            "title": "Head heterogeneity and Q/K",
            "text": f"Mean within-layer head ranges are {min(within_ranges):.2f}-{max(within_ranges):.2f}. Head-index means across layers span {min(head_ranges):.2f}-{max(head_ranges):.2f}. One illustrative asymmetry is layer 0/head 3, whose width high-band Q/K means are {example[0]:.3f}/{example[1]:.3f}.",
            "scope": "First average 80 prompt/step conditions at each layer/head. Within-layer max-minus-min spans 12 heads, then averages 30 layers. Head-index means also average layers. The example holds layer 0/head 3 and width pairs 43-49 fixed. All values are pre-RoPE."
        },
        {
            "title": "Denoising",
            "text": f"Middle/low K means decrease from step 1 to 20 for every prompt. Pooled middle/low Q dips around steps 8-9 and recovers late. Spatial high bands strengthen, while temporal high-K falls from {temporal_k[0]:.3f} to {temporal_k[-1]:.3f}.",
            "scope": "The table gives pre-RoPE height-axis means over 4 prompts x 30 layers x 12 heads = 1,440 cells at each stated step, using high pairs 22-28 and low pairs 36-42. Per-prompt decline checks average 30 layers x 12 heads at steps 1 and 20.",
            "table": {"headers": ["Step", "High Q", "High K", "Low Q", "Low K"], "rows": step_rows}
        },
        {
            "title": "First-layer exception",
            "text": f"Layer 0 low-band K grows from {min(layer0_first):.2f}-{max(layer0_first):.2f} at step 1 to {min(layer0_last):.2f}-{max(layer0_last):.2f} at step 20 across axes. Later layers drive the pooled decline.",
            "scope": "Hold layer 0 and each endpoint step fixed. Average 4 prompts x 12 heads = 48 cells, separately by axis, before RoPE."
        },
        {
            "title": "Gradual map reorganization",
            "text": f"Neighboring steps have median MRL-map correlations of {min(neighbor):.3f}-{max(neighbor):.3f}. First/last low-Q maps correlate only {min(endpoint):.2f}-{max(endpoint):.2f}.",
            "scope": "At each step, average 4 prompts, then correlate the 360 layer/head MRL cells. Neighbor values are medians of 19 adjacent-step comparisons, separately for 18 axis/band/QK views. Endpoint values span the three low-Q axis views. All are pre-RoPE."
        }
    ]
    return {
        "title": "5.4 Heatmap patterns and averaging scopes",
        "definition": "Each pair R_f uses all 10,800 conditional video-token Q or K vectors in one 2D frequency pair at one prompt/step/layer/head. R_f = ||mean_tokens(x_f)|| / mean_tokens(||x_f||). A band cell equally averages its listed pair R_f values. Q/K and t/h/w remain separate. Layer/head IDs are zero-based and denoising steps are numbered 1-20.",
        "pair_table": {"headers": ["Axis", "High pair IDs", "Middle pair IDs", "Low pair IDs"], "rows": pair_rows},
        "findings": findings
    }


def html_section(summary):
    escape = html.escape
    def table(content):
        head = "".join(f"<th>{escape(cell)}</th>" for cell in content["headers"])
        rows = "".join("<tr>" + "".join(f"<td>{escape(cell)}</td>" for cell in row) + "</tr>" for row in content["rows"])
        return f'<div class="scroll"><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>'
    blocks = ['<section id="mrl-analysis"><h2>Findings across the complete sweep</h2>',
              f'<p>{escape(summary["definition"])}</p>', table(summary["pair_table"])]
    for finding in summary["findings"]:
        blocks.extend([f'<h3>{escape(finding["title"])}</h3>', f'<p>{escape(finding["text"])}</p>'])
        if "table" in finding:
            blocks.append(table(finding["table"]))
        blocks.append(f'<p class="note">{escape(finding["scope"])}</p>')
    blocks.append('<p class="small"><a href="mrl-heatmaps/pattern-analysis.json">Computed profiles and source definitions</a></p></section>')
    return "\n".join(blocks)


def update_markdown(path, summary, prefix=""):
    if not path.exists():
        return
    text = path.read_text()
    marker = "## 5.4 Heatmap patterns and averaging scopes"
    end = text.index("## Interpretation limits")
    start = text.index(marker) if marker in text else end
    blocks = [marker, summary["definition"]]
    def table(content):
        return "\n".join(["| " + " | ".join(content["headers"]) + " |",
                          "| " + " | ".join("---" for _ in content["headers"]) + " |",
                          *["| " + " | ".join(row) + " |" for row in content["rows"]]])
    blocks.append(table(summary["pair_table"]))
    for finding in summary["findings"]:
        blocks.extend([f"**{finding['title']}**", finding["text"]])
        if "table" in finding:
            blocks.append(table(finding["table"]))
        blocks.append(f"*{finding['scope']}*")
    blocks.append(f"[Computed profiles and source definitions]({prefix}mrl-heatmaps/pattern-analysis.json).")
    path.write_text(text[:start] + "\n\n".join(blocks) + "\n\n" + text[end:])
