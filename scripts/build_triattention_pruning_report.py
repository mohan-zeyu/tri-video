"""Create the paired Wan query-block pruning report, gallery and figures."""

import argparse
import html
import json
from pathlib import Path
import subprocess
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage, ImageDraw, ImageFont
from build_center_benchmark_report import LABELS, ORDER, SCOPES, number, interval

ARMS = ("baseline", "tri50", "tri75")
NAMES = {"baseline":"Baseline", "tri50":"50% retained", "tri75":"75% retained"}
COLORS = {"baseline":"#31566f", "tri50":"#b77b32", "tri75":"#448572"}


def findings(data):
    return [
        "75% retention is closer to baseline than 50% in all six continuous quality/alignment dimensions. Its mean aesthetics, imaging quality and text-video consistency are lower than baseline by 0.0103, 0.0403 and 0.0056. At 50%, the corresponding losses are 0.0564, 0.0824 and 0.0281. Each comparison uses the same 100 prompt/seed pairs, after official per-video frame aggregation.",
        "Original-vector pruning substantially improves aesthetics, imaging quality and text alignment over both earlier Q/K mean-replacement arms. Residual degradation remains, especially in imaging quality. The paired intervals below quantify uncertainty across prompts.",
        "At 75% retention, the paired 95% intervals include zero for aesthetics [-0.0281, +0.0067] and text-video consistency [-0.0132, +0.0016]. The imaging-quality interval is entirely below zero [-0.0646, -0.0172]. These are prompt-bootstrap intervals for this 100-prompt subset. At 50%, intervals are below zero in all six continuous dimensions.",
        "Dynamic degree rises from 0.33 to 0.53 at 50% retention and 0.44 at 75%. This is the fraction of 100 videos classified as dynamic by the official flow threshold. Increased motion incidence alone does not establish better quality, particularly when smoothness and appearance scores decrease.",
        "Recorded mean generation-plus-decode time is 15.41 s for baseline, 20.57 s for 50% and 22.38 s for 75%. Ranking/gathering and four attention calls per module make this implementation slower. The baseline timings largely come from the preceding run, so these are descriptive measurements across runs. The experiment establishes quality at reduced attention-bank sizes, with full projections still materialized.",
    ]


def method_paragraphs(data):
    return [
        "Wan2.1-T2V-1.3B-Diffusers generates 45 frames at 480 x 480 and 16 fps, with 20 denoising steps, CFG 5 and an empty negative prompt. The grid contains 12 temporal x 30 height x 30 width positions, totaling 10,800 tokens. All 30 layers, 12 heads and both CFG branches receive the pruning policy. Native Q/K normalization, 3D RoPE, softmax and output projection run through the original Wan processor.",
        "Calibrate Q centers and mean 2D-pair norms on four separate prompts, each using seed 0 and a full 20-step native trajectory. Each statistic pools 4 prompts x 20 steps x 10,800 tokens, totaling 864,000 tokens per branch/layer. CFG branches, layers, heads and individual pairs remain separate. The evaluation prompts are excluded. The resulting Q means and mean norms are shared across denoising steps during ranking.",
        "For each actual key and each of all 64 pairs, use the calibrated Q center for a trigonometric ranking term and add (mean Q pair norm - Q center norm) x actual K pair norm. Average the temporal trigonometric term over the actual three query-frame positions in the current block. The ranking proxy sets height and width relative offsets to zero. Native attention uses the actual full 3D positions of Q and retained K.",
        "Divide the full-video queries into four three-frame blocks. Protect every key in the current live query block and the first sink frame. Their union contains 2,700 tokens for the first block and 3,600 for later blocks. Independently rank remaining keys for each layer/head/CFG forward. At 50% retention select 5,400 keys per bank, with 2,700/1,800 ranked keys for first/later blocks. At 75% select 8,100, with 5,400/4,500 ranked keys. Gather original post-RoPE K and original V. Actual Q vectors are used directly.",
        "Use the same 100 distinct category prompts and fixed per-prompt seeds as the preceding benchmark. Each seed is shared across the three arms. Ninety-nine baseline videos are reused with original metadata and hashes. The first baseline is regenerated with the full-retention controller and its final latent tensor must match the original native baseline exactly.",
        "Each VBench score first follows its official within-video frame aggregation, then averages 100 video scores separately per arm. Paired changes subtract the matching prompt/seed baseline. Confidence intervals resample 100 paired prompt units, retaining all arms together, 10,000 times with seed 20261109 and percentile 95% bounds. Tokens, heads, layers and denoising steps are kept within their video during this bootstrap.",
    ]


def scope_paragraphs():
    return [
        "This is a query-block adaptation of TriAttention scoring and original-K/V selection for joint full-video Wan diffusion. Its current-query positions and live-block protection adapt the causal method to the bidirectional model. All 64 pairs participate in ranking. Each query block uses a selected K/V bank, with original Q and retained native vectors.",
        "The preliminary v000 pilot transferred fixed last-frame protection and future-frame probes directly from LongLive. Middle-frame queries could then lose their own keys. The final protocol protects each live block and averages over its actual temporal query positions. Preliminary global-protection outputs are stored separately and excluded from this result table. The first prompt also served as an implementation/protection pilot, and remains included among the 100 reported prompts.",
        "Calibration pools across the four calibration prompts and 20 steps before forming a Q center per branch/layer/head/pair. It captures a shared query distribution. Actual per-token Q remains available for attention at every denoising step. The spatial-zero ranking proxy is an approximation to the full 3D attention pattern.",
        "Retention denotes keys per query block and head. Full Q/K/V projections are materialized before bank selection. Timing measures this Python implementation's generation plus VAE decoding and includes ranking, gathering and four attention calls per self-attention module. It describes the implemented runtime and its attention-operand reduction.",
        "This study uses 100 official category prompts, one seed per prompt and seven VBench dimensions in custom-input evaluation. The complete official VBench protocol has its own dimension-specific prompt lists and seed multiplicities. PDF examples use eight prompts fixed before inspecting final scores and decoded frame index 22 of 45.",
    ]


def figures(root, data):
    folder = root / "figures";folder.mkdir(exist_ok=True)
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":170})
    dimensions = [d for d in ORDER if d in data["dimensions"]]
    fig,ax=plt.subplots(figsize=(8,4.2));positions=np.arange(len(dimensions))
    for offset,arm in ((-.12,"tri50"),(.12,"tri75")):
        items=[data["dimensions"][d]["paired"][arm+"_minus_baseline"] for d in dimensions]
        means=np.array([v["mean"] for v in items]);bounds=np.array([v["ci95"] for v in items])
        ax.errorbar(means,positions+offset,xerr=[means-bounds[:,0],bounds[:,1]-means],fmt="o",capsize=3,color=COLORS[arm],label=NAMES[arm])
    ax.axvline(0,color="#78848c",lw=1);ax.set_yticks(positions,[LABELS[d] for d in dimensions]);ax.invert_yaxis()
    ax.set_xlabel("Paired score change from native baseline (95% prompt-bootstrap CI)");ax.grid(axis="x",alpha=.15)
    ax.legend(loc="upper left",bbox_to_anchor=(0,1.16),ncol=2,frameon=False);fig.tight_layout();fig.savefig(folder/"paired-deltas.png");plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(9,3.1))
    labels=["Native","Mean all","Mean high","Prune50%","Prune75%"]
    colors=[COLORS['baseline'],'#ba5055','#cf9b43',COLORS['tri50'],COLORS['tri75']]
    for ax,dimension in zip(axes,("aesthetic_quality","imaging_quality","overall_consistency")):
        previous=data["previous_replacement"][dimension]["means"];current=data["dimensions"][dimension]["means"]
        ax.bar(np.arange(5),[current['baseline'],previous['all'],previous['high'],current['tri50'],current['tri75']],color=colors)
        ax.set_xticks(np.arange(5),labels,rotation=35,ha="right",fontsize=8);ax.set_title(LABELS[dimension],fontsize=10);ax.grid(axis="y",alpha=.15)
    fig.tight_layout();fig.savefig(folder/"comparison-with-replacement.png");plt.close(fig)
    stills=root/"stills";stills.mkdir(exist_ok=True)
    font=ImageFont.truetype(str(Path(matplotlib.get_data_path())/"fonts/ttf/DejaVuSans.ttf"),23)
    for prompt in [p for p in data["config"]["prompts"] if p["audit"]]:
        canvas=PILImage.new('RGB',(1440,524),'white');draw=ImageDraw.Draw(canvas)
        for i,arm in enumerate(ARMS):
            image=stills/f"{prompt['id']}-{arm}.png"
            if not image.exists():subprocess.run(["ffmpeg","-hide_banner","-loglevel","error","-ss","1.375","-i",str(root/"videos"/arm/f"{prompt['id']}.mp4"),"-frames:v","1","-y",str(image)],check=True)
            canvas.paste(PILImage.open(image),(480*i,44));draw.text((480*i+12,10),NAMES[arm],font=font,fill=COLORS[arm])
        canvas.save(folder/f"{prompt['id']}-comparison.png")


def score_rows(data):
    rows=[]
    for dimension in ORDER:
        if dimension not in data["dimensions"]:continue
        d=data["dimensions"][dimension]
        rows.append([LABELS[dimension],*[number(d['means'][a]) for a in ARMS],interval(d['paired']['tri50_minus_baseline']),interval(d['paired']['tri75_minus_baseline'])])
    return rows


def report_markdown(root,data):
    lines=["# TriAttention-style pruning in full-video Wan","","9 November 2026 | 100 paired prompts | 50% and 75% K/V banks","",
        "## Results","","Each mean uses 100 paired videos. Changes are absolute score differences with paired 95% bootstrap intervals.","",
        "| Dimension | Baseline | 50% retained | 75% retained | 50% minus baseline, 95% CI | 75% minus baseline, 95% CI |","|---|---:|---:|---:|---|---|"]
    lines += ['| '+' | '.join(row)+' |' for row in score_rows(data)]
    lines += ["", "## Findings", "", *sum(([p, ""] for p in findings(data)), [])]
    lines += ["","![Paired changes](figures/paired-deltas.png)","","## Native representations and calibration","",*sum(([p,""] for p in method_paragraphs(data)),[]),
        "## Comparison with earlier mean replacement","","The figure compares the same 100 prompts and seeds across the previous replacement trials and the present pruning trials. Each plotted entry is a mean of official per-video scores.","",
        "![Earlier replacement and current pruning](figures/comparison-with-replacement.png)","","## Exact VBench populations",""]
    for d in ORDER:
        if d in data['dimensions']:lines += ['- **'+LABELS[d]+'**. '+SCOPES[d],""]
    lines += ["## Adaptation and verification","",*sum(([p,""] for p in scope_paragraphs()),[]),
        "All 300 video hashes and frame/fps/size metadata pass. The 200 pruning runs share one source-code version and calibration hash, and each covers 1,200 layer/step/branch self-attention dispatches. Four query banks are evaluated within each pruned dispatch.","",
        "The selection check compares the factored implementation against the pinned official scorer for all four query blocks, 12 heads and both budgets. It checks exact preservation of gathered K/V coordinates and every live query key. The full-retention v000 control matches the original native final latent tensor exactly.","",
        "## Sources","","[Official LongLive scorer](https://github.com/WeianMao/triattention/blob/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b/longlive/kv_compression.py#L430-L467), [original K/V gathering](https://github.com/WeianMao/triattention/blob/a4bc3c8f709db60f016ef42c3feb290fd0c00c1b/longlive/kv_compression.py#L731-L741)","",
        "[Official VBench](https://github.com/Vchitect/VBench). The numerical source is summary.json, with all 700 paired per-prompt rows in per-prompt-results.csv.",""]
    (root/'REPORT.md').write_text('\n'.join(lines))


def report_html(root,data):
    esc=html.escape
    table=''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>' for row in score_rows(data))
    paragraphs=''.join('<p>'+esc(p)+'</p>' for p in method_paragraphs(data))
    limits=''.join('<p>'+esc(p)+'</p>' for p in scope_paragraphs())
    conclusions=''.join('<p>'+esc(p)+'</p>' for p in findings(data))
    scope=''.join('<h3>'+LABELS[d]+'</h3><p>'+esc(SCOPES[d])+'</p>' for d in ORDER if d in data['dimensions'])
    encoded=json.dumps(data,separators=(',',':')).replace('<','\\u003c')
    doc='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>TriAttention-style Wan pruning</title><style>
*{box-sizing:border-box}body{margin:0;background:#f5f6f8;color:#20313d;font:16px/1.65 system-ui,sans-serif}main{max-width:1160px;margin:auto;padding:38px 24px}h1{font-size:35px;line-height:1.2}h2{margin-top:40px;border-bottom:1px solid #ccd5db;padding-bottom:8px}a{color:#285a80}p{max-width:1030px}.scope{font-size:14px;color:#536571}.card{background:white;border:1px solid #d6dfe5;border-radius:9px;padding:22px;margin:20px 0}.scroll{overflow:auto}table{border-collapse:collapse;width:100%;font-size:14px}td,th{text-align:left;border-bottom:1px solid #d7dfe5;padding:9px;font-variant-numeric:tabular-nums}img{max-width:100%;display:block;margin:20px auto}select,button{font:inherit;padding:8px;margin:0 8px 8px 0;max-width:100%}#prompt{width:760px}.videos{display:grid;grid-template-columns:repeat(3,1fr);gap:12px}video{width:100%;background:#172129}h3{font-size:18px}@media(max-width:760px){main{padding:22px 14px}.card{padding:14px}.videos{grid-template-columns:1fr}h1{font-size:28px}}</style><main>
<h1>TriAttention-style pruning in full-video Wan</h1><p class="scope">100 paired prompts · actual Q and selected original K/V · 50% and 75% keys per query bank · 9 November 2026</p>
<p><a href="report.pdf">Detailed PDF</a> · <a href="REPORT.md">Markdown report</a> · <a href="summary.json">Full numerical results</a> · <a href="per-prompt-results.csv">All per-prompt scores</a> · <a href="config.json">Prompts and exact settings</a> · <a href="selection-verification.json">Selection checks</a></p>
<h2>Paired VBench results</h2><p class="scope">Means over 100 per-video scores per arm. Intervals resample the 100 paired prompt/seed units 10,000 times. Score changes are absolute differences.</p><div class="scroll"><table><thead><tr><th>Dimension</th><th>Baseline</th><th>50% retained</th><th>75% retained</th><th>50% − baseline, 95% CI</th><th>75% − baseline, 95% CI</th></tr></thead><tbody>__ROWS__</tbody></table></div>
<img src="figures/paired-deltas.png" alt="Paired score changes and confidence intervals"><h2>Comparison with direct mean replacement</h2><p class="scope">The same 100 prompts and seeds in every arm, using each metric's official frame aggregation.</p><img src="figures/comparison-with-replacement.png" alt="Native baseline, earlier mean replacement and current pruning">
<h2 id="gallery">Explore all 300 videos</h2><div class="card"><select id="category" aria-label="Category"><option value="">All categories</option></select><select id="prompt" aria-label="Prompt"></select><p id="prompt-text"></p><p id="seed" class="scope"></p><button id="play">Play all</button><button id="pause">Pause all</button><button id="restart">Restart all</button><div class="videos"><div><h3>Baseline</h3><video id="baseline" controls muted loop playsinline preload="metadata"></video></div><div><h3>50% retained</h3><video id="tri50" controls muted loop playsinline preload="metadata"></video></div><div><h3>75% retained</h3><video id="tri75" controls muted loop playsinline preload="metadata"></video></div></div><div class="scroll"><table id="selected-scores"></table></div></div>
<h2>What is averaged and what is retained</h2>__METHOD__<h2>Adaptation and verification</h2>__LIMITS__<p>The actual native full-retention v000 trajectory matches the original baseline exactly. Every live query's key is protected. Scoring is independently checked against the pinned official code; selected K/V values are preserved exactly.</p><details><summary>Exact evaluator populations</summary>__SCOPES__</details>
<p><a href="https://github.com/WeianMao/triattention">Official TriAttention code</a> · <a href="https://github.com/Vchitect/VBench">Official VBench</a></p>
<script type="application/json" id="dataset">__DATA__</script><script>
const data=JSON.parse(document.querySelector('#dataset').textContent),arms=['baseline','tri50','tri75'],labels=__LABELS__,order=__ORDER__;
const category=document.querySelector('#category'),prompt=document.querySelector('#prompt'),videos=arms.map(a=>document.getElementById(a));
for(const c of [...new Set(data.config.prompts.map(p=>p.category))].sort()){const o=document.createElement('option');o.value=c;o.textContent=c;category.append(o)}
function populate(){prompt.replaceChildren();for(const p of data.config.prompts.filter(p=>!category.value||p.category===category.value)){const o=document.createElement('option');o.value=p.id;o.textContent=p.id+' · '+p.category+' · '+p.text;prompt.append(o)}update()}
function update(){const p=data.config.prompts.find(p=>p.id===prompt.value);document.querySelector('#prompt-text').textContent=p.text;document.querySelector('#seed').textContent='Seed '+p.seed+' · 20 steps · 30 layers × 12 heads · both CFG branches';for(let i=0;i<3;i++){videos[i].pause();videos[i].src='videos/'+arms[i]+'/'+p.id+'.mp4';videos[i].load()}const table=document.querySelector('#selected-scores');table.replaceChildren();const header=document.createElement('tr');for(const s of ['Per-video score','Baseline','50% retained','75% retained']){const th=document.createElement('th');th.textContent=s;header.append(th)}table.append(header);for(const d of order){if(!data.dimensions[d])continue;const tr=document.createElement('tr');for(const s of [labels[d],...arms.map(a=>data.dimensions[d].per_prompt[p.id][a].toFixed(4))]){const td=document.createElement('td');td.textContent=s;tr.append(td)}table.append(tr)}}
category.onchange=populate;prompt.onchange=update;document.querySelector('#play').onclick=()=>{const t=videos[0].currentTime;for(const v of videos){v.currentTime=t;v.play().catch(()=>{})}};document.querySelector('#pause').onclick=()=>videos.forEach(v=>v.pause());document.querySelector('#restart').onclick=()=>{for(const v of videos){v.currentTime=0;v.play().catch(()=>{})}};populate();</script></main></html>'''
    doc=doc.replace('<h2>Comparison with direct mean replacement</h2>', '<h2>Findings</h2>'+conclusions+'<h2>Comparison with direct mean replacement</h2>')
    mapping={'__ROWS__':table,'__METHOD__':paragraphs,'__LIMITS__':limits,'__SCOPES__':scope,'__DATA__':encoded,'__LABELS__':json.dumps(LABELS),'__ORDER__':json.dumps(ORDER)}
    import re
    doc=re.sub('|'.join(mapping),lambda m:mapping[m.group()],doc)
    (root/'index.html').write_text(doc)


def report_pdf(root,data,path):
    from reportlab.lib import colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Image,Table,TableStyle,Spacer,PageBreak
    body=ParagraphStyle('body',fontName='Helvetica',fontSize=9.4,leading=12.6,spaceAfter=8)
    small=ParagraphStyle('small',parent=body,fontSize=8.2,leading=10.8)
    head=ParagraphStyle('head',parent=body,fontName='Helvetica-Bold',fontSize=15,leading=19,spaceAfter=12)
    sub=ParagraphStyle('sub',parent=body,fontName='Helvetica-Bold',fontSize=11,leading=14,spaceBefore=8)
    title=ParagraphStyle('title',parent=head,fontSize=24,leading=29,spaceAfter=15)
    story=[]
    def p(text,style=body):story.append(Paragraph(html.escape(text),style))
    def image(name,maxheight):
        f=root/'figures'/name;im=PILImage.open(f);w=503;h=w*im.height/im.width
        if h>maxheight:w*=maxheight/h;h=maxheight
        story.append(Image(str(f),width=w,height=h))
    def table(rows,widths,size=8):
        style=ParagraphStyle('cell',parent=small,fontSize=size,leading=size+2,spaceAfter=0)
        t=Table([[Paragraph(html.escape(str(v)),style) for v in row] for row in rows],colWidths=widths,repeatRows=1)
        t.setStyle(TableStyle([('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),6),('BOTTOMPADDING',(0,0),(-1,-1),6),('BACKGROUND',(0,0),(-1,0),colors.HexColor('#edf2f5')),('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#31566f')),('LINEBELOW',(0,-1),(-1,-1),.5,colors.HexColor('#abbac3'))]));story.append(t);story.append(Spacer(1,10))
    p('TriAttention-style pruning\nin full-video Wan',title)
    p('100 paired prompts | 50% and 75% K/V banks | 9 November 2026',small)
    p('1. Benchmark results',head)
    p('Each row averages 100 videos per arm, with one matching prompt/seed. The method uses actual Q and selected original K/V after native 3D RoPE.')
    table([['Dimension','Baseline','50% retained','75% retained']]+[[r[0],r[1],r[2],r[3]] for r in score_rows(data)],[188,105,105,105],9)
    for dimension in ('aesthetic_quality','imaging_quality','overall_consistency'):
        item=data['dimensions'][dimension]
        p(LABELS[dimension],sub)
        p('50% minus baseline '+interval(item['paired']['tri50_minus_baseline'])+'. 75% minus baseline '+interval(item['paired']['tri75_minus_baseline'])+'. Intervals use 10,000 paired prompt bootstrap samples.',small)
    p('The calibrated Q center is used for ranking. Actual per-token Q and selected native K/V perform attention. The two earlier mean-replacement arms are shown separately in the comparison figure.',small)
    story.append(PageBreak());p('2. Method and exact averaging',head)
    for paragraph in method_paragraphs(data):p(paragraph)
    p('Attention-bank budgets',sub)
    table([['Arm','Keys per head/bank','Protected live/sink union','Ranked keys, first/later blocks'],['Baseline','10,800','Full native key set','Full native key set'],['50%','5,400','2,700 / 3,600','2,700 / 1,800'],['75%','8,100','2,700 / 3,600','5,400 / 4,500']],[68,110,150,175],8)
    story.append(PageBreak());p('3. Paired changes and earlier trials',head)
    image('paired-deltas.png',310)
    p('Each interval resamples prompt/seed units. Frames and all within-video model operations stay inside that unit.',small)
    image('comparison-with-replacement.png',220)
    p('All five arms use the same 100 prompts/seeds. Aesthetics and imaging quality use all 45 decoded frames per video. ViCLIP text-video consistency uses eight segment-midpoint frames.',small)
    table([['Arm','Mean generation+decode seconds','Median seconds']]+[[NAMES[a],f"{data['timing'][a]['mean']:.2f}",f"{data['timing'][a]['median']:.2f}"] for a in ARMS],[160,190,153],8)
    story.append(PageBreak());p('4. Interpretation',head)
    for paragraph in findings(data):p(paragraph)
    p('Paired evidence',sub)
    for dimension in ('aesthetic_quality','imaging_quality','overall_consistency'):
        item=data['dimensions'][dimension]
        p(LABELS[dimension]+'. 75% minus 50% '+interval(item['paired']['tri75_minus_tri50'])+'.',small)
    examples=[p for p in data['config']['prompts'] if p['audit']]
    for start in range(0,len(examples),2):
        story.append(PageBreak());p(f'5. Prespecified examples ({start+1}-{start+2} of 8)',head)
        p('Decoded frame index 22 of 45. Each comparison shares the prompt and seed. The eight example prompts were fixed before inspecting final metric scores.',small)
        for prompt in examples[start:start+2]:
            p(prompt['id']+' / '+prompt['category']+' / seed '+str(prompt['seed']),sub);p(prompt['text'],small)
            image(prompt['id']+'-comparison.png',200)
            values=data['dimensions']['imaging_quality']['per_prompt'][prompt['id']]
            p('MUSIQ / 100, averaged over all 45 frames. '+', '.join(NAMES[a]+' '+number(values[a]) for a in ARMS)+'.',small)
    story.append(PageBreak());p('6. Exact evaluator populations',head)
    for dimension in ORDER:
        if dimension in data['dimensions']:p(LABELS[dimension],sub);p(SCOPES[dimension],small)
    story.append(PageBreak());p('7. Adaptation, checks and reproducibility',head)
    for paragraph in scope_paragraphs():p(paragraph,small)
    p('Verification',sub)
    p('All 300 video hashes and 45-frame/480x480/16-fps metadata pass. All 200 pruned trajectories share one generation-source/calibration hash and each covers 1,200 layer/step/branch dispatches. The native full-retention v000 final latent exactly matches the original saved baseline.',small)
    p('The selected-key test compares the factored scorer with the pinned official implementation over all four query blocks, 12 heads and both budgets. It verifies every live query key is retained and each gathered K/V coordinate is exact. Scores use all 64 pairs and the per-pair norm correction.',small)
    p('Reproducibility files',sub)
    p('config.json contains every prompt, seed and policy setting. calibration/manifest.json states the calibration population. summary.json and per-prompt-results.csv contain all scores, paired intervals and video hashes. selection-verification.json records scorer/gather tests. The HTML gallery exposes all 300 videos.',small)
    p('Official source',sub)
    p('TriAttention commit a4bc3c8f709db60f016ef42c3feb290fd0c00c1b. https://github.com/WeianMao/triattention',small)
    p('VBench commit '+data['config']['vbench_commit']+'. https://github.com/Vchitect/VBench',small)
    path.parent.mkdir(parents=True,exist_ok=True)
    def footer(canvas,doc):
        canvas.setFont('Helvetica',8);canvas.setFillColor(colors.HexColor('#657783'));canvas.drawString(46,24,'Wan query-block K/V pruning | 9 November 2026');canvas.drawRightString(549,24,str(doc.page))
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=46,rightMargin=46,topMargin=37,bottomMargin=39,title='TriAttention-style pruning in full-video Wan').build(story,onFirstPage=footer,onLaterPages=footer)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--pdf',required=True);args=parser.parse_args()
    root=Path(args.root);data=json.loads((root/'summary.json').read_text());figures(root,data);report_markdown(root,data);report_html(root,data);report_pdf(root,data,Path(args.pdf));print('Report generated',args.pdf,flush=True)


if __name__=='__main__':main()
