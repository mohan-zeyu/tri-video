/* Saved 2D-pair MRL values, with explicit frequency grouping and token scope. */
const MrlHeatmaps = (() => {
    const element = id => document.getElementById(id);
    const cache = new Map();
    const geometry = {width: 360, height: 554, left: 48, top: 18, cellWidth: 24, cellHeight: 16};
    let metadataPromise;
    let metadata;
    let current;
    let revision = 0;
    let selectCell;
    let canvases = [];

    async function readJson(path) {
        const response = await fetch(path);
        if (!response.ok) throw new Error(`HTTP ${response.status} for ${path}`);
        return response.json();
    }

    function cellValue(side, pairs, layer, head) {
        const stage = element('mrl-stage').value;
        const row = current.record.mrl[`${side}_pair_mrl_${stage}`][layer][head];
        return pairs.reduce((sum, pair) => sum + row[pair], 0) / pairs.length;
    }

    function point(event, canvas) {
        const bounds = canvas.getBoundingClientRect();
        const x = (event.clientX - bounds.left) * geometry.width / bounds.width;
        const y = (event.clientY - bounds.top) * geometry.height / bounds.height;
        const head = Math.floor((x - geometry.left) / geometry.cellWidth);
        const layer = Math.floor((y - geometry.top) / geometry.cellHeight);
        return head >= 0 && head < 12 && layer >= 0 && layer < 30 ? {layer, head} : null;
    }

    function draw(side, title, pairs) {
        const card = document.createElement('div');
        const heading = document.createElement('h3');
        heading.textContent = `${side.toUpperCase()} · ${title}`;
        const canvas = document.createElement('canvas');
        canvas.width = geometry.width * 2;
        canvas.height = geometry.height * 2;
        canvas.setAttribute('role', 'img');
        canvas.setAttribute('aria-label', `${heading.textContent}, MRL for 30 layers and 12 heads. Pairs ${pairs.join(', ')}.`);
        const context = canvas.getContext('2d');
        context.scale(2, 2);
        context.fillStyle = '#fff';
        context.fillRect(0, 0, geometry.width, geometry.height);
        for (let layer = 0; layer < 30; layer++) {
            for (let head = 0; head < 12; head++) {
                const value = cellValue(side, pairs, layer, head);
                context.fillStyle = metadata.palette[Math.min(255, Math.floor(value * 256))];
                context.fillRect(geometry.left + head * geometry.cellWidth,
                    geometry.top + layer * geometry.cellHeight, geometry.cellWidth, geometry.cellHeight);
            }
        }
        context.font = '11px system-ui';
        context.fillStyle = '#202a36';
        context.textAlign = 'right';
        for (let layer = 0; layer < 30; layer++) {
            context.fillText(String(layer), geometry.left - 7, geometry.top + (layer + .5) * geometry.cellHeight + 4);
        }
        context.textAlign = 'center';
        for (let head = 0; head < 12; head++) {
            context.fillText(String(head), geometry.left + (head + .5) * geometry.cellWidth, 518);
        }
        context.font = '13px system-ui';
        context.fillText('Head (zero-based)', 192, 544);
        context.save();
        context.translate(13, 258);
        context.rotate(-Math.PI / 2);
        context.fillText('Layer (zero-based)', 0, 0);
        context.restore();
        if (current.head !== 'all') {
            context.strokeStyle = '#fff';
            context.lineWidth = 2;
            context.strokeRect(geometry.left + Number(current.head) * geometry.cellWidth + 1,
                geometry.top + current.layer * geometry.cellHeight + 1, geometry.cellWidth - 2, geometry.cellHeight - 2);
        }
        canvas.addEventListener('mousemove', event => {
            const selected = point(event, canvas);
            if (!selected) return;
            const value = cellValue(side, pairs, selected.layer, selected.head);
            element('mrl-readout').textContent = `${current.prompt} · step ${current.step} · ${current.axis} axis · ` +
                `layer ${selected.layer} · head ${selected.head} · ${side.toUpperCase()} ${title} · MRL ${value.toFixed(7)}. ` +
                `${pairs.length} pair${pairs.length === 1 ? '' : 's'}, each computed over all 10,800 tokens.`;
        });
        canvas.addEventListener('click', event => {
            const selected = point(event, canvas);
            if (selected) selectCell(selected.layer, selected.head);
        });
        const label = document.createElement('p');
        label.className = 'note small';
        label.textContent = `Pair IDs ${pairs.join(', ')}`;
        card.append(heading, canvas, label);
        element('mrl-panels').append(card);
        canvases.push({canvas, title: heading.textContent});
    }

    function render() {
        if (!current || !metadata) return;
        const axis = metadata.groups[current.axis];
        const choice = element('mrl-band').value;
        const groups = choice === 'bands' ? Object.entries(axis.bands).map(([name, definition]) => ({
            title: `${name} frequency mean`, pairs: definition.pairs
        })) : [{title: `pair ${choice}`, pairs: [Number(choice)]}];
        element('mrl-panels').replaceChildren();
        element('mrl-panels').classList.toggle('single-pair', choice !== 'bands');
        canvases = [];
        for (const side of ['q', 'k']) {
            for (const group of groups) draw(side, group.title, group.pairs);
        }
        const frequency = value => value.toPrecision(4);
        element('mrl-definition').textContent = 'Bands are nearly equal-count thirds of actual angular frequency, ' +
            'ordered from fast to slow rotation independently within this axis. ' +
            Object.entries(axis.bands).map(([name, band]) => `${name}: ${band.pairs.length} pairs, ` +
                `${frequency(band.frequency_min)}–${frequency(band.frequency_max)} radians per token`).join('. ') + '.';
        element('mrl-status').textContent = `${current.prompt} · denoising step ${current.step} · ${current.axis} axis · ` +
            `${element('mrl-stage').value}-RoPE · all 30 layers × 12 heads. Each pair uses all 10,800 tokens at one prompt/step/layer/head.`;
    }

    function download() {
        if (!current || !canvases.length) return;
        const columns = canvases.length === 6 ? 3 : 2;
        const rows = canvases.length / columns;
        const output = document.createElement('canvas');
        output.width = columns * 720;
        output.height = 150 + rows * 1170 + 140;
        const context = output.getContext('2d');
        context.fillStyle = '#fff';
        context.fillRect(0, 0, output.width, output.height);
        context.fillStyle = '#202a36';
        context.font = '30px Georgia';
        context.fillText(`${current.prompt} | step ${current.step} | ${current.axis} axis | ${element('mrl-stage').value}-RoPE`, 35, 50);
        context.font = '21px Georgia';
        context.fillText('Layer × head MRL. Each pair uses all 10,800 tokens at one prompt/step/layer/head.', 35, 93);
        canvases.forEach((item, index) => {
            const x = (index % columns) * 720;
            const y = 150 + Math.floor(index / columns) * 1170;
            context.fillStyle = '#202a36';
            context.font = '24px Georgia';
            context.fillText(item.title, x + 55, y - 10);
            context.drawImage(item.canvas, x, y);
        });
        const barY = output.height - 100;
        metadata.palette.forEach((color, index) => {
            context.fillStyle = color;
            context.fillRect(80 + index * 2, barY, 3, 20);
        });
        context.fillStyle = '#202a36';
        context.font = '19px Georgia';
        context.fillText('0', 55, barY + 18);
        context.fillText('1   MRL', 610, barY + 18);
        context.fillText('Band panels average the explicitly listed pair MRL values. Individual-pair panels use one saved pair.', 35, output.height - 35);
        output.toBlob(blob => {
            const link = document.createElement('a');
            const url = URL.createObjectURL(blob);
            link.href = url;
            link.download = `${current.prompt}-step-${current.step}-${current.axis}-${element('mrl-stage').value}-mrl.png`;
            link.click();
            setTimeout(() => URL.revokeObjectURL(url), 1000);
        });
    }

    function init(onSelected) {
        selectCell = onSelected;
        metadataPromise = readJson('mrl-heatmaps/manifest.json');
        element('mrl-stage').addEventListener('change', render);
        element('mrl-band').addEventListener('change', render);
        element('mrl-export').addEventListener('click', download);
    }

    async function load(prompt, step, axis, layer, head) {
        const requestRevision = ++revision;
        const previousAxis = current?.axis;
        if (current && (current.prompt !== prompt || current.step !== step || current.axis !== axis)) {
            element('mrl-panels').replaceChildren();
            current = null;
            canvases = [];
        }
        element('mrl-status').textContent = 'Loading all-layer pair measurements…';
        const path = `mrl-heatmaps/data/${prompt}-step-${String(step).padStart(3, '0')}.json`;
        element('mrl-data').href = path;
        try {
            if (!cache.has(path)) cache.set(path, readJson(path));
            const [definitions, record] = await Promise.all([metadataPromise, cache.get(path)]);
            if (requestRevision !== revision) return;
            metadata = definitions;
            current = {prompt, step, axis, layer, head, record};
            if (previousAxis !== axis) {
                const options = [new Option('High / middle / low frequency means', 'bands')];
                for (const pair of metadata.groups[axis].pairs) {
                    const omega = metadata.groups[axis].radians_per_token[String(pair)].toPrecision(4);
                    options.push(new Option(`Pair ${pair} · ${omega} rad/token`, String(pair)));
                }
                element('mrl-band').replaceChildren(...options);
            }
            if (cache.size > 3) cache.delete(cache.keys().next().value);
            element('mrl-gradient').style.background = `linear-gradient(to right, ${metadata.palette.join(',')})`;
            render();
        } catch (error) {
            cache.delete(path);
            if (requestRevision === revision) element('mrl-status').textContent = `Heatmap loading failed. ${error.message}`;
        }
    }

    return {init, load};
})();
