'use strict';

// ── state ──────────────────────────────────────────────────────────────────
let currentMethod  = 'phase2';
let uploadedB64    = null;   // original image base64
let isColor        = false;
let detections     = [];     // [{id,label,score,bbox}]
const instPrompts  = {};     // {id: promptString}

const SWATCH_COLORS = [
  { name: 'red',    hex: '#e74c3c' },
  { name: 'yellow', hex: '#f1c40f' },
  { name: 'green',  hex: '#27ae60' },
  { name: 'blue',   hex: '#3498db' },
  { name: 'orange', hex: '#e67e22' },
  { name: 'purple', hex: '#9b59b6' },
  { name: 'brown',  hex: '#8B4513' },
  { name: 'gray',   hex: '#95a5a6' },
];

const BBOX_COLORS = [
  '#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4',
  '#FFEAA7', '#DDA0DD', '#98D8C8', '#F7DC6F',
];

const METHOD_HINTS = {
  phase1: 'CNN global colorization (Zhang et al. 2016). No instance control. ~5s on CPU.',
  phase2: 'FiLM instance-aware colorization (default). Click an instance card to view its pre-fusion coloring. ~15s on CPU.',
  phase3: 'CLIP text-guided colorization. Assign a colour prompt to each instance. ~15s on CPU.',
};

// ── DOM refs ───────────────────────────────────────────────────────────────
const dropZone      = document.getElementById('drop-zone');
const fileInput     = document.getElementById('file-input');
const uploadPreview = document.getElementById('upload-preview');
const methodBtns    = document.querySelectorAll('.method-btn');
const methodHint    = document.getElementById('method-hint');
const instSection   = document.getElementById('instance-section');
const instList      = document.getElementById('inst-list');
const runBtn        = document.getElementById('run-btn');
const spinner       = document.getElementById('spinner');
const statusMsg     = document.getElementById('status-msg');
const resultSection = document.getElementById('result-section');
const resultGrid    = document.getElementById('result-grid');
const downloadBtn   = document.getElementById('download-btn');
const modalOverlay  = document.getElementById('modal-overlay');
const modalClose    = document.getElementById('modal-close');
const modalTitle    = document.getElementById('modal-title');
const modalGray     = document.getElementById('modal-gray');
const modalColored  = document.getElementById('modal-colored');

// ── upload ─────────────────────────────────────────────────────────────────
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', e => { e.preventDefault(); dropZone.classList.add('over'); });
dropZone.addEventListener('dragleave',  () => dropZone.classList.remove('over'));
dropZone.addEventListener('drop', e => {
  e.preventDefault();
  dropZone.classList.remove('over');
  handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', () => handleFile(fileInput.files[0]));

function handleFile(file) {
  if (!file) return;
  if (file.size > 4 * 1024 * 1024) {
    alert('Image too large (max 4 MB)');
    return;
  }
  const reader = new FileReader();
  reader.onload = e => {
    uploadedB64 = e.target.result.split(',')[1];   // strip data:... prefix
    uploadPreview.src = e.target.result;
    uploadPreview.style.display = 'block';
    detectInstances();
  };
  reader.readAsDataURL(file);
}

// ── detect ─────────────────────────────────────────────────────────────────
async function detectInstances() {
  setStatus('Detecting instances…');
  resultSection.style.display = 'none';
  instSection.style.display   = 'none';
  detections = [];

  try {
    const res  = await fetch('/api/detect', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({image_b64: uploadedB64}),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    isColor    = data.is_color;
    detections = data.instances;
    uploadPreview.src = 'data:image/png;base64,' + data.preview_b64;
    setStatus(
      `Detected ${detections.length} instance(s). ` +
      (isColor ? 'Colour image detected.' : 'Grayscale image detected.')
    );
    renderInstances();
  } catch (err) {
    setStatus('Detection failed: ' + err.message);
  }
}

// ── instance list ──────────────────────────────────────────────────────────
function renderInstances() {
  instList.innerHTML = '';
  if (detections.length === 0) {
    instSection.style.display = 'none';
    return;
  }
  if (currentMethod === 'phase1') {
    instSection.style.display = 'none';
    return;
  }
  instSection.style.display = 'block';

  detections.forEach(d => {
    const card = document.createElement('div');
    card.className = 'inst-card' + (currentMethod === 'phase2' ? ' p2-card' : '');
    const bcolor = BBOX_COLORS[d.id % BBOX_COLORS.length];

    let inner = `
      <div class="inst-header">
        <span class="inst-badge" style="background:${bcolor}"></span>
        inst:${d.id} &nbsp;<strong>${d.label}</strong>
        <span class="score-chip">${d.score.toFixed(2)}</span>
      </div>
    `;

    if (currentMethod === 'phase3') {
      const swatches = SWATCH_COLORS.map(sc =>
        `<div class="swatch" title="${sc.name}"
              style="background:${sc.hex}"
              data-color="${sc.name}"
              data-id="${d.id}"></div>`
      ).join('');
      const defaultVal = instPrompts[d.id] || `a ${d.label}`;
      inner += `
        <div class="color-swatches">${swatches}</div>
        <div class="prompt-row">
          <input class="prompt-input" type="text"
                 data-id="${d.id}"
                 placeholder="a red ${d.label}"
                 value="${defaultVal}">
          <button class="reset-prompt-btn" data-id="${d.id}" data-label="${d.label}" title="Reset to default">Reset</button>
        </div>
      `;
    } else if (currentMethod === 'phase2') {
      inner += `<div class="inst-view-hint">Click to view instance coloring →</div>`;
    }

    card.innerHTML = inner;
    instList.appendChild(card);
  });

  // Phase 3 swatch + input events
  if (currentMethod === 'phase3') {
    instList.querySelectorAll('.swatch').forEach(sw => {
      sw.addEventListener('click', () => {
        const id    = sw.dataset.id;
        const color = sw.dataset.color;
        const label = detections.find(d => String(d.id) === id)?.label || 'object';
        // deselect siblings
        sw.closest('.inst-card').querySelectorAll('.swatch').forEach(s => s.classList.remove('selected'));
        sw.classList.add('selected');
        const inp = instList.querySelector(`.prompt-input[data-id="${id}"]`);
        if (inp) inp.value = `a ${color} ${label}`;
        instPrompts[id] = `a ${color} ${label}`;
      });
    });
    instList.querySelectorAll('.prompt-input').forEach(inp => {
      inp.addEventListener('input', () => { instPrompts[inp.dataset.id] = inp.value; });
    });
    instList.querySelectorAll('.reset-prompt-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const id    = btn.dataset.id;
        const label = btn.dataset.label;
        const inp   = instList.querySelector(`.prompt-input[data-id="${id}"]`);
        // clear selected swatch
        btn.closest('.inst-card').querySelectorAll('.swatch').forEach(s => s.classList.remove('selected'));
        const defaultPrompt = `a ${label}`;
        if (inp) inp.value = defaultPrompt;
        instPrompts[id] = defaultPrompt;
      });
    });
  }

  // Phase 2 click to open modal (filled later by colorize response)
  if (currentMethod === 'phase2') {
    instList.querySelectorAll('.p2-card').forEach(card => {
      card.addEventListener('click', () => {
        const header = card.querySelector('.inst-header');
        const id = parseInt(header.querySelector('.inst-badge').nextSibling.textContent.trim().split(' ')[0].replace('inst:', ''));
        openModalForInst(id);
      });
    });
  }
}

// ── method switch ──────────────────────────────────────────────────────────
methodBtns.forEach(btn => {
  btn.addEventListener('click', () => {
    methodBtns.forEach(b => b.classList.remove('active'));
    btn.classList.add('active');
    currentMethod = btn.dataset.method;
    methodHint.textContent = METHOD_HINTS[currentMethod];
    renderInstances();
  });
});

// ── run colorize ───────────────────────────────────────────────────────────
runBtn.addEventListener('click', async () => {
  if (!uploadedB64) { alert('Please upload an image first.'); return; }

  // collect prompts for phase3
  const prompts = {};
  if (currentMethod === 'phase3') {
    instList.querySelectorAll('.prompt-input').forEach(inp => {
      if (inp.value.trim()) prompts[inp.dataset.id] = inp.value.trim();
    });
  }

  runBtn.disabled = true;
  spinner.style.display = 'inline-block';
  const hints = { phase1: '~5s', phase2: '~15s', phase3: '~15s' };
  setStatus(`Colorizing with ${currentMethod.toUpperCase()} (expected ${hints[currentMethod]})…`);
  resultSection.style.display = 'none';

  try {
    const res  = await fetch('/api/colorize', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        image_b64:  uploadedB64,
        method:     currentMethod,
        prompts:    prompts,
        detections: detections,
      }),
    });
    const data = await res.json();
    if (data.error) throw new Error(data.error);

    renderResults(data);
    // store instance crops for modal
    if (data.instance_crops) {
      window._instanceCrops = {};
      data.instance_crops.forEach(ic => { window._instanceCrops[ic.id] = ic; });
    }
    setStatus('Done!');
  } catch (err) {
    setStatus('Colorization failed: ' + err.message);
  } finally {
    runBtn.disabled = false;
    spinner.style.display = 'none';
  }
});

// ── render results ─────────────────────────────────────────────────────────
function renderResults(data) {
  resultGrid.innerHTML = '';

  const addItem = (label, b64) => {
    const item = document.createElement('div');
    item.className = 'result-item';
    item.innerHTML = `<p>${label}</p><img src="data:image/png;base64,${b64}">`;
    resultGrid.appendChild(item);
  };

  if (isColor) {
    addItem('Original', uploadedB64);
    if (data.gray_b64) addItem('Grayscale', data.gray_b64);
  } else {
    addItem('Grayscale input', uploadedB64);
  }
  addItem('Colorized result', data.colorized_b64);

  downloadBtn.onclick = () => {
    const a = document.createElement('a');
    a.href = 'data:image/png;base64,' + data.colorized_b64;
    a.download = 'colorized.png';
    a.click();
  };

  resultSection.style.display = 'block';
}

// ── instance modal (Phase 2) ───────────────────────────────────────────────
function openModalForInst(id) {
  const crops = window._instanceCrops || {};
  const ic = crops[id];
  const d  = detections.find(x => x.id === id);
  if (!ic) {
    alert('Run colorization first to see instance details.');
    return;
  }
  modalTitle.textContent = `inst:${id} — ${d ? d.label : 'unknown'} (FiLM instance branch, pre-fusion)`;
  modalGray.src    = 'data:image/png;base64,' + ic.gray_b64;
  modalColored.src = 'data:image/png;base64,' + ic.colored_b64;
  modalOverlay.classList.add('open');
}

modalClose.addEventListener('click',  () => modalOverlay.classList.remove('open'));
modalOverlay.addEventListener('click', e => {
  if (e.target === modalOverlay) modalOverlay.classList.remove('open');
});

// ── helpers ────────────────────────────────────────────────────────────────
function setStatus(msg) { statusMsg.textContent = msg; }
