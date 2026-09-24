/* CLM playground.
 *
 * Talks to the same clm-serve that served this page:
 *   GET  /health        liveness + whether the encoder is reachable
 *   GET  /v1/models     model picker
 *   POST /v1/systemone  the one call this page is about
 *
 * Plain ES2020, no build step and no network imports, so it also works on a
 * cluster node behind `ssh -L 8700:localhost:8700`.
 */
'use strict';

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));
const pct = (p) => (p * 100).toFixed(p >= 0.9995 ? 1 : p >= 0.1 ? 1 : 2) + '%';

const LS = {
  req: 'clm.playground.request',
  base: 'clm.playground.baseUrl',
  key: 'clm.playground.apiKey',
  theme: 'clm.playground.theme',
};

/* ───────────────────────────── presets ───────────────────────────── */

const PRESETS = {
  support: {
    label: 'Support ticket triage',
    mode: 'ask',
    state: "Customer: my invoice was charged twice and nobody answers the phone!",
    questions: [
      { id: 'urgency', type: 'noul', instructions: 'Is this urgent?' },
      { id: 'department', type: 'choice', instructions: 'Which team should handle this?',
        choiceCrit: [
          { key: 'billing', desc: 'Charges, invoices, refunds' },
          { key: 'technical', desc: 'Bugs and outages' },
        ] },
      { id: 'frustration', type: 'score', instructions: 'How frustrated is the customer?',
        scoreCrit: ['Calm', 'Frustrated', 'Very angry'] },
    ],
  },

  trajectory: {
    label: 'Agent trajectory verifier',
    mode: 'ask',
    state:
`Task: the test suite fails with "ModuleNotFoundError: No module named 'yaml'".

Agent trajectory:
  $ pytest -q
  ModuleNotFoundError: No module named 'yaml'
  $ pip install pyyaml
  Successfully installed pyyaml-6.0.2
  $ pytest -q
  34 passed in 4.21s`,
    questions: [
      { id: 'solved', type: 'noul', instructions: 'Did the agent actually solve the task?' },
      { id: 'failure_mode', type: 'choice', instructions: 'If it went wrong, how?',
        choiceCrit: [
          { key: 'none', desc: 'Nothing went wrong; the task is complete' },
          { key: 'gave_up', desc: 'The agent stopped before finishing' },
          { key: 'faked_it', desc: 'The agent edited or skipped the tests instead of fixing the code' },
          { key: 'wrong_fix', desc: 'The agent changed something unrelated to the failure' },
        ] },
      { id: 'quality', type: 'score', instructions: 'How good is this trajectory?',
        scoreCrit: ['Broken — the task is not done', 'Works but is sloppy', 'Clean, minimal and correct'] },
    ],
  },

  review: {
    label: 'Pull-request risk review',
    mode: 'ask',
    state:
`diff --git a/auth/session.py b/auth/session.py
@@
-    if token and verify(token):
-        return load_user(token)
-    return None
+    if token:
+        return load_user(token)
+    return None`,
    questions: [
      { id: 'safe_to_merge', type: 'noul', instructions: 'Is this change safe to merge as-is?' },
      { id: 'area', type: 'choice', instructions: 'Which area does this touch?',
        choiceCrit: [
          { key: 'security', desc: 'Authentication, authorization, secrets, crypto' },
          { key: 'performance', desc: 'Latency, memory, throughput' },
          { key: 'style', desc: 'Formatting, naming, comments' },
          { key: 'feature', desc: 'New user-visible behaviour' },
        ] },
      { id: 'risk', type: 'score', instructions: 'How risky is this change?',
        scoreCrit: ['Trivial', 'Worth a second pair of eyes', 'Do not merge without a test'] },
    ],
  },

  bestofn: {
    label: 'Best-of-N answers',
    mode: 'rank',
    state: 'What causes tides on Earth?',
    rankInstructions: '',
    rankCandidates:
`The Moon's gravitational pull.
Photosynthesis in plants.
Because the Earth is round.
The gravitational pull of the Moon and, to a lesser extent, the Sun.`,
  },

  tools: {
    label: 'Tool routing',
    mode: 'rank',
    state: "User: I'm in Berlin until Friday — will I need an umbrella?",
    rankInstructions: 'Which tool should the agent call next?',
    rankCandidates:
`get_weather_forecast(city, days) — multi-day forecast for a city
search_web(query) — general web search
send_email(to, subject, body) — send an email on the user's behalf
book_flight(origin, destination, date) — search and book flights
get_calendar_events(start, end) — the user's calendar`,
  },
};

/* ───────────────────────────── app state ───────────────────────────── */

let S = blankState();
let lastResult = null;   // {body, json, latency, raw?}
let qSeq = 0;

function blankState() {
  return {
    mode: 'ask',
    state: '',
    stateJson: false,
    questions: [],
    rankInstructions: '',
    rankCandidates: '',
    model: 'clm-latest',
  };
}

function newQuestion(type) {
  qSeq += 1;
  return {
    id: type + (qSeq > 1 ? String(qSeq) : ''),
    type,
    instructions: '',
    noulCrit: { true: '', false: '' },
    showNoulCrit: false,
    choiceCrit: [{ key: '', desc: '' }, { key: '', desc: '' }],
    scoreCrit: ['', '', ''],
  };
}

function hydrate(o) {
  const s = blankState();
  if (!o || typeof o !== 'object') return s;
  s.mode = o.mode === 'rank' ? 'rank' : 'ask';
  s.state = typeof o.state === 'string' ? o.state : '';
  s.stateJson = !!o.stateJson;
  s.rankInstructions = o.rankInstructions || '';
  s.rankCandidates = o.rankCandidates || '';
  s.model = o.model || 'clm-latest';
  s.questions = (Array.isArray(o.questions) ? o.questions : []).map((q, i) => {
    const d = newQuestion(q.type === 'choice' || q.type === 'score' ? q.type : 'noul');
    d.id = q.id || d.id;
    d.instructions = q.instructions || '';
    if (q.noulCrit) d.noulCrit = { true: q.noulCrit.true || '', false: q.noulCrit.false || '' };
    d.showNoulCrit = !!(d.noulCrit.true || d.noulCrit.false);
    if (Array.isArray(q.choiceCrit) && q.choiceCrit.length) {
      d.choiceCrit = q.choiceCrit.map((r) => ({ key: r.key || '', desc: r.desc || '' }));
    }
    if (Array.isArray(q.scoreCrit) && q.scoreCrit.length) d.scoreCrit = q.scoreCrit.slice();
    return d;
  });
  return s;
}

/* ───────────────────────────── connection ───────────────────────────── */

const baseUrl = () => ($('#base-url').value || '').trim().replace(/\/+$/, '');
const apiKey = () => ($('#api-key').value || '').trim();
const url = (path) => baseUrl() + path;

function headers(json) {
  const h = json ? { 'Content-Type': 'application/json' } : {};
  const k = apiKey();
  if (k) h['Authorization'] = 'Bearer ' + k;
  return h;
}

function setStatus(cls, text, title) {
  const el = $('#status');
  el.className = 'status ' + cls;
  $('#status-text').textContent = text;
  if (title) el.title = title;
}

// The playground assumes the GPU encoder is up; when it is not, say so up front
// and keep re-checking until it is.
let healthTimer = null;
function encoderDown(text) {
  $('#encoder-banner-text').textContent = text;
  $('#encoder-banner').hidden = false;
  clearTimeout(healthTimer);
  healthTimer = setTimeout(refreshStatus, 10000);
}

async function refreshStatus() {
  setStatus('busy', 'connecting…');
  clearTimeout(healthTimer);
  let health;
  try {
    const r = await fetch(url('/health'), { headers: headers(false) });
    health = await r.json();
  } catch (e) {
    setStatus('bad', 'no server', `Could not reach ${baseUrl() || 'this origin'}/health — is clm-serve running?`);
    $('#mock-banner').hidden = true;
    encoderDown(`Could not reach clm-serve at ${baseUrl() || location.origin}; start clm-serve and the encoder.`);
    return;
  }
  const models = health.models || [];
  $('#mock-banner').hidden = !health.mock;
  $('#encoder-banner').hidden = health.embedder !== false;
  if (health.embedder === false) {
    setStatus('warn', 'encoder down',
      'clm-serve is up but its /v1/embeddings encoder is not reachable — requests will fail with 502.');
    encoderDown('clm-serve is up, but its Qwen3-8B /v1/embeddings server is not reachable, so every request will fail.');
  } else {
    setStatus('ok', models.length ? `ready · ${models.length} model${models.length > 1 ? 's' : ''}` : 'ready',
      'clm-serve and the encoder are both up.');
  }
  try {
    const r = await fetch(url('/v1/models'), { headers: headers(false) });
    if (r.ok) fillModels((await r.json()).models || []);
  } catch (e) { /* the health list is enough */ }
}

function fillModels(models) {
  const sel = $('#model');
  const want = S.model;
  sel.innerHTML = '';
  for (const m of models) {
    const o = document.createElement('option');
    o.value = m.name;
    o.textContent = m.name;
    o.title = m.description || '';
    sel.appendChild(o);
  }
  if (!models.length) sel.innerHTML = '<option value="clm-latest">clm-latest</option>';
  sel.value = models.some((m) => m.name === want) ? want : sel.options[0].value;
  S.model = sel.value;
}

/* ───────────────────────────── request building ───────────────────────────── */

function parsedState() {
  if (!S.stateJson) return S.state;
  return JSON.parse(S.state);          // caller guards
}

function questionToWire(q) {
  const ins = q.instructions;
  if (q.type === 'noul') {
    const d = { type: 'noul', instructions: ins };
    const c = {};
    if (q.noulCrit.true.trim()) c['true'] = q.noulCrit.true;
    if (q.noulCrit.false.trim()) c['false'] = q.noulCrit.false;
    if (Object.keys(c).length) d.criteria = c;
    return d;
  }
  if (q.type === 'choice') {
    const criteria = {};
    for (const r of q.choiceCrit) {
      const k = r.key.trim();
      if (!k) continue;
      // criteria is a JSON object: a repeated key would silently overwrite the earlier one
      if (k in criteria) throw new Error(`Duplicate option key "${k}" — each option needs its own key.`);
      criteria[k] = r.desc;
    }
    return { type: 'choice', instructions: ins, criteria };
  }
  return { type: 'score', instructions: ins, criteria: q.scoreCrit.filter((s) => s.trim()) };
}

function rankCandidateList() {
  return S.rankCandidates.split('\n').map((s) => s.trim()).filter(Boolean);
}

/** -> {endpoint, body} or throws Error with a human message. */
function buildRequest(model) {
  if (S.stateJson) {
    try { JSON.parse(S.state); } catch (e) { throw new Error('State is not valid JSON: ' + e.message); }
  }
  if (!S.state.trim()) throw new Error('Add a state — the context the questions are asked about.');

  if (S.mode === 'rank') {
    const answers = rankCandidateList();
    if (answers.length < 2) throw new Error('Add at least two candidates, one per line.');
    // POST /v1/rank takes the answers verbatim: no option keys, nothing prefixed
    return {
      endpoint: '/v1/rank',
      body: { context: parsedState(), question: S.rankInstructions || null,
              answers, model: model || S.model },
    };
  }

  const body = { state: parsedState(), model: model || S.model };

  if (!S.questions.length) throw new Error('Add at least one question.');
  const questions = {};
  for (const q of S.questions) {
    const id = q.id.trim();
    if (!id) throw new Error('Every question needs an id.');
    if (questions[id]) throw new Error(`Duplicate question id "${id}".`);
    if (!q.instructions.trim()) throw new Error(`Question "${id}" needs instructions.`);
    const w = questionToWire(q);
    if (q.type === 'choice' && Object.keys(w.criteria).length < 2) {
      throw new Error(`Choice question "${id}" needs at least two options with a key.`);
    }
    if (q.type === 'score' && w.criteria.length < 2) {
      throw new Error(`Score question "${id}" needs at least two rubric levels.`);
    }
    questions[id] = w;
  }
  body.questions = questions;
  return { endpoint: '/v1/systemone', body };
}

/* ───────────────────────────── run ───────────────────────────── */

async function post(path, body) {
  const t0 = performance.now();
  const r = await fetch(url(path), {
    method: 'POST', headers: headers(true), body: JSON.stringify(body),
  });
  const roundTrip = performance.now() - t0;
  const text = await r.text();
  let json = null;
  try { json = JSON.parse(text); } catch (e) { /* non-JSON error body */ }
  if (!r.ok) {
    const detail = (json && (json.detail || json.error)) || text || r.statusText;
    const err = new Error(typeof detail === 'string' ? detail : JSON.stringify(detail));
    err.status = r.status;
    throw err;
  }
  const served = r.headers.get('X-CLM-Latency-Ms');
  return { json, latency: served !== null ? Number(served) : null, roundTrip };
}

async function run() {
  const btn = $('#run');
  let req;
  try {
    req = buildRequest();
  } catch (e) {
    e.validation = true;          // nothing was sent: do not blame the network
    showError(e, null);
    return;
  }

  btn.disabled = true;
  btn.textContent = 'Running…';
  setStatus('busy', 'running…');

  try {
    const main = await post(req.endpoint, req.body);
    lastResult = { req, main };
    renderAnswers(req, main);
    renderJson(main.json);
    setStatus('ok', `${(main.latency ?? main.roundTrip).toFixed(0)} ms`,
      'Server-side time from the X-CLM-Latency-Ms header.');
  } catch (e) {
    showError(e, req.body);
    setStatus(e.status === 502 ? 'warn' : 'bad', e.status ? 'HTTP ' + e.status : 'failed');
  } finally {
    btn.disabled = false;
    btn.innerHTML = 'Run <kbd>\u2318</kbd><kbd>\u21b5</kbd>';
    renderCode();
  }
}

/* ───────────────────────────── rendering answers ───────────────────────────── */

function renderMeta(main) {
  const u = main.json.usage || {};
  const bits = [
    `<span>model <b>${esc(main.json.model)}</b></span>`,
    `<span>latency <b>${(main.latency ?? main.roundTrip).toFixed(1)} ms</b></span>`,
  ];
  if (main.latency !== null) bits.push(`<span>round trip <b>${main.roundTrip.toFixed(0)} ms</b></span>`);
  if (u.input_tokens !== undefined) bits.push(`<span>encoder tokens <b>${u.input_tokens}</b></span>`);
  if (u.billing_units !== undefined) bits.push(`<span>billing units <b>${u.billing_units}</b></span>`);
  const meta = $('#meta');
  meta.innerHTML = bits.join('');
  meta.hidden = false;
}

function barsHtml(entries, labelOf) {
  const max = Math.max(...entries.map((e) => e[1]), 1e-9);
  const top = entries.reduce((a, b) => (b[1] > a[1] ? b : a), entries[0]);
  return `<div class="bars">${entries.map(([k, p]) => {
    const desc = labelOf ? labelOf(k) : '';
    return `<div class="bar ${k === top[0] ? 'is-top' : 'is-dim'}">
      <div class="bar-fill" style="width:${(p / max * 100).toFixed(2)}%"></div>
      <span class="bar-key">${esc(k)}</span>
      ${desc ? `<span class="bar-desc">${esc(desc)}</span>` : ''}
      <span class="bar-pct">${pct(p)}</span>
    </div>`;
  }).join('')}</div>`;
}

function answerHtml(id, q, a) {
  const type = a.type;
  let head = '', body = '';

  if (type === 'noul') {
    const p = a.noul;
    head = `<div class="headline">
      <span class="headline-val">${pct(p)}</span>
      <span class="headline-lab">p(true)</span>
      <span class="headline-sub">leans ${p >= 0.5 ? 'true' : 'false'}</span>
    </div>`;
    body = barsHtml([['true', p], ['false', 1 - p]], (k) => (q && q.criteria && q.criteria[k]) || '');
  } else if (type === 'choice') {
    const entries = Object.entries(a.probabilities).sort((x, y) => y[1] - x[1]);
    head = `<div class="headline">
      <span class="headline-val">${esc(a.choice)}</span>
      <span class="headline-sub">confidence ${pct(a.confidence)}</span>
    </div>`;
    body = barsHtml(entries, (k) => (q && q.criteria && q.criteria[k]) || '');
  } else {
    const legend = a.legend || {};
    const n = Object.keys(a.probabilities).length;
    const frac = n > 1 ? a.score / (n - 1) : 0;
    const entries = Object.entries(a.probabilities).sort((x, y) => Number(x[0]) - Number(y[0]));
    head = `<div class="headline">
        <span class="headline-val">${a.score.toFixed(2)}</span>
        <span class="headline-lab">${esc(legend[String(Math.round(a.score))] || '')}</span>
        <span class="headline-sub">expected level of ${n - 1} · confidence ${pct(a.confidence)}</span>
      </div>
      <div class="gauge">
        <div class="gauge-track"><div class="gauge-needle" style="left:${(frac * 100).toFixed(2)}%"></div></div>
        <div class="gauge-scale"><span>0 · ${esc(legend['0'] || '')}</span><span>${n - 1} · ${esc(legend[String(n - 1)] || '')}</span></div>
      </div>`;
    body = barsHtml(entries, (k) => legend[k] || '');
  }

  return `<div class="ans">
    <div class="ans-head">
      <span class="ans-id">${esc(id)}</span>
      <span class="ans-type">${esc(type)}</span>
    </div>
    ${q && q.instructions ? `<p class="ans-q">${esc(q.instructions)}</p>` : ''}
    ${head}${body}
  </div>`;
}

function rankHtml(ranked) {
  const max = ranked.length ? ranked[0].prob : 1;
  return `<div class="ans">
    <div class="ans-head"><span class="ans-id">ranking</span><span class="ans-type">rank</span></div>
    ${ranked.map((r, n) => `<div class="rank-row ${n === 0 ? 'is-top' : ''}">
        <div class="bar-fill" style="width:${(r.prob / max * 100).toFixed(2)}%"></div>
        <span class="rank-n">${r.rank}</span>
        <span class="rank-text">${esc(r.candidate)}</span>
        <span class="rank-pct">${pct(r.prob)}</span>
      </div>`).join('')}
  </div>`;
}

function resultBlock(req, res) {
  if (req.endpoint === '/v1/rank') return rankHtml(res.json.ranked || []);
  const answers = res.json.answers || {};
  return Object.entries(answers)
    .map(([id, a]) => answerHtml(id, req.body.questions[id], a))
    .join('');
}

function renderAnswers(req, main) {
  renderMeta(main);
  $('#view-answer').innerHTML = resultBlock(req, main);
  showView('answer');
}

const HINTS = {
  401: ['Unauthorized',
        'This server was started with <code>CLM_API_KEY</code> set. Put the same key into connection settings (the gear icon).'],
  422: ['Invalid request',
        'The server rejected the request. The JSON tab shows exactly what was sent.'],
  502: ['Encoder unreachable',
        'clm-serve is up, but it cannot reach the <code>/v1/embeddings</code> backend that does the embedding. ' +
        'Start the encoder and point <code>--emb-url</code> at it. ' +
        'Apple Silicon / MPS: <code>clm-mps-embed --port 8090</code>. Linux / NVIDIA:' +
        '<pre class="cmd">vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling \\\n     --enable-prefix-caching --max-model-len 2048 --gpu-memory-utilization 0.35 --port 8090</pre>'],
};

function showError(e, body) {
  let title, fix;
  if (e.validation) {
    [title, fix] = ['Incomplete request', ''];          // never left the browser
  } else if (HINTS[e.status]) {
    [title, fix] = HINTS[e.status];
  } else if (e.status) {
    [title, fix] = ['HTTP ' + e.status, ''];
  } else {
    [title, fix] = ['Could not reach the server',
      `Is <code>clm-serve</code> running at <code>${esc(baseUrl() || location.origin)}</code>? ` +
      'A server on another host also has to allow this page to call it: <code>clm-serve --cors</code>'];
  }
  $('#view-answer').innerHTML = `<div class="err">
      <div class="err-title">${esc(title)}</div>
      <div class="err-msg">${esc(e.message)}</div>
      ${fix ? `<div class="err-fix">${fix}</div>` : ''}
    </div>`;
  if (body) renderJson(body);
  showView('answer');
  $('#meta').hidden = true;
}

/* ───────────────────────────── json / code views ───────────────────────────── */

function highlight(obj) {
  return JSON.stringify(obj, null, 2)
    .replace(/[&<>]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' }[c]))
    .replace(/"([^"\\]*(?:\\.[^"\\]*)*)"(\s*:)?/g,
      (m, _s, colon) => `<span class="${colon ? 'k' : 's'}">${m.replace(/:$/, '')}</span>${colon || ''}`)
    .replace(/\b(-?\d+\.?\d*(?:e[-+]?\d+)?)\b/gi, '<span class="n">$1</span>')
    .replace(/\b(true|false|null)\b/g, '<span class="n">$1</span>');
}

function renderJson(obj) {
  $('#view-json').innerHTML = highlight(obj);
}

function shQuote(s) {
  return "'" + String(s).replace(/'/g, `'\\''`) + "'";
}

function pyRepr(v, indent) {
  const pad = ' '.repeat(indent);
  if (typeof v === 'string') {
    if (v.includes('\n')) return '"""' + v.replace(/\\/g, '\\\\').replace(/"""/g, '\\"\\"\\"') + '"""';
    return JSON.stringify(v);
  }
  if (v === null) return 'None';
  if (typeof v === 'boolean') return v ? 'True' : 'False';
  if (Array.isArray(v)) return '[' + v.map((x) => pyRepr(x, indent)).join(', ') + ']';
  if (typeof v === 'object') {
    const rows = Object.entries(v).map(([k, x]) => `${pad}    ${JSON.stringify(k)}: ${pyRepr(x, indent + 4)}`);
    return '{\n' + rows.join(',\n') + `\n${pad}}`;
  }
  return String(v);
}

function renderCode() {
  let req;
  try { req = buildRequest(); } catch (e) {
    const msg = '# ' + e.message;
    $('#view-curl').textContent = msg;
    $('#view-python').textContent = msg;
    return;
  }
  const base = baseUrl() || location.origin;
  const auth = apiKey() ? ` \\\n  -H "Authorization: Bearer $CLM_API_KEY"` : '';
  $('#view-curl').textContent =
    `curl -s ${base}${req.endpoint} \\\n  -H 'Content-Type: application/json'${auth} \\\n  -d ${shQuote(JSON.stringify(req.body, null, 2))}`;

  if (req.endpoint === '/v1/rank') {
    $('#view-python').textContent =
`# pip install -r requirements.txt  (from a clone of the CLM repo)
from clm import CLMClient

client = CLMClient(base_url=${JSON.stringify(base)}${apiKey() ? ', api_key=os.environ["CLM_API_KEY"]' : ''})
ranked = client.rank(
    context=${pyRepr(req.body.context, 4)},
    question=${pyRepr(req.body.question, 4)},
    answers=${pyRepr(req.body.answers, 4)},${req.body.model !== 'clm-latest' ? `\n    model=${JSON.stringify(req.body.model)},` : ''}
)
for r in ranked:
    print(r["rank"], round(r["prob"], 4), r["candidate"])`;
    return;
  }

  const qs = Object.entries(req.body.questions).map(([id, q]) => {
    const cls = { noul: 'Noul', choice: 'Choice', score: 'Score' }[q.type];
    const args = [`instructions=${pyRepr(q.instructions, 8)}`];
    if (q.criteria !== undefined) args.push(`criteria=${pyRepr(q.criteria, 8)}`);
    return `        ${JSON.stringify(id)}: ${cls}(${args.join(', ')})`;
  }).join(',\n');

  const kinds = [...new Set(Object.values(req.body.questions).map((q) => q.type))]
    .map((t) => ({ noul: 'Noul', choice: 'Choice', score: 'Score' }[t])).sort();

  $('#view-python').textContent =
`# pip install -r requirements.txt  (from a clone of the CLM repo)
from clm import CLMClient, ${kinds.join(', ')}

client = CLMClient(base_url=${JSON.stringify(base)}${apiKey() ? ', api_key=os.environ["CLM_API_KEY"]' : ''})
r = client.system_one(
    state=${pyRepr(req.body.state, 4)},
    questions={
${qs}
    },${req.body.model !== 'clm-latest' ? `\n    model=${JSON.stringify(req.body.model)},` : ''}
)
for name, a in r.answers.items():
    print(name, a)`;
}

function showView(name) {
  $$('.pane-res .tab').forEach((t) => t.classList.toggle('is-active', t.dataset.view === name));
  for (const v of ['answer', 'json', 'curl', 'python']) $('#view-' + v).hidden = v !== name;
  $('.pane-body-res').scrollTop = 0;
  $('#copy-view').hidden = name === 'answer';
  if (name === 'curl' || name === 'python') renderCode();
  if (name === 'json' && !lastResult) renderJsonPreview();
}

function renderJsonPreview() {
  try { renderJson(buildRequest().body); } catch (e) { $('#view-json').textContent = '// ' + e.message; }
}

/* ───────────────────────────── question editor ───────────────────────────── */

const Q_HELP = {
  noul: 'Answered with one probability: how true the statement is of the state.',
  choice: 'The action head embeds each description verbatim — or the key itself, if the description is blank. Keys only name the answer.',
  score: 'Ordered rubric, each level embedded verbatim. The answer is the expected level index, so keep the levels monotonic.',
};

function renderQuestions() {
  const wrap = $('#questions');
  wrap.innerHTML = '';
  S.questions.forEach((q, i) => wrap.appendChild(questionCard(q, i)));
}

function questionCard(q, i) {
  const el = $('#tpl-question').content.firstElementChild.cloneNode(true);
  el.dataset.type = q.type;
  $('.q-type', el).textContent = q.type;
  $('.q-help', el).textContent = Q_HELP[q.type];

  const idIn = $('.q-id', el);
  idIn.value = q.id;
  idIn.addEventListener('input', () => { q.id = idIn.value; save(); });

  const sel = $('.q-type-sel', el);
  sel.value = q.type;
  sel.addEventListener('change', () => { q.type = sel.value; renderQuestions(); save(); });

  const ins = $('.q-instructions', el);
  ins.value = q.instructions;
  ins.placeholder = q.type === 'noul' ? 'A statement to test — "Is this urgent?"'
    : q.type === 'choice' ? 'The question — "Which team should handle this?"'
    : 'What is being rated — "How frustrated is the customer?"';
  ins.addEventListener('input', () => { q.instructions = ins.value; save(); });

  $('.q-del', el).addEventListener('click', () => {
    S.questions.splice(i, 1); renderQuestions(); save();
  });

  const crit = $('.q-criteria', el);
  const add = $('.q-add', el);

  if (q.type === 'noul') {
    add.hidden = false;
    add.textContent = q.showNoulCrit ? '− descriptions' : '+ descriptions';
    add.addEventListener('click', () => {
      q.showNoulCrit = !q.showNoulCrit;
      if (!q.showNoulCrit) q.noulCrit = { true: '', false: '' };
      renderQuestions(); save();
    });
    if (q.showNoulCrit) {
      for (const k of ['true', 'false']) {
        crit.appendChild(critRow(
          { fixed: k, value: q.noulCrit[k], placeholder: `what "${k}" looks like (optional)` },
          (v) => { q.noulCrit[k] = v; save(); }));
      }
    }
  } else if (q.type === 'choice') {
    add.hidden = false;
    add.textContent = '+ option';
    add.addEventListener('click', () => { q.choiceCrit.push({ key: '', desc: '' }); renderQuestions(); save(); });
    q.choiceCrit.forEach((row, j) => {
      crit.appendChild(critRow(
        { key: row.key, value: row.desc, placeholder: 'what this option means', removable: q.choiceCrit.length > 2 },
        (v) => { row.desc = v; save(); },
        (k) => { row.key = k; save(); },
        () => { q.choiceCrit.splice(j, 1); renderQuestions(); save(); }));
    });
  } else {
    add.hidden = false;
    add.textContent = '+ level';
    add.addEventListener('click', () => { q.scoreCrit.push(''); renderQuestions(); save(); });
    q.scoreCrit.forEach((lvl, j) => {
      crit.appendChild(critRow(
        { index: j, value: lvl, placeholder: `level ${j}`, removable: q.scoreCrit.length > 2 },
        (v) => { q.scoreCrit[j] = v; save(); },
        null,
        () => { q.scoreCrit.splice(j, 1); renderQuestions(); save(); }));
    });
  }
  return el;
}

function critRow(opt, onDesc, onKey, onRemove) {
  const row = document.createElement('div');
  row.className = 'crit-row';

  if (opt.index !== undefined) {
    const n = document.createElement('span');
    n.className = 'crit-idx';
    n.textContent = opt.index;
    row.appendChild(n);
  } else if (opt.fixed) {
    const n = document.createElement('input');
    n.className = 'crit-key';
    n.value = opt.fixed;
    n.disabled = true;
    row.appendChild(n);
  } else {
    const k = document.createElement('input');
    k.className = 'crit-key';
    k.type = 'text';
    k.spellcheck = false;
    k.placeholder = 'option key';
    k.value = opt.key || '';
    k.addEventListener('input', () => onKey(k.value));
    row.appendChild(k);
  }

  const d = document.createElement('input');
  d.className = 'crit-desc';
  d.type = 'text';
  d.placeholder = opt.placeholder || '';
  d.value = opt.value || '';
  d.addEventListener('input', () => onDesc(d.value));
  row.appendChild(d);

  if (opt.removable && onRemove) {
    const b = document.createElement('button');
    b.className = 'icon-btn small';
    b.type = 'button';
    b.title = 'Remove';
    b.innerHTML = '<svg viewBox="0 0 24 24"><path d="M18 6 6 18M6 6l12 12"/></svg>';
    b.addEventListener('click', onRemove);
    row.appendChild(b);
  }
  return row;
}

/* ───────────────────────────── sync form <-> state ───────────────────────────── */

function paint() {
  $('#state').value = S.state;
  $('#state-json').checked = S.stateJson;
  $('#rank-instructions').value = S.rankInstructions;
  $('#rank-candidates').value = S.rankCandidates;
  $('#model').value = S.model;
  $('#mode-ask').hidden = S.mode !== 'ask';
  $('#mode-rank').hidden = S.mode !== 'rank';
  $$('.pane-req .tab').forEach((t) => t.classList.toggle('is-active', t.dataset.mode === S.mode));
  updateCount();
  renderQuestions();
}

function updateCount() {
  const n = S.state.length;
  $('#state-count').textContent = n.toLocaleString() + ' char' + (n === 1 ? '' : 's');
  const err = $('#state-json-err');
  if (S.stateJson && S.state.trim()) {
    try { JSON.parse(S.state); err.hidden = true; }
    catch (e) { err.textContent = e.message; err.hidden = false; }
  } else err.hidden = true;
}

function save() {
  try { localStorage.setItem(LS.req, JSON.stringify(S)); } catch (e) { /* private mode */ }
}

/* ───────────────────────────── share links ───────────────────────────── */

const b64u = {
  enc: (s) => btoa(String.fromCharCode(...new TextEncoder().encode(s))).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, ''),
  dec: (s) => new TextDecoder().decode(Uint8Array.from(atob(s.replace(/-/g, '+').replace(/_/g, '/')), (c) => c.charCodeAt(0))),
};

function shareLink() {
  const u = new URL(location.href);
  u.hash = 'r=' + b64u.enc(JSON.stringify(S));
  return u.toString();
}

function loadFromHash() {
  const m = /(?:^|[#&])r=([^&]+)/.exec(location.hash);
  if (!m) return false;
  try { S = hydrate(JSON.parse(b64u.dec(m[1]))); return true; }
  catch (e) { return false; }
}

async function copy(text, what) {
  try {
    await navigator.clipboard.writeText(text);
    toast(what + ' copied');
  } catch (e) {
    const ta = document.createElement('textarea');
    ta.value = text;
    document.body.appendChild(ta);
    ta.select();
    try { document.execCommand('copy'); toast(what + ' copied'); }
    catch (e2) { toast('Copy failed — select the text manually'); }
    ta.remove();
  }
}

let toastTimer = null;
function toast(msg) {
  const t = $('#toast');
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 1800);
}

/* ───────────────────────────── theme ───────────────────────────── */

const MOON = '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>';
const SUN = '<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41' +
            'M2 12h2M20 12h2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41"/>';

function applyTheme(t) {
  document.documentElement.dataset.theme = t;
  const b = $('#btn-theme');
  b.querySelector('svg').innerHTML = t === 'dark' ? SUN : MOON;
  b.title = t === 'dark' ? 'Switch to light' : 'Switch to dark';
}

/* ───────────────────────────── wiring ───────────────────────────── */

function loadPreset(name) {
  const p = PRESETS[name];
  if (!p) return;
  const keep = { model: S.model };
  S = hydrate({ ...p, ...keep });
  paint();
  save();
  $('#view-answer').innerHTML =
    `<div class="empty"><p>Loaded <b>${esc(p.label)}</b>.</p><p class="empty-sub">Press <kbd>\u2318</kbd><kbd>\u21b5</kbd> to run it.</p></div>`;
  $('#meta').hidden = true;
  lastResult = null;
}

function init() {
  // theme: light is the default, '' or anything but 'dark' means light
  applyTheme(localStorage.getItem(LS.theme) === 'dark' ? 'dark' : '');
  $('#btn-theme').addEventListener('click', () => {
    const now = document.documentElement.dataset.theme === 'dark' ? '' : 'dark';
    applyTheme(now);
    localStorage.setItem(LS.theme, now);
  });

  // connection settings
  $('#base-url').value = localStorage.getItem(LS.base) || '';
  $('#api-key').value = localStorage.getItem(LS.key) || '';
  $('#btn-settings').addEventListener('click', () => { $('#settings').hidden = !$('#settings').hidden; });
  for (const [el, k] of [[$('#base-url'), LS.base], [$('#api-key'), LS.key]]) {
    el.addEventListener('change', () => { localStorage.setItem(k, el.value); refreshStatus(); renderCode(); });
  }
  $('#status').addEventListener('click', refreshStatus);

  // presets
  const psel = $('#preset');
  for (const [k, p] of Object.entries(PRESETS)) {
    const o = document.createElement('option');
    o.value = k; o.textContent = p.label;
    psel.appendChild(o);
  }
  psel.addEventListener('change', () => { if (psel.value) loadPreset(psel.value); psel.value = ''; });

  // restore: url hash > localStorage > default preset
  if (!loadFromHash()) {
    let stored = null;
    try { stored = JSON.parse(localStorage.getItem(LS.req) || 'null'); } catch (e) { /* ignore */ }
    S = stored ? hydrate(stored) : hydrate(PRESETS.support);
  }
  paint();

  // mode tabs
  $$('.pane-req .tab').forEach((t) => t.addEventListener('click', () => {
    S.mode = t.dataset.mode; paint(); save(); renderCode();
  }));
  $$('.pane-res .tab').forEach((t) => t.addEventListener('click', () => showView(t.dataset.view)));

  // fields
  $('#state').addEventListener('input', (e) => { S.state = e.target.value; updateCount(); save(); });
  $('#state-json').addEventListener('change', (e) => { S.stateJson = e.target.checked; updateCount(); save(); });
  $('#rank-instructions').addEventListener('input', (e) => { S.rankInstructions = e.target.value; save(); });
  $('#rank-candidates').addEventListener('input', (e) => { S.rankCandidates = e.target.value; save(); });
  $('#model').addEventListener('change', (e) => { S.model = e.target.value; save(); });

  $$('[data-add]').forEach((b) => b.addEventListener('click', () => {
    S.questions.push(newQuestion(b.dataset.add));
    renderQuestions(); save();
    const cards = $$('.q-card');
    const last = cards[cards.length - 1];
    if (last) $('.q-instructions', last).focus();
  }));

  // actions
  $('#run').addEventListener('click', run);
  $('#share').addEventListener('click', () => {
    const link = shareLink();
    history.replaceState(null, '', link);
    copy(link, 'Link');
  });
  $('#copy-view').addEventListener('click', () => {
    const name = $$('.pane-res .tab').find((t) => t.classList.contains('is-active')).dataset.view;
    copy($('#view-' + name).textContent, name === 'json' ? 'JSON' : name === 'curl' ? 'curl command' : 'Snippet');
  });
  document.addEventListener('keydown', (e) => {
    if ((e.metaKey || e.ctrlKey) && e.key === 'Enter') { e.preventDefault(); run(); }
  });

  refreshStatus();
}

document.addEventListener('DOMContentLoaded', init);
