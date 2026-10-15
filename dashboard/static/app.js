const state = {
  data: null,
  category: "all",
  search: "",
  activeSection: "extractor",
  selectedRuns: {},
  // Register-only filters: column filters by column key, and a histogram bin selection.
  register: { extractor: {}, inference: {} },
  binFilter: { extractor: null, inference: null },
  sort: { extractor: null, inference: null },
};

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

const money = (value) => new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  minimumFractionDigits: value >= 100 ? 0 : 2,
  maximumFractionDigits: value >= 100 ? 0 : 4,
}).format(Number(value || 0));

const compact = (value) => new Intl.NumberFormat("en-US", {
  notation: "compact",
  maximumFractionDigits: 1,
}).format(Number(value || 0));

const fullNumber = (value) => new Intl.NumberFormat("en-US").format(Number(value || 0));

const runDate = (value) => {
  if (!value) return "Not cached yet";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? value : date.toLocaleString();
};

function runFolderUrl(run) {
  if (!run || run.target !== "sharepoint" || !run.site_url || !run.folder_path) return "";
  const encodedPath = String(run.folder_path)
    .split("/")
    .filter(Boolean)
    .map(encodeURIComponent)
    .join("/");
  return `${String(run.site_url).replace(/\/$/, "")}/${encodedPath}`;
}

const escapeHtml = (value) => String(value ?? "")
  .replaceAll("&", "&amp;")
  .replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;")
  .replaceAll('"', "&quot;")
  .replaceAll("'", "&#039;");

function sumMetrics(rows) {
  const total = {};
  for (const row of rows) {
    for (const [key, value] of Object.entries(row.metrics || {})) {
      total[key] = (total[key] || 0) + Number(value || 0);
    }
  }
  return total;
}

function visibleRows(section) {
  if (!state.data) return [];
  const rows = state.data.sections[section]?.cases || [];
  const query = state.search.trim().toLowerCase();
  return rows.filter((row) => {
    const runMatch = !state.selectedRuns[section] || row.run_key === state.selectedRuns[section];
    const categoryMatch = state.category === "all" || row.category === state.category;
    const searchMatch = !query || [row.asset, row.year, row.exception_id, row.category]
      .join(" ").toLowerCase().includes(query);
    return runMatch && categoryMatch && searchMatch;
  });
}

function metricCard(label, value, note, kind = "") {
  return `<article class="metric-card ${kind}">
    <span class="metric-label">${escapeHtml(label)}</span>
    <strong class="metric-value">${escapeHtml(value)}</strong>
    <small class="metric-note">${escapeHtml(note)}</small>
  </article>`;
}

function renderGlobalMetrics() {
  const rows = ["extractor", "inference"].flatMap(visibleRows);
  const metrics = sumMetrics(rows);
  const casesWithRuns = new Set(
    rows.filter((row) => row.metrics.artifacts > 0).map((row) => row.key),
  ).size;
  $("#heroSpend").textContent = money(metrics.total_cost_usd);
  $("#heroSavings").textContent = `${money(metrics.cache_savings_usd)} saved by caching`;
  $("#globalMetrics").innerHTML = [
    metricCard("Total spend", money(metrics.total_cost_usd), `${metrics.priced_artifacts || 0} priced output files`, "accent"),
    metricCard("Cache savings", money(metrics.cache_savings_usd), `${money(metrics.uncached_cost_usd)} uncached equivalent`, "saving"),
    metricCard("Total tokens", compact(metrics.total_tokens), `${compact(metrics.prompt_tokens)} prompt · ${compact(metrics.output_tokens)} output`),
    metricCard("Cases with outputs", fullNumber(casesWithRuns), `${fullNumber(metrics.artifacts)} rebuttal JSONs across both sections`),
  ].join("");
}

// Token billing classes. Colors follow the class everywhere (mix bar, histogram) and are CSS
// variables, so each theme supplies its own validated step without a redraw.
const TOKEN_CLASSES = [
  { key: "total_tokens", label: "All tokens", short: "All", color: "var(--series-total)" },
  { key: "input_tokens", label: "New input", short: "Input", color: "var(--series-input)" },
  { key: "cache_read_tokens", label: "Cache read", short: "Cache read", color: "var(--series-cache-read)" },
  { key: "cache_write_tokens", label: "Cache write", short: "Cache write", color: "var(--series-cache-write)" },
  { key: "output_tokens", label: "Output", short: "Output", color: "var(--series-output)" },
  { key: "thinking_tokens", label: "Thinking (subset of output)", short: "Thinking", color: "var(--series-thinking)" },
];
const tokenClass = (key) => TOKEN_CLASSES.find((item) => item.key === key) || TOKEN_CLASSES[0];

// Dollar spend by billing class; each reuses its token class color so the two cards read as a pair.
const COST_CLASSES = [
  { key: "total_cost_usd", label: "Total spend", short: "Total", color: "var(--series-total)" },
  { key: "input_cost_usd", label: "New input spend", short: "Input", color: "var(--series-input)" },
  { key: "cache_read_cost_usd", label: "Cache read spend", short: "Cache read", color: "var(--series-cache-read)" },
  { key: "cache_write_cost_usd", label: "Cache write spend", short: "Cache write", color: "var(--series-cache-write)" },
  { key: "output_cost_usd", label: "Output spend", short: "Output", color: "var(--series-output)" },
];
const costClass = (key) => COST_CLASSES.find((item) => item.key === key) || COST_CLASSES[0];

const TIER_ORDER = ["light", "med", "medium", "high"];
const tierLabel = (name) => ({ light: "Light", med: "Medium", medium: "Medium", high: "High" }[name]
  || name.charAt(0).toUpperCase() + name.slice(1));

function tokenMix(metrics) {
  const pieces = TOKEN_CLASSES.slice(1, 5).map((item) => [item.label, Number(metrics[item.key] || 0), item.color]);
  const total = pieces.reduce((acc, item) => acc + item[1], 0) || 1;
  return `<div class="mix-bar" role="img" aria-label="Token mix: ${pieces.map(([label, value]) => `${label} ${compact(value)}`).join(", ")}">
      ${pieces.filter(([, value]) => value > 0).map(([label, value, color]) => `<span title="${label} · ${fullNumber(value)} (${((value / total) * 100).toFixed(1)}%)" style="flex-grow:${value};background:${color}"></span>`).join("")}
    </div>
    <div class="mix-legend">
      ${pieces.map(([label, value, color]) => `<div><i style="background:${color}"></i><span>${label}</span><strong>${compact(value)}</strong><em>${((value / total) * 100).toFixed(0)}%</em></div>`).join("")}
    </div>`;
}

// ---------------------------------------------------------------------------
// Distribution histograms
// ---------------------------------------------------------------------------

// Scale is null until the user picks one; null means "auto" (log for long-tailed data).
state.dist = {
  spendClass: "total_cost_usd", spendTier: "all", tokenClass: "total_tokens", tokenTier: "all",
  spendScale: null, tokensScale: null, spendTable: false, tokensTable: false,
  // Hypothetical per-case caps for the coverage estimate (Total views only). Nothing is enforced.
  spendCap: null, tokensCap: null,
};

function niceStep(raw) {
  if (!(raw > 0)) return 1;
  const exponent = 10 ** Math.floor(Math.log10(raw));
  const fraction = raw / exponent;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 2.5 ? 2.5 : fraction <= 5 ? 5 : 10;
  return nice * exponent;
}

function quantile(sorted, q) {
  if (!sorted.length) return 0;
  const position = (sorted.length - 1) * q;
  const base = Math.floor(position);
  const rest = position - base;
  return sorted[base + 1] === undefined ? sorted[base] : sorted[base] + rest * (sorted[base + 1] - sorted[base]);
}

function sectionTiers(rows) {
  const names = new Set(rows.flatMap((row) => Object.keys(row.tiers || {})));
  const models = {};
  for (const row of rows) {
    for (const [name, tier] of Object.entries(row.tiers || {})) {
      if (tier.model && !models[name]) models[name] = tier.model;
    }
  }
  const rank = (name) => (TIER_ORDER.includes(name) ? TIER_ORDER.indexOf(name) : 99);
  return [...names]
    .sort((a, b) => rank(a) - rank(b) || a.localeCompare(b))
    .map((name) => ({ name, label: tierLabel(name), model: models[name] || "" }));
}

function distributionRows(section) {
  return visibleRows(section).filter((row) => row.metrics.artifacts > 0);
}

// Returns the per-case values plus everything the card needs to describe them.
function distributionSeries(section, kind) {
  const rows = distributionRows(section);
  const tiers = sectionTiers(rows);
  const isMoney = kind === "spend";
  const tier = state.dist[isMoney ? "spendTier" : "tokenTier"];
  const active = tier === "all" || tiers.some((item) => item.name === tier) ? tier : "all";
  const cls = isMoney ? costClass(state.dist.spendClass) : tokenClass(state.dist.tokenClass);
  const scoped = active === "all" ? rows : rows.filter((row) => row.tiers?.[active]);
  const values = scoped.map((row) => Math.max(Number(
    active === "all" ? row.metrics[cls.key] : row.tiers[active][cls.key],
  ) || 0, 0));
  const meta = tiers.find((item) => item.name === active);
  return {
    kind, rows: scoped, values, tiers, active, cls, isMoney,
    color: cls.color,
    measure: `${cls.label} per case`,
    detail: active === "all" ? "All model tiers combined" : `${meta.label} tier${meta.model ? ` · ${meta.model}` : ""}`,
  };
}

// Readable money at any magnitude: $0.0042, $0.858, $15.18, $1,184.
function moneyShort(value) {
  const amount = Number(value || 0);
  const digits = amount === 0 ? 0 : amount < 0.01 ? 4 : amount < 1 ? 3 : amount < 100 ? 2 : 0;
  return new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", minimumFractionDigits: digits ? Math.min(digits, 2) : 0, maximumFractionDigits: digits,
  }).format(amount);
}

function edgeLabel(value, isMoney) {
  if (!isMoney) return compact(value);
  const digits = value >= 10 ? 0 : value >= 1 ? 1 : value >= 0.1 ? 2 : value >= 0.01 ? 3 : 4;
  return new Intl.NumberFormat("en-US", {
    style: "currency", currency: "USD", minimumFractionDigits: 0, maximumFractionDigits: digits,
  }).format(value);
}

function linearBins(values, width) {
  const minimum = Math.min(...values);
  const maximum = Math.max(...values);
  const target = Math.max(6, Math.min(24, Math.round(width / 38)));
  const step = niceStep((maximum - minimum) / target || maximum / target || 1);
  const lower = Math.floor(minimum / step) * step;
  const count = maximum === lower ? 1 : Math.max(1, Math.ceil((maximum - lower) / step - 1e-9));
  const edges = Array.from({ length: count + 1 }, (_, index) => lower + index * step);
  return edges;
}

// Log bins on a 1-2-5 (or finer 1-1.5-2-3-5-7) ladder so every edge is a round number.
function logBins(values, width) {
  const positive = values.filter((value) => value > 0);
  const minimum = Math.min(...positive);
  const maximum = Math.max(...positive);
  const decades = Math.log10(maximum) - Math.log10(minimum);
  const ladders = [[1, 1.5, 2, 3, 5, 7], [1, 2, 5], [1, 3], [1]];
  for (const ladder of ladders) {
    const edges = [];
    for (let power = Math.floor(Math.log10(minimum)) - 1; edges.length < 400; power += 1) {
      for (const mult of ladder) {
        const edge = Number((mult * 10 ** power).toPrecision(6));
        if (edge <= minimum) edges.length = 0;
        edges.push(edge);
        if (edge > maximum) break;
      }
      if (edges[edges.length - 1] > maximum) break;
    }
    const binCount = edges.length - 1;
    if (width / binCount >= 14 || ladder.length === 1 || decades > 6) return edges;
  }
  return [minimum, maximum];
}

// Bins are [lower, upper) except the last, which also takes the maximum. Log bins drop zeros.
function binIndexOf(value, bins, scale) {
  if (scale === "log" && value <= 0) return -1;
  const index = bins.findIndex((bin) => value < bin.upper);
  return index === -1 ? bins.length - 1 : index;
}

function buildBins(values, edges, scale) {
  const bins = edges.slice(0, -1).map((lower, index) => ({ lower, upper: edges[index + 1], count: 0, sum: 0 }));
  for (const value of values) {
    const index = binIndexOf(value, bins, scale);
    if (index === -1) continue;
    bins[index].count += 1;
    bins[index].sum += value;
  }
  return bins;
}

function distributionStats(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const total = values.reduce((acc, value) => acc + value, 0);
  return {
    count: values.length,
    zeros: values.filter((value) => value <= 0).length,
    total,
    mean: values.length ? total / values.length : 0,
    median: quantile(sorted, 0.5),
    p90: quantile(sorted, 0.9),
    max: sorted[sorted.length - 1] || 0,
  };
}

function segmented(name, options, active, ariaLabel, extraClass = "") {
  return `<div class="segmented ${extraClass}" role="radiogroup" aria-label="${escapeHtml(ariaLabel)}">
    ${options.map((option) => `<button type="button" role="radio" aria-checked="${option.value === active}" class="${option.value === active ? "is-active" : ""}" data-dist-control="${name}" data-value="${escapeHtml(option.value)}" ${option.title ? `title="${escapeHtml(option.title)}"` : ""}>${option.swatch ? `<i style="background:${option.swatch}"></i>` : ""}${escapeHtml(option.label)}</button>`).join("")}
  </div>`;
}

function distributionControls(series) {
  const scaleControl = segmented(`${series.kind}Scale`, [
    { value: "log", label: "Normalized", title: "Logarithmic bins — spreads out long right tails" },
    { value: "linear", label: "Raw", title: "Equal-width bins on the raw value" },
  ], series.scale, "X-axis scale", "segmented-compact");
  const isMoney = series.kind === "spend";
  const classes = isMoney ? COST_CLASSES : TOKEN_CLASSES;
  // No tier data means the server predates per-tier aggregation; say so instead of a silent one-option list.
  const tierSelect = series.tiers.length
    ? `<label class="tier-select"><span>Tier</span><select data-dist-control="${isMoney ? "spendTier" : "tokenTier"}" aria-label="Model tier">
        <option value="all" ${series.active === "all" ? "selected" : ""}>All tiers</option>
        ${series.tiers.map((tier) => `<option value="${escapeHtml(tier.name)}" ${tier.name === series.active ? "selected" : ""}>${escapeHtml(tier.label)}${tier.model ? ` · ${escapeHtml(tier.model)}` : ""}</option>`).join("")}
      </select></label>`
    : `<span class="tier-missing" title="Restart the dashboard server to load per-tier metrics">No tier data — restart server</span>`;
  return `${segmented(isMoney ? "spendClass" : "tokenClass", classes.map((item) => ({
    value: item.key, label: item.short, swatch: item.color, title: item.label,
  })), series.cls.key, isMoney ? "Spend class" : "Token class")}<div class="control-end">${tierSelect}${scaleControl}</div>`;
}

function withScale(series) {
  const chosen = state.dist[`${series.kind}Scale`];
  series.scale = chosen || "log";
  if (series.scale === "log" && !series.values.some((value) => value > 0)) series.scale = "linear";
  return series;
}

// ---------------------------------------------------------------------------
// Cap coverage estimate: what share of past cases finished at or under a per-case cap
// ---------------------------------------------------------------------------

const isTotalView = (series) => series.cls.key === "total_cost_usd" || series.cls.key === "total_tokens";
const capKey = (kind) => `${kind}Cap`;
const formatCap = (value, isMoney) => (value == null ? "" : isMoney ? moneyShort(value) : compact(value));

// "$5", "0.5", "2m", "1,500" → number; NaN when unparseable.
function parseAmount(text) {
  const match = String(text || "").toLowerCase().replace(/[\s,$]/g, "").match(/^(-?\d*\.?\d+)([kmb])?$/);
  return match ? Number(match[1]) * ({ k: 1e3, m: 1e6, b: 1e9 }[match[2]] || 1) : NaN;
}

// Wilson score interval: an honest range for a proportion from a small sample.
function wilson(successes, trials, z = 1.96) {
  if (!trials) return [0, 0];
  const p = successes / trials;
  const denominator = 1 + (z * z) / trials;
  const center = (p + (z * z) / (2 * trials)) / denominator;
  const half = (z * Math.sqrt((p * (1 - p)) / trials + (z * z) / (4 * trials * trials))) / denominator;
  return [Math.max(0, center - half), Math.min(1, center + half)];
}

function capStats(values, cap) {
  const total = values.reduce((acc, value) => acc + value, 0);
  const over = values.filter((value) => value > cap);
  const overSum = over.reduce((acc, value) => acc + value, 0);
  const covered = values.length - over.length;
  const [low, high] = wilson(covered, values.length);
  return {
    n: values.length, covered, share: values.length ? covered / values.length : 0, low, high,
    underSum: total - overSum, underShare: total ? (total - overSum) / total : 0,
    overCount: over.length, overSum, overShare: total ? overSum / total : 0,
  };
}

// Smallest observed value that covers at least the target share of cases.
function capForCoverage(values, target) {
  const sorted = [...values].sort((a, b) => a - b);
  if (!sorted.length) return null;
  return sorted[Math.min(Math.max(Math.ceil(target * sorted.length) - 1, 0), sorted.length - 1)];
}

const percent = (value) => `${Math.round(value * 100)}%`;

function capReadout(series, cap) {
  if (cap == null) {
    return `<p class="cap-hint">Type a cap, or a coverage target, to estimate how many past cases would have finished under it.</p>`;
  }
  const result = capStats(series.values, cap);
  const unit = series.isMoney ? "spend" : "tokens";
  const amount = (value) => (series.isMoney ? moneyShort(value) : `${compact(value)} tokens`);
  const cases = (count) => `${fullNumber(count)} case${count === 1 ? "" : "s"}`;
  // Share of the section's total spend/tokens on each side of the cap; widths are value-weighted.
  const split = (label, count, sum, share, className) => `<div class="cap-split-row ${className}">
      <i aria-hidden="true"></i><span>${label}</span><strong>${cases(count)}</strong>
      <span>${escapeHtml(amount(sum))}</span><b>${percent(share)} of ${unit}</b>
    </div>`;
  return `<p class="cap-result"><strong>${percent(result.share)}</strong> of cases covered <span>${fullNumber(result.covered)} of ${fullNumber(result.n)}</span>
      <span class="cap-range" title="95% Wilson interval: the likely coverage for future cases like these, given only ${fullNumber(result.n)} observed">likely ${percent(result.low)}–${percent(result.high)}</span></p>
    <div class="cap-split-bar" role="img" aria-label="${escapeHtml(`Under the cap: ${percent(result.underShare)} of ${unit}. Over the cap: ${percent(result.overShare)} of ${unit}.`)}">
      ${result.underSum > 0 ? `<span class="is-under" style="flex-grow:${result.underSum}"></span>` : ""}${result.overSum > 0 ? `<span class="is-over" style="flex-grow:${result.overSum}"></span>` : ""}
    </div>
    <div class="cap-split">
      ${split("Under cap", result.covered, result.underSum, result.underShare, "is-under")}
      ${split("Over cap", result.overCount, result.overSum, result.overShare, "is-over")}
    </div>`;
}

function capPanel(series) {
  const kind = series.kind;
  const cap = state.dist[capKey(kind)];
  return `<div class="cap-panel" data-cap-panel="${kind}">
      <div class="cap-inputs">
        <label class="cap-field"><span>Cap per case</span><input type="text" data-cap-input="${kind}" value="${escapeHtml(formatCap(cap, series.isMoney))}" placeholder="${series.isMoney ? "e.g. $5" : "e.g. 2m"}" inputmode="decimal" autocomplete="off" aria-label="Hypothetical cap per case"/></label>
        <span class="cap-or">or</span>
        <label class="cap-field"><span>Cover</span><input type="text" class="cap-target" data-cap-target="${kind}" placeholder="90" inputmode="decimal" autocomplete="off" aria-label="Target share of cases to cover, in percent"/><em>% of cases</em></label>
        <button type="button" class="cap-clear" data-cap-clear="${kind}" ${cap == null ? "hidden" : ""}>Clear</button>
      </div>
      <div class="cap-readout" aria-live="polite">${capReadout(series, cap)}</div>
    </div>`;
}

// Refresh every card of a kind in place, so the input being typed in keeps focus.
function updateCapViews(kind, { redraw = true } = {}) {
  const cap = state.dist[capKey(kind)];
  for (const section of ["extractor", "inference"]) {
    const card = $(`[data-card="${section}-${kind}"]`);
    const panel = card && $(".cap-panel", card);
    if (!panel) continue;
    const series = withScale(distributionSeries(section, kind));
    $(".cap-readout", panel).innerHTML = capReadout(series, cap);
    $(".cap-clear", panel).hidden = cap == null;
    const input = $("[data-cap-input]", panel);
    if (input !== document.activeElement) {
      input.value = formatCap(cap, series.isMoney);
      input.removeAttribute("aria-invalid");
    }
    if (redraw) drawHistogram($(".chart-canvas", card));
  }
}

function distributionCard(section, kind) {
  const series = withScale(distributionSeries(section, kind));
  const stats = distributionStats(series.values);
  const format = series.isMoney ? moneyShort : compact;
  const tableOpen = state.dist[`${kind}Table`];
  const statTiles = [
    ["Cases", fullNumber(stats.count)],
    ["P90", format(stats.p90)],
    ["Max", format(stats.max)],
    ["Total", format(stats.total)],
  ];
  return `<article class="dist-card" data-card="${section}-${kind}">
    <header class="dist-head">
      <div>
        <p class="eyebrow">${kind === "spend" ? "Spend distribution" : "Token distribution"}</p>
        <h3>${escapeHtml(series.measure)}</h3>
        <p class="dist-sub">${escapeHtml(series.detail)}</p>
      </div>
      <button type="button" class="view-toggle" data-dist-control="${kind}Table" aria-pressed="${tableOpen}">${tableOpen ? "Show chart" : "Show table"}</button>
    </header>
    <div class="dist-controls">${distributionControls(series)}</div>
    <dl class="dist-stats">${statTiles.map(([label, value]) => `<div><dt>${label}</dt><dd>${escapeHtml(value)}</dd></div>`).join("")}</dl>
    ${isTotalView(series) ? capPanel(series) : ""}
    <div class="chart-canvas" data-section="${section}" data-kind="${kind}" ${tableOpen ? "data-table" : ""}></div>
  </article>`;
}

const CHART_HEIGHT = 250;

function drawHistogram(canvas) {
  const { section, kind } = canvas.dataset;
  const width = Math.floor(canvas.clientWidth);
  if (!width) return;
  canvas.dataset.drawnWidth = String(width);
  const series = withScale(distributionSeries(section, kind));
  if (!series.values.length) {
    canvas.innerHTML = `<div class="chart-empty">No cases with ${escapeHtml(series.measure.toLowerCase())} for the current filters.</div>`;
    return;
  }
  const isLog = series.scale === "log";
  const margin = { top: 14, right: 16, bottom: 30, left: 38 };
  const plotWidth = width - margin.left - margin.right;
  const plotHeight = CHART_HEIGHT - margin.top - margin.bottom;
  const edges = isLog ? logBins(series.values, plotWidth) : linearBins(series.values, plotWidth);
  const bins = buildBins(series.values, edges, series.scale);
  const stats = distributionStats(series.values);
  const label = (value) => edgeLabel(value, series.isMoney);
  const exact = series.isMoney ? moneyShort : compact;
  const plotted = bins.reduce((acc, bin) => acc + bin.count, 0) || 1;
  const cap = isTotalView(series) ? state.dist[capKey(kind)] : null;
  const overCap = (bin) => cap != null && bin.lower >= cap;
  const selection = state.binFilter[section];
  const selectedIndex = selection?.kind === kind && selection.signature === distSignature(kind)
    ? bins.findIndex((bin) => bin.lower === selection.lower && bin.upper === selection.upper)
    : -1;
  const toggle = (index) => {
    if (!bins[index]?.count) return;
    toggleBinFilter(section, series, bins, index, label);
    const focusTarget = canvas.hasAttribute("data-table") ? `tr[data-index="${index}"]` : `.hit[data-index="${index}"]`;
    $(focusTarget, $(`[data-card="${section}-${kind}"] .chart-canvas`))?.focus();
  };

  if (canvas.hasAttribute("data-table")) {
    canvas.innerHTML = `<div class="bin-table-wrap"><table class="bin-table">
      <thead><tr><th>${series.isMoney ? "Spend" : "Tokens"} per case</th><th>Cases</th><th>Share</th><th>Combined</th></tr></thead>
      <tbody>${bins.map((bin, index) => `<tr data-index="${index}" class="${bin.count ? "is-filterable" : "is-empty"}${index === selectedIndex ? " is-selected" : ""}${overCap(bin) ? " is-over" : ""}" ${bin.count ? `tabindex="0" aria-pressed="${index === selectedIndex}" title="${index === selectedIndex ? "Click to clear the register filter" : "Click to filter the register to these cases"}"` : ""}><td>${escapeHtml(label(bin.lower))} – ${escapeHtml(label(bin.upper))}</td><td>${bin.count}</td><td>${((bin.count / stats.count) * 100).toFixed(1)}%</td><td>${escapeHtml(exact(bin.sum))}</td></tr>`).join("")}</tbody>
    </table></div>`;
    $$("tr.is-filterable", canvas).forEach((row) => {
      const index = Number(row.dataset.index);
      row.addEventListener("click", () => toggle(index));
      row.addEventListener("keydown", (event) => {
        if (event.key !== "Enter" && event.key !== " ") return;
        event.preventDefault();
        toggle(index);
      });
    });
    return;
  }

  const lower = edges[0];
  const upper = edges[edges.length - 1];
  const x = isLog
    ? (value) => margin.left + ((Math.log10(Math.max(value, lower)) - Math.log10(lower)) / (Math.log10(upper) - Math.log10(lower))) * plotWidth
    : (value) => margin.left + ((value - lower) / (upper - lower || 1)) * plotWidth;
  const maxCount = Math.max(...bins.map((bin) => bin.count), 1);
  const yStep = Math.max(1, Math.ceil(niceStep(maxCount / 4)));
  const yMax = Math.ceil(maxCount / yStep) * yStep;
  const y = (count) => margin.top + plotHeight - (count / yMax) * plotHeight;
  const baseline = margin.top + plotHeight;

  const gridLines = [];
  for (let tick = 0; tick <= yMax; tick += yStep) {
    gridLines.push(`<line class="grid${tick === 0 ? " base" : ""}" x1="${margin.left}" x2="${width - margin.right}" y1="${y(tick)}" y2="${y(tick)}"/>
      <text class="tick" x="${margin.left - 8}" y="${y(tick) + 4}" text-anchor="end">${tick}</text>`);
  }

  // X ticks sit on bin edges; greedily keep edges >= 56px apart so labels never collide.
  // The final edge always stays, so any earlier tick that crowds it is dropped instead.
  const lastIndex = edges.length - 1;
  const endPosition = x(edges[lastIndex]);
  const tickIndexes = [];
  let lastTick = -Infinity;
  edges.forEach((edge, index) => {
    const position = x(edge);
    if (index === lastIndex || position - lastTick < 56 || endPosition - position < 56) return;
    tickIndexes.push(index);
    lastTick = position;
  });
  tickIndexes.push(lastIndex);
  const xTicks = tickIndexes.map((index) => {
    const edge = edges[index];
    const position = x(edge);
    const anchor = index === 0 ? "start" : index === lastIndex ? "end" : "middle";
    return `<line class="xtick" x1="${position}" x2="${position}" y1="${baseline}" y2="${baseline + 4}"/>
      <text class="tick" x="${position}" y="${baseline + 18}" text-anchor="${anchor}">${escapeHtml(label(edge))}</text>`;
  });

  const bars = bins.map((bin, index) => {
    if (!bin.count) return "";
    const left = x(bin.lower) + 1;
    const barWidth = Math.max(x(bin.upper) - x(bin.lower) - 2, 1);
    const top = y(bin.count);
    const radius = Math.min(4, barWidth / 2, baseline - top);
    return `<path class="bar${index === selectedIndex ? " is-selected" : ""}${overCap(bin) ? " is-over" : ""}" data-index="${index}" style="fill:${series.color}" d="M${left},${baseline} V${top + radius} Q${left},${top} ${left + radius},${top} H${left + barWidth - radius} Q${left + barWidth},${top} ${left + barWidth},${top + radius} V${baseline} Z"/>`;
  }).join("");

  const hits = bins.map((bin, index) => `<rect class="hit${bin.count ? " is-filterable" : ""}" tabindex="0" data-index="${index}" x="${x(bin.lower)}" y="${margin.top}" width="${Math.max(x(bin.upper) - x(bin.lower), 1)}" height="${plotHeight}" role="button" aria-pressed="${index === selectedIndex}" aria-label="${escapeHtml(`${label(bin.lower)} to ${label(bin.upper)}: ${bin.count} cases${bin.count ? (index === selectedIndex ? ". Press to clear the register filter" : ". Press to filter the register") : ""}`)}"/>`).join("");

  const markers = [
    { name: "Median", value: stats.median, className: "ref-median" },
    { name: "Mean", value: stats.mean, className: "ref-mean" },
  ].filter((marker) => !isLog || marker.value > 0);
  // A surface-colored halo under each line keeps it legible over bars of any class color.
  const markerSvg = markers.map((marker) => {
    const coords = `x1="${x(marker.value)}" x2="${x(marker.value)}" y1="${margin.top}" y2="${baseline}"`;
    return `<line class="ref-halo" ${coords}/><line class="ref ${marker.className}" ${coords}/>`;
  }).join("");

  // Cap marker: a tinted covered (under-cap) zone plus a line with a wide grab area, drawn above the hit layer.
  const plotLeft = margin.left;
  const plotRight = width - margin.right;
  const clampX = (px) => Math.min(Math.max(px, plotLeft), plotRight);
  const invert = isLog
    ? (px) => 10 ** (Math.log10(lower) + ((clampX(px) - plotLeft) / plotWidth) * (Math.log10(upper) - Math.log10(lower)))
    : (px) => lower + ((clampX(px) - plotLeft) / plotWidth) * (upper - lower);
  const capX = cap == null ? null : clampX(x(cap));
  const capLabel = cap == null ? "" : exact(cap);
  const capZone = cap == null ? "" : `<rect class="cap-zone" x="${plotLeft}" y="${margin.top}" width="${capX - plotLeft}" height="${plotHeight}"/>`;
  const capSvg = cap == null ? "" : `<g class="cap-marker" transform="translate(${capX},0)">
      <line class="ref-halo" y1="${margin.top}" y2="${baseline}"/>
      <line class="cap-line" y1="${margin.top}" y2="${baseline}"/>
      <circle class="cap-knob" cy="${margin.top}" r="5"/>
      <rect class="cap-grip" x="-9" y="${margin.top - 8}" width="18" height="${plotHeight + 8}" tabindex="0" role="slider" aria-label="Cap per case" aria-valuetext="${escapeHtml(capLabel)}" aria-valuemin="${lower}" aria-valuemax="${upper}" aria-valuenow="${cap}"/>
    </g>`;

  const infoId = `scale-info-${section}-${kind}`;
  const info = isLog
    ? `<span class="info-wrap"><button type="button" class="info-button" aria-label="About the normalized view" aria-describedby="${infoId}" aria-expanded="false">i</button><span class="info-pop" role="tooltip" id="${infoId}">${escapeHtml(scaleInfo(series, stats))}</span></span>`
    : "";

  canvas.innerHTML = `<div class="ref-key">
      <span class="axis-note">Cases per ${isLog ? "normalized" : "raw"} bin${info}</span>
      ${markers.map((marker) => `<span class="${marker.className}"><i></i>${marker.name} <strong>${escapeHtml(exact(marker.value))}</strong></span>`).join("")}
      ${cap == null ? "" : `<span class="ref-cap"><i></i>Cap <strong>${escapeHtml(capLabel)}</strong></span>`}
    </div>
    <svg class="histogram" width="${width}" height="${CHART_HEIGHT}" viewBox="0 0 ${width} ${CHART_HEIGHT}" role="img" aria-label="${escapeHtml(`${series.measure}: ${stats.count} cases, median ${exact(stats.median)}, mean ${exact(stats.mean)}`)}">
      <g>${gridLines.join("")}</g>
      <g>${capZone}</g>
      <g class="bars${selectedIndex >= 0 ? " has-selection" : ""}">${bars}</g>
      <g>${xTicks.join("")}</g>
      <g>${markerSvg}</g>
      <g class="hits">${hits}</g>
      ${capSvg}
    </svg>
    <div class="chart-tooltip" hidden><span class="tt-range"></span><strong class="tt-count"></strong><span class="tt-note"></span><span class="tt-hint"></span></div>`;

  const tooltip = $(".chart-tooltip", canvas);
  const barsGroup = $(".bars", canvas);
  const svg = $("svg", canvas);
  const show = (index) => {
    const bin = bins[index];
    $(".tt-range", tooltip).textContent = `${label(bin.lower)} – ${label(bin.upper)} per case`;
    $(".tt-count", tooltip).textContent = `${bin.count} case${bin.count === 1 ? "" : "s"}`;
    $(".tt-note", tooltip).textContent = `${((bin.count / plotted) * 100).toFixed(1)}% of plotted · ${exact(bin.sum)} combined`;
    $(".tt-hint", tooltip).textContent = !bin.count ? "" : index === selectedIndex ? "Click to clear register filter" : "Click to filter register";
    tooltip.hidden = false;
    barsGroup.classList.add("has-active");
    $$(".bar", canvas).forEach((bar) => bar.classList.toggle("is-active", Number(bar.dataset.index) === index));
    const center = (x(bin.lower) + x(bin.upper)) / 2;
    const tipWidth = tooltip.offsetWidth;
    tooltip.style.left = `${Math.min(Math.max(center - tipWidth / 2, 0), width - tipWidth)}px`;
    tooltip.style.top = `${Math.max(svg.offsetTop + y(bin.count) - tooltip.offsetHeight - 10, 0)}px`;
  };
  const hide = () => {
    tooltip.hidden = true;
    barsGroup.classList.remove("has-active");
    $$(".bar.is-active", canvas).forEach((bar) => bar.classList.remove("is-active"));
  };
  $$(".hit", canvas).forEach((hit) => {
    const index = Number(hit.dataset.index);
    hit.addEventListener("pointerenter", () => show(index));
    hit.addEventListener("focus", () => show(index));
    hit.addEventListener("pointerleave", hide);
    hit.addEventListener("blur", hide);
    hit.addEventListener("click", () => toggle(index));
    hit.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      toggle(index);
    });
  });

  const grip = $(".cap-grip", canvas);
  if (!grip) return;
  // Dragging moves the marker and readout live; the full redraw happens on release.
  const marker = $(".cap-marker", canvas);
  const zone = $(".cap-zone", canvas);
  const setCap = (value) => {
    state.dist[capKey(kind)] = Number(value.toPrecision(3));
    const position = clampX(x(state.dist[capKey(kind)]));
    marker.setAttribute("transform", `translate(${position},0)`);
    zone.setAttribute("width", position - plotLeft);
    $$(".bar", canvas).forEach((bar) => bar.classList.toggle("is-over", bins[Number(bar.dataset.index)].lower >= state.dist[capKey(kind)]));
    $(".ref-cap strong", canvas).textContent = exact(state.dist[capKey(kind)]);
    updateCapViews(kind, { redraw: false });
  };
  let dragging = false;
  grip.addEventListener("pointerdown", (event) => {
    event.preventDefault();
    dragging = true;
    try { grip.setPointerCapture(event.pointerId); } catch { /* capture is best-effort */ }
    canvas.classList.add("is-dragging");
    hide();
  });
  grip.addEventListener("pointermove", (event) => {
    if (!dragging) return;
    setCap(invert(event.clientX - svg.getBoundingClientRect().left));
  });
  const release = (event) => {
    if (!dragging) return;
    dragging = false;
    if (grip.hasPointerCapture(event.pointerId)) grip.releasePointerCapture(event.pointerId);
    canvas.classList.remove("is-dragging");
    updateCapViews(kind);
    $(".cap-grip", canvas)?.focus();
  };
  grip.addEventListener("pointerup", release);
  grip.addEventListener("pointercancel", release);
  grip.addEventListener("keydown", (event) => {
    if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
    event.preventDefault();
    const step = (event.shiftKey ? 20 : 4) * (event.key === "ArrowRight" ? 1 : -1);
    setCap(invert(clampX(x(state.dist[capKey(kind)])) + step));
    updateCapViews(kind);
    $(".cap-grip", canvas)?.focus();
  });
}

function scaleInfo(series, stats) {
  const example = series.isMoney ? "$0.10–$0.20 and $1–$2" : "10K–20K and 100K–200K";
  const zeros = stats.zeros
    ? ` ${fullNumber(stats.zeros)} case${stats.zeros === 1 ? "" : "s"} at zero ${stats.zeros === 1 ? "is" : "are"} left out, since a log axis has no zero.`
    : "";
  return `Logarithmic x-axis: each bin covers the same ratio rather than the same width (${example} are equally wide), so small cases and the long tail fit on one axis. Bar heights are still raw case counts.${zeros} Switch to Raw for equal-width bins.`;
}

// Identifies the chart configuration a bin selection was made under.
const distSignature = (kind) => (kind === "spend"
  ? `${state.dist.spendClass}|${state.dist.spendTier}|${state.dist.spendScale}`
  : `${state.dist.tokenClass}|${state.dist.tokenTier}|${state.dist.tokensScale}`);

// One bin selection per section; clicking the selected bin again clears it.
function toggleBinFilter(section, series, bins, index, label) {
  const bin = bins[index];
  const current = state.binFilter[section];
  const signature = distSignature(series.kind);
  if (current && current.kind === series.kind && current.signature === signature
    && current.lower === bin.lower && current.upper === bin.upper) {
    state.binFilter[section] = null;
  } else {
    const keys = new Set(series.rows
      .filter((row, rowIndex) => binIndexOf(series.values[rowIndex], bins, series.scale) === index)
      .map((row) => row.key));
    state.binFilter[section] = {
      kind: series.kind, signature, lower: bin.lower, upper: bin.upper, keys,
      label: `${series.cls.short === "Total" || series.cls.short === "All" ? (series.isMoney ? "Spend" : "Tokens") : series.cls.label} ${label(bin.lower)} – ${label(bin.upper)}`,
    };
  }
  redrawSectionCharts(section);
  updateRegister(section, { full: true });
}

function redrawSectionCharts(section) {
  $$(`#section-${section} .chart-canvas`).forEach((canvas) => drawHistogram(canvas));
}

function clearBinFilter(section, kind) {
  if (!state.binFilter[section] || (kind && state.binFilter[section].kind !== kind)) return;
  state.binFilter[section] = null;
  updateRegister(section, { full: true });
}

// Charts draw at their real pixel width (crisp text), and redraw when the width changes,
// which also covers a hidden tab becoming visible.
const chartObserver = new ResizeObserver((entries) => {
  for (const entry of entries) {
    const canvas = entry.target;
    if (Math.floor(canvas.clientWidth) && canvas.dataset.drawnWidth !== String(Math.floor(canvas.clientWidth))) {
      drawHistogram(canvas);
    }
  }
});

function mountCharts(root = document) {
  $$(".chart-canvas", root).forEach((canvas) => {
    chartObserver.observe(canvas);
    drawHistogram(canvas);
  });
}

function refreshDistributionCards(kind, focusSelector) {
  for (const section of ["extractor", "inference"]) {
    const card = $(`[data-card="${section}-${kind}"]`);
    if (!card) continue;
    $$(".chart-canvas", card).forEach((canvas) => chartObserver.unobserve(canvas));
    const template = document.createElement("template");
    template.innerHTML = distributionCard(section, kind).trim();
    const replacement = template.content.firstElementChild;
    card.replaceWith(replacement);
    mountCharts(replacement);
  }
  if (focusSelector) {
    const activeSection = $(".pipeline-section.active");
    $(focusSelector, activeSection || document)?.focus();
  }
}

// A bin selection only means something for the chart settings it was made under.
function changeDistControl(name, value, focusSelector) {
  const kind = name.startsWith("spend") ? "spend" : "tokens";
  state.dist[name] = value;
  if (!name.endsWith("Table")) ["extractor", "inference"].forEach((section) => clearBinFilter(section, kind));
  refreshDistributionCards(kind, focusSelector);
}

$("#sections").addEventListener("click", (event) => {
  const info = event.target.closest(".info-button");
  if (info) {
    const open = info.getAttribute("aria-expanded") !== "true";
    $$(".info-button[aria-expanded='true']").forEach((button) => button.setAttribute("aria-expanded", "false"));
    info.setAttribute("aria-expanded", String(open));
    return;
  }
  const details = event.target.closest(".details-button");
  if (details) {
    openCase(JSON.parse(details.dataset.detail));
    return;
  }
  const capClear = event.target.closest("[data-cap-clear]");
  if (capClear) {
    const kind = capClear.dataset.capClear;
    state.dist[capKey(kind)] = null;
    refreshDistributionCards(kind, `[data-card$="-${kind}"] [data-cap-input]`);
    return;
  }
  const sortButton = event.target.closest("[data-register-sort]");
  if (sortButton) {
    cycleSort(sortButton.dataset.section, sortButton.dataset.registerSort);
    return;
  }
  const clear = event.target.closest("[data-register-clear]");
  if (clear) {
    const { section, registerClear } = clear.dataset;
    if (registerClear === "bin" || registerClear === "all") state.binFilter[section] = null;
    if (registerClear === "columns" || registerClear === "all") state.register[section] = {};
    redrawSectionCharts(section);
    updateRegister(section, { full: true });
    return;
  }
  const control = event.target.closest("button[data-dist-control]");
  if (!control) return;
  const name = control.dataset.distControl;
  if (name.endsWith("Table")) {
    changeDistControl(name, !state.dist[name], `[data-card$="-${name.replace("Table", "")}"] [data-dist-control="${name}"]`);
    return;
  }
  const kind = name.startsWith("spend") ? "spend" : "tokens";
  changeDistControl(name, control.dataset.value, `[data-card$="-${kind}"] [data-dist-control="${name}"][data-value="${CSS.escape(control.dataset.value)}"]`);
});
$("#sections").addEventListener("change", (event) => {
  const filter = event.target.closest("select[data-register-filter]");
  if (filter) {
    setRegisterFilter(filter.dataset.section, filter.dataset.registerFilter, filter.value);
    return;
  }
  const control = event.target.closest("select[data-dist-control]");
  if (!control) return;
  const name = control.dataset.distControl;
  const kind = name.startsWith("spend") ? "spend" : "tokens";
  changeDistControl(name, control.value, `[data-card$="-${kind}"] select[data-dist-control="${name}"]`);
});
$("#sections").addEventListener("input", (event) => {
  const filter = event.target.closest("input[data-register-filter]");
  if (filter) {
    setRegisterFilter(filter.dataset.section, filter.dataset.registerFilter, filter.value);
    return;
  }
  const capInput = event.target.closest("input[data-cap-input]");
  if (capInput) {
    const kind = capInput.dataset.capInput;
    const value = parseAmount(capInput.value);
    const empty = !capInput.value.trim();
    const invalid = !empty && !(value >= 0);
    capInput.toggleAttribute("aria-invalid", invalid);
    if (invalid) return;
    state.dist[capKey(kind)] = empty ? null : value;
    const target = $("[data-cap-target]", capInput.closest(".cap-panel"));
    if (target) target.value = "";
    updateCapViews(kind);
    return;
  }
  // A coverage target sets the cap to the smallest observed value that reaches it in this section.
  const targetInput = event.target.closest("input[data-cap-target]");
  if (targetInput) {
    const kind = targetInput.dataset.capTarget;
    const target = Number(targetInput.value.replace("%", "").trim());
    const empty = !targetInput.value.trim();
    const invalid = !empty && !(target > 0 && target <= 100);
    targetInput.toggleAttribute("aria-invalid", invalid);
    if (empty || invalid) return;
    const section = targetInput.closest("[data-card]").dataset.card.split("-")[0];
    state.dist[capKey(kind)] = capForCoverage(withScale(distributionSeries(section, kind)).values, target / 100);
    updateCapViews(kind);
  }
});
document.addEventListener("click", (event) => {
  if (event.target.closest(".info-wrap")) return;
  $$(".info-button[aria-expanded='true']").forEach((button) => button.setAttribute("aria-expanded", "false"));
});
$("#sections").addEventListener("keydown", (event) => {
  // Arrow keys move between options inside a segmented control (radio-group behavior).
  const control = event.target.closest(".segmented button");
  if (!control || !["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const buttons = $$("button", control.parentElement);
  const next = buttons[(buttons.indexOf(control) + (event.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length];
  event.preventDefault();
  next.click();
});

function cycleBadges(cycles) {
  if (!cycles?.length) return '<span class="muted-zero">No outputs</span>';
  return `<div class="cycle-badges">${cycles.map((cycle) => `<span>R${cycle}</span>`).join("")}</div>`;
}

function caseRows(section, rows) {
  if (!rows.length) return `<tr><td colspan="${REGISTER_COLUMNS.length}"><div class="empty-table">No cases match the current filters.</div></td></tr>`;
  return rows.map((row) => {
    const hasOutput = row.metrics.artifacts > 0;
    return `<tr>
      <td><span class="case-dot">${escapeHtml(row.exception_id)}</span></td>
      <td class="asset-cell">${escapeHtml(row.asset)}</td>
      <td>${escapeHtml(row.year)}</td>
      <td><span class="category-chip" title="${escapeHtml(row.category)}">${escapeHtml(row.category)}</span></td>
      <td>${cycleBadges(row.cycles)}</td>
      <td class="${hasOutput ? "" : "muted-zero"}">${compact(row.metrics.input_tokens)}</td>
      <td class="${hasOutput ? "" : "muted-zero"}">${compact(row.metrics.cache_read_tokens)}</td>
      <td class="${hasOutput ? "" : "muted-zero"}">${compact(row.metrics.output_tokens)}</td>
      <td class="spend-cell ${hasOutput ? "" : "muted-zero"}">${money(row.metrics.total_cost_usd)}</td>
      <td class="savings-cell ${hasOutput ? "" : "muted-zero"}">${money(row.metrics.cache_savings_usd)}</td>
      <td>${hasOutput ? `<button class="details-button" type="button" data-detail='${escapeHtml(JSON.stringify({section, run_key: row.run_key, year: row.year, asset: row.asset, exception_id: row.exception_id}))}'>View rebuttals →</button>` : '<span class="muted-zero">Pending</span>'}</td>
    </tr>`;
  }).join("");
}

function renderSection(section) {
  const config = state.data.sections[section];
  const rows = visibleRows(section);
  const metrics = sumMetrics(rows);
  const selectedRun = config.runs.find((run) => run.run_key === state.selectedRuns[section]) || config.runs[0];
  const location = config.source.location || "Not configured";
  const selectedFolder = selectedRun?.folder_path || location;
  const selectedFolderUrl = runFolderUrl(selectedRun);
  const description = section === "extractor"
    ? "Evidence extraction and synthesis spend, including every discovered rebuttal cycle."
    : "Labor exception orchestration spend, including node-driven inference and rebuttal reruns.";
  const warning = config.error_count
    ? `<div class="section-warning">${config.error_count} output item${config.error_count === 1 ? "" : "s"} could not be indexed. ${escapeHtml(config.errors[0] || "")}</div>`
    : "";

  $(`#section-${section}`).innerHTML = `
    <div class="section-heading">
      <div><p class="eyebrow">${section === "extractor" ? "Source evidence" : "Decision intelligence"}</p><h2>${escapeHtml(config.label)}</h2><p>${description}</p></div>
      <div class="run-control"><label for="run-${section}">Cached run</label><select id="run-${section}" class="run-select" data-run-section="${section}" title="${escapeHtml(selectedFolder)}">${config.runs.map((run, index) => `<option value="${escapeHtml(run.run_key)}" title="${escapeHtml(run.folder_path || location)}" ${run.run_key === state.selectedRuns[section] ? "selected" : ""}>${index === 0 ? "Latest — " : ""}${escapeHtml(run.run_id)} · ${escapeHtml(runDate(run.updated_at))}</option>`).join("")}</select>${selectedFolderUrl ? `<a class="source-pill source-link" href="${escapeHtml(selectedFolderUrl)}" target="_blank" rel="noreferrer" title="${escapeHtml(selectedFolder)}"><strong>${escapeHtml(selectedRun?.target || config.source.target)}</strong> · ${escapeHtml(selectedFolder)}</a>` : `<div class="source-pill" title="${escapeHtml(selectedFolder)}"><strong>${escapeHtml(selectedRun?.target || config.source.target)}</strong> · ${escapeHtml(selectedFolder)}</div>`}</div>
    </div>
    ${warning}
    <div class="section-spend-grid">
      <article class="spend-feature"><small>Filtered section spend</small><strong>${money(metrics.total_cost_usd)}</strong><p>${money(metrics.cache_savings_usd)} saved through prompt caching across ${fullNumber(metrics.artifacts || 0)} rebuttal outputs.</p></article>
      <article class="mix-card"><h3>Token composition</h3><p>Prompt billing classes and generated output</p>${tokenMix(metrics)}</article>
    </div>
    <div class="distribution-grid">${distributionCard(section, "spend")}${distributionCard(section, "tokens")}</div>
    <div class="metric-ribbon">
      <div class="ribbon-item"><span>New input</span><strong>${compact(metrics.input_tokens)}</strong></div>
      <div class="ribbon-item"><span>Cache read</span><strong>${compact(metrics.cache_read_tokens)}</strong></div>
      <div class="ribbon-item"><span>Cache write</span><strong>${compact(metrics.cache_write_tokens)}</strong></div>
      <div class="ribbon-item"><span>Output</span><strong>${compact(metrics.output_tokens)}</strong></div>
      <div class="ribbon-item"><span>LLM calls</span><strong>${fullNumber(metrics.calls || 0)}</strong></div>
      <div class="ribbon-item"><span>Uncached cost</span><strong>${money(metrics.uncached_cost_usd)}</strong></div>
    </div>
    ${registerPanel(section)}`;
}

const REGISTER_COLUMNS = [
  { key: "exception_id", label: "Exception", filter: "text", placeholder: "e.g. 0003" },
  { key: "asset", label: "Asset", filter: "select" },
  { key: "year", label: "Year", filter: "select" },
  { key: "category", label: "Category", filter: "select" },
  { key: "cycles", label: "Rebuttals", filter: "select" },
  { key: "input_tokens", label: "Input", filter: "range", sortable: true },
  { key: "cache_read_tokens", label: "Cache read", filter: "range", sortable: true },
  { key: "output_tokens", label: "Output", filter: "range", sortable: true },
  { key: "total_cost_usd", label: "Spend", filter: "range", isMoney: true, sortable: true },
  { key: "cache_savings_usd", label: "Saved", filter: "range", isMoney: true, sortable: true },
  { key: "actions", label: "" },
];
const RANGE_HELP = "Filter by value: >1k, <=0.5, 10k-50k. A bare number means at least that much.";

// "10k-50k", ">1.5", "<=$0.2", "2m" (at least 2M). Returns null when empty, { invalid } when unparseable.
function parseRange(text) {
  const cleaned = String(text || "").toLowerCase().replace(/[\s,$]/g, "");
  if (!cleaned) return null;
  const number = parseAmount;
  const dash = cleaned.indexOf("-", 1);
  if (dash > 0) {
    const [min, max] = [number(cleaned.slice(0, dash)), number(cleaned.slice(dash + 1))];
    return Number.isNaN(min) || Number.isNaN(max) ? { invalid: true } : { min: Math.min(min, max), max: Math.max(min, max) };
  }
  const [, op = ">=", rest] = cleaned.match(/^(>=|<=|>|<|=)?(.*)$/);
  const value = number(rest);
  if (Number.isNaN(value)) return { invalid: true };
  return {
    ">=": { min: value }, ">": { min: value, minExclusive: true },
    "<=": { max: value }, "<": { max: value, maxExclusive: true }, "=": { min: value, max: value },
  }[op];
}

function inRange(value, range) {
  if (!range || range.invalid) return true;
  if (range.min !== undefined && (range.minExclusive ? value <= range.min : value < range.min)) return false;
  if (range.max !== undefined && (range.maxExclusive ? value >= range.max : value > range.max)) return false;
  return true;
}

function registerRows(section) {
  const filters = state.register[section];
  const bin = state.binFilter[section];
  const text = (filters.exception_id || "").trim().toLowerCase();
  const rows = visibleRows(section).filter((row) => {
    if (bin && !bin.keys.has(row.key)) return false;
    if (text && !String(row.exception_id).toLowerCase().includes(text)) return false;
    if (filters.asset && row.asset !== filters.asset) return false;
    if (filters.year && String(row.year) !== filters.year) return false;
    if (filters.category && row.category !== filters.category) return false;
    if (filters.cycles === "none" && row.cycles?.length) return false;
    if (filters.cycles && filters.cycles !== "none" && !(row.cycles || []).map(String).includes(filters.cycles)) return false;
    return REGISTER_COLUMNS.filter((column) => column.filter === "range")
      .every((column) => inRange(Number(row.metrics[column.key] || 0), parseRange(filters[column.key])));
  });
  const sort = state.sort[section];
  if (!sort) return rows;
  const direction = sort.dir === "asc" ? 1 : -1;
  return rows.sort((a, b) => (Number(a.metrics[sort.key] || 0) - Number(b.metrics[sort.key] || 0)) * direction);
}

// Header click cycles one column: highest first → lowest first → original order.
function cycleSort(section, key) {
  const current = state.sort[section];
  state.sort[section] = current?.key !== key ? { key, dir: "desc" } : current.dir === "desc" ? { key, dir: "asc" } : null;
  updateRegister(section, { full: true });
  $(`#register-${section} [data-register-sort="${key}"]`)?.focus();
}

function registerHeader(section, column) {
  if (!column.sortable) return `<th>${escapeHtml(column.label)}</th>`;
  const sort = state.sort[section];
  const dir = sort?.key === column.key ? sort.dir : null;
  const next = !dir ? "highest first" : dir === "desc" ? "lowest first" : "original order";
  return `<th aria-sort="${dir === "asc" ? "ascending" : dir === "desc" ? "descending" : "none"}"><button type="button" class="sort-button${dir ? " is-sorted" : ""}" data-register-sort="${column.key}" data-section="${section}" title="Sort ${next}">${escapeHtml(column.label)}<span class="sort-icon" aria-hidden="true">${dir === "asc" ? "▲" : dir === "desc" ? "▼" : "↕"}</span></button></th>`;
}

function registerFilterCell(section, column) {
  const value = state.register[section][column.key] || "";
  const attrs = `data-register-filter="${column.key}" data-section="${section}" aria-label="Filter ${escapeHtml(column.label)}"`;
  if (column.filter === "text") {
    return `<input type="search" class="column-filter" ${attrs} value="${escapeHtml(value)}" placeholder="${column.placeholder}" autocomplete="off"/>`;
  }
  if (column.filter === "range") {
    const invalid = parseRange(value)?.invalid;
    return `<input type="text" class="column-filter column-range" ${attrs} value="${escapeHtml(value)}" placeholder="${column.isMoney ? "e.g. >0.5" : "e.g. >1k"}" title="${RANGE_HELP}" inputmode="decimal" autocomplete="off" ${invalid ? 'aria-invalid="true"' : ""}/>`;
  }
  if (column.filter === "select") {
    const rows = state.data.sections[section]?.cases || [];
    let options;
    if (column.key === "cycles") {
      const cycles = [...new Set(rows.flatMap((row) => row.cycles || []))].sort((a, b) => a - b);
      options = [["none", "No outputs"], ...cycles.map((cycle) => [String(cycle), `Has R${cycle}`])];
    } else {
      const values = [...new Set(rows.map((row) => String(row[column.key] ?? "")).filter(Boolean))];
      values.sort((a, b) => (column.key === "year" ? b.localeCompare(a) : a.localeCompare(b)));
      options = values.map((item) => [item, item]);
    }
    return `<select class="column-filter" ${attrs}><option value="">All</option>${options.map(([optionValue, text]) => `<option value="${escapeHtml(optionValue)}" ${optionValue === value ? "selected" : ""}>${escapeHtml(text)}</option>`).join("")}</select>`;
  }
  return "";
}

function registerSummary(section, rows) {
  const total = visibleRows(section).length;
  const outputs = rows.filter((row) => row.metrics.artifacts > 0).length;
  return `${outputs} with outputs · ${rows.length === total ? `${rows.length} shown` : `${rows.length} of ${total} shown`}`;
}

function registerChips(section) {
  const bin = state.binFilter[section];
  const columnCount = Object.values(state.register[section]).filter((value) => String(value || "").trim()).length;
  const chips = [];
  if (bin) {
    chips.push(`<button type="button" class="filter-chip is-bin" data-register-clear="bin" data-section="${section}" aria-label="Clear chart filter ${escapeHtml(bin.label)}"><span>From chart</span>${escapeHtml(bin.label)}<b aria-hidden="true">×</b></button>`);
  }
  if (columnCount) {
    chips.push(`<button type="button" class="filter-chip" data-register-clear="columns" data-section="${section}">${columnCount} column filter${columnCount === 1 ? "" : "s"}<b aria-hidden="true">×</b></button>`);
  }
  if (chips.length > 1) chips.push(`<button type="button" class="filter-clear" data-register-clear="all" data-section="${section}">Clear all</button>`);
  return chips.join("");
}

function registerPanel(section) {
  const rows = registerRows(section);
  return `<article class="case-panel" id="register-${section}">
      <header class="case-panel-header">
        <div><h3>Exception spend register</h3><p>Token details come directly from the local run-scoped cache. Click a histogram bar to filter.</p></div>
        <div class="register-status"><div class="register-chips">${registerChips(section)}</div><span class="case-count">${registerSummary(section, rows)}</span></div>
      </header>
      <div class="table-wrap"><table class="case-table">
        <thead>
          <tr>${REGISTER_COLUMNS.map((column) => registerHeader(section, column)).join("")}</tr>
          <tr class="filter-row">${REGISTER_COLUMNS.map((column) => `<th>${registerFilterCell(section, column)}</th>`).join("")}</tr>
        </thead>
        <tbody>${caseRows(section, rows)}</tbody>
      </table></div>
    </article>`;
}

// Typing only refreshes the body so the filter input keeps focus; a full refresh also resets inputs.
function updateRegister(section, { full = false } = {}) {
  const panel = $(`#register-${section}`);
  if (!panel) return;
  if (full) {
    const template = document.createElement("template");
    template.innerHTML = registerPanel(section).trim();
    panel.replaceWith(template.content.firstElementChild);
    return;
  }
  const rows = registerRows(section);
  $("tbody", panel).innerHTML = caseRows(section, rows);
  $(".case-count", panel).textContent = registerSummary(section, rows);
  $(".register-chips", panel).innerHTML = registerChips(section);
  $$("input.column-range", panel).forEach((input) => {
    if (parseRange(input.value)?.invalid) input.setAttribute("aria-invalid", "true");
    else input.removeAttribute("aria-invalid");
  });
}

function setRegisterFilter(section, key, value) {
  state.register[section][key] = value;
  updateRegister(section);
}

function renderAll() {
  renderGlobalMetrics();
  renderSection("extractor");
  renderSection("inference");
  mountCharts();
  const uniqueKeys = new Set(["extractor", "inference"].flatMap((section) => visibleRows(section).map((row) => row.key)));
  $("#visibleCases").textContent = `${uniqueKeys.size} case${uniqueKeys.size === 1 ? "" : "s"} in view`;
  $$(".run-select").forEach((select) => select.addEventListener("change", () => {
    selectRun(select.dataset.runSection, select.value);
  }));
}

async function selectRun(section, runKey) {
  const query = new URLSearchParams({ section, run_key: runKey });
  try {
    const response = await fetch(`/api/run?${query}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Cached run could not be loaded");
    state.selectedRuns[section] = runKey;
    state.binFilter[section] = null;
    state.data.sections[section].cases = payload.cases;
    renderAll();
  } catch (error) {
    $("#errorState").hidden = false;
    $("#errorMessage").textContent = error.message;
  }
}

function tierUsageRows(usage, cost) {
  const tiers = [...new Set([...Object.keys(usage || {}), ...Object.keys(cost || {})])];
  if (!tiers.length) return '<tr><td colspan="8" class="muted-zero">No tier details recorded</td></tr>';
  return tiers.map((name) => {
    const u = usage[name] || {};
    const c = cost[name] || {};
    return `<tr><td><strong>${escapeHtml(name)}</strong><br><span class="muted-zero">${escapeHtml(u.model || c.model || "")}</span></td><td>${fullNumber(u.calls)}</td><td>${compact(u.tokens_in)}</td><td>${compact(u.cache_read_tokens)}</td><td>${compact(u.cache_creation_tokens)}</td><td>${compact(u.tokens_out)}</td><td>${compact(u.thinking_tokens)}</td><td>${money(c.total)}</td></tr>`;
  }).join("");
}

function tierCostRows(cost) {
  const entries = Object.entries(cost || {});
  if (!entries.length) return '<tr><td colspan="7" class="muted-zero">No priced tier details recorded</td></tr>';
  return entries.map(([name, c]) => `<tr><td><strong>${escapeHtml(name)}</strong></td><td>${money(c.input)}</td><td>${money(c.cache_read)}</td><td>${money(c.cache_write)}</td><td>${money(c.output)}</td><td>${money(c.total)}</td><td class="savings-cell">${money(c.cache_savings)}</td></tr>`).join("");
}

function resultExcerpt(result) {
  if (!result) return "";
  const main = result.synthesis || result.draft_response || result.disposition || "";
  const label = result.synthesis ? "Extractor synthesis" : result.draft_response ? "Draft response" : "Disposition";
  if (!main) return "";
  return `<div class="result-excerpt"><strong>${label}:</strong> ${escapeHtml(main.slice(0, 1000))}${main.length > 1000 ? "…" : ""}</div>`;
}

function artifactCard(artifact) {
  const m = artifact.metrics;
  const price = artifact.cost_metadata;
  return `<article class="cycle-card">
    <header class="cycle-header">
      <div class="cycle-title"><span class="cycle-number">R${artifact.cycle}</span><span><strong>Rebuttal ${artifact.cycle}</strong><small>${escapeHtml(artifact.file_name)} · ${escapeHtml(artifact.estimator || "unknown estimator")}</small></span></div>
      <a class="artifact-link" href="${escapeHtml(artifact.source_url)}" target="_blank" rel="noreferrer">Open source JSON ↗</a>
    </header>
    <div class="cycle-kpis">
      <div class="cycle-kpi"><span>Spend</span><strong>${money(m.total_cost_usd)}</strong></div>
      <div class="cycle-kpi"><span>Cache saved</span><strong>${money(m.cache_savings_usd)}</strong></div>
      <div class="cycle-kpi"><span>Total tokens</span><strong>${compact(m.total_tokens)}</strong></div>
      <div class="cycle-kpi"><span>Prompt</span><strong>${compact(m.prompt_tokens)}</strong></div>
      <div class="cycle-kpi"><span>Output</span><strong>${compact(m.output_tokens)}</strong></div>
      <div class="cycle-kpi"><span>Elapsed</span><strong>${artifact.elapsed_minutes == null ? "—" : `${artifact.elapsed_minutes}m`}</strong></div>
    </div>
    <div class="detail-grid">
      <div class="detail-block"><h4>Usage by model tier</h4><div class="table-wrap"><table class="detail-table"><thead><tr><th>Tier / model</th><th>Calls</th><th>Input</th><th>Read</th><th>Write</th><th>Output</th><th>Think</th><th>Cost</th></tr></thead><tbody>${tierUsageRows(artifact.usage_by_tier, artifact.cost_by_tier)}</tbody></table></div></div>
      <div class="detail-block"><h4>Spend by billing class</h4><div class="table-wrap"><table class="detail-table"><thead><tr><th>Tier</th><th>Input</th><th>Read</th><th>Write</th><th>Output</th><th>Total</th><th>Saved</th></tr></thead><tbody>${tierCostRows(artifact.cost_by_tier)}</tbody></table></div></div>
    </div>
    ${resultExcerpt(artifact.result)}
    <div class="pricing-note">${price.priced ? `Priced from ${escapeHtml(price.rates_source || "configured model rates")}${price.as_of ? ` · as of ${escapeHtml(price.as_of)}` : ""}${price.cache_ttl ? ` · ${escapeHtml(price.cache_ttl)} cache TTL` : ""}` : "This artifact was not priced."}</div>
  </article>`;
}

async function openCase(params) {
  const dialog = $("#caseDialog");
  $("#dialogEyebrow").textContent = params.section === "extractor" ? "Excel Extractor" : "Labor Inference";
  $("#dialogTitle").textContent = `${params.asset} · Exception ${params.exception_id}`;
  $("#dialogSubtitle").textContent = `${params.year} · Loading cached rebuttal metrics…`;
  $("#dialogBody").innerHTML = '<div class="dialog-loading"><div><div class="loader"></div><p>Reading the local token metric index</p></div></div>';
  if (!dialog.open) dialog.showModal();
  const query = new URLSearchParams(params);
  try {
    const response = await fetch(`/api/case?${query}`);
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Case details could not be loaded");
    const item = payload.case;
    $("#dialogTitle").textContent = `${item.asset} · Exception ${item.exception_id}`;
    $("#dialogSubtitle").textContent = `${item.year} · ${item.category}`;
    $("#dialogBody").innerHTML = `
      <div class="case-meta-grid">
        <div class="meta-box"><span>Pipeline</span><strong>${escapeHtml(payload.label)}</strong></div>
        <div class="meta-box"><span>Run ID</span><strong>${escapeHtml(item.run_id)}</strong></div>
        <div class="meta-box"><span>Rebuttal files</span><strong>${payload.artifacts.length}</strong></div>
        <div class="meta-box"><span>Total spend</span><strong>${money(item.metrics.total_cost_usd)}</strong></div>
        <div class="meta-box"><span>Output folder</span><strong title="${escapeHtml(item.folder_path)}">${escapeHtml(item.folder_path || "—")}</strong></div>
      </div>
      ${payload.artifacts.length ? payload.artifacts.map(artifactCard).join("") : '<div class="empty-table">No output JSONs were found for this case.</div>'}`;
  } catch (error) {
    $("#dialogBody").innerHTML = `<div class="dialog-error"><strong>Could not load this case.</strong><p>${escapeHtml(error.message)}</p></div>`;
  }
}

async function loadDashboard(refresh = false) {
  $("#loadingState").hidden = false;
  $("#sections").hidden = true;
  $("#errorState").hidden = true;
  $("#refreshButton").disabled = true;
  try {
    const response = await fetch(refresh ? "/api/refresh" : "/api/summary", { method: refresh ? "POST" : "GET" });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Output index could not be loaded");
    state.data = payload;
    for (const section of ["extractor", "inference"]) {
      const config = payload.sections[section];
      if (!config.runs.some((run) => run.run_key === state.selectedRuns[section])) {
        state.selectedRuns[section] = config.default_run_key;
      }
    }
    const categories = $("#categoryFilter");
    categories.innerHTML = '<option value="all">All categories</option>' + payload.categories
      .map((category) => `<option value="${escapeHtml(category)}">${escapeHtml(category)}</option>`).join("");
    categories.value = state.category;
    const targets = new Set(Object.values(payload.sections).map((section) => section.source.target));
    $("#targetBadge").textContent = `${payload.record_count} cached metrics`;
    $("#lastUpdated").textContent = `Indexed ${new Date(payload.generated_at).toLocaleString()}`;
    renderAll();
    $("#loadingState").hidden = true;
    $("#sections").hidden = false;
  } catch (error) {
    $("#loadingState").hidden = true;
    $("#errorState").hidden = false;
    $("#errorMessage").textContent = error.message;
  } finally {
    $("#refreshButton").disabled = false;
  }
}

$("#categoryFilter").addEventListener("change", (event) => {
  state.category = event.target.value;
  renderAll();
});
$("#caseSearch").addEventListener("input", (event) => {
  state.search = event.target.value;
  renderAll();
});
$("#refreshButton").addEventListener("click", () => loadDashboard(true));
$("#errorRetry").addEventListener("click", () => loadDashboard(true));
$("#closeDialog").addEventListener("click", () => $("#caseDialog").close());
$("#caseDialog").addEventListener("click", (event) => {
  if (event.target === event.currentTarget) event.currentTarget.close();
});
$$('.section-tab').forEach((tab) => tab.addEventListener("click", () => {
  state.activeSection = tab.dataset.section;
  $$('.section-tab').forEach((item) => item.classList.toggle("active", item === tab));
  $$('.pipeline-section').forEach((section) => section.classList.toggle("active", section.dataset.section === state.activeSection));
}));

// Theme: "light", "dark", or "system" (follows the OS). The head script already stamped the
// initial theme; this keeps it in sync with the toggle and with OS changes while on "system".
const systemDark = matchMedia("(prefers-color-scheme: dark)");
const THEME_BAR_COLOR = { light: "#ffd500", dark: "#1a1917" };

function applyTheme(choice) {
  const theme = choice === "dark" || (choice === "system" && systemDark.matches) ? "dark" : "light";
  document.documentElement.dataset.theme = theme;
  document.documentElement.dataset.themePref = choice;
  $('meta[name="theme-color"]')?.setAttribute("content", THEME_BAR_COLOR[theme]);
  $$(".theme-toggle [data-theme-choice]").forEach((button) => {
    const active = button.dataset.themeChoice === choice;
    button.setAttribute("aria-checked", String(active));
    button.classList.toggle("is-active", active);
  });
}

$(".theme-toggle").addEventListener("click", (event) => {
  const button = event.target.closest("[data-theme-choice]");
  if (!button) return;
  try { localStorage.setItem("dashboard-theme", button.dataset.themeChoice); } catch { /* storage blocked */ }
  applyTheme(button.dataset.themeChoice);
});
$(".theme-toggle").addEventListener("keydown", (event) => {
  // Arrow keys move between options (radio-group behavior).
  if (!["ArrowLeft", "ArrowRight"].includes(event.key)) return;
  const buttons = $$("[data-theme-choice]", event.currentTarget);
  const current = buttons.indexOf(event.target.closest("[data-theme-choice]"));
  const next = buttons[(current + (event.key === "ArrowRight" ? 1 : buttons.length - 1)) % buttons.length];
  event.preventDefault();
  next.focus();
  next.click();
});
systemDark.addEventListener("change", () => {
  if (document.documentElement.dataset.themePref === "system") applyTheme("system");
});
applyTheme(document.documentElement.dataset.themePref || "system");

loadDashboard();
