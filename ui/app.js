const state = {
  data: null,
  activeTab: "overview"
};

const overviewPanel = document.getElementById("overview");
const baselinePanel = document.getElementById("baseline");
const cascadedPanel = document.getElementById("cascaded");
const adaptivePanel = document.getElementById("adaptive");
const statusPill = document.getElementById("status-pill");
const configSummary = document.getElementById("config-summary");
const fileInput = document.getElementById("file-input");
const loadDemoButton = document.getElementById("load-demo");
const serverResultSelect = document.getElementById("server-result-select");
const loadServerResultButton = document.getElementById("load-server-result");


function pct(value) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return `${(value * 100).toFixed(1)}%`;
}


function fixed(value, digits = 2) {
  if (value === null || value === undefined || Number.isNaN(value)) {
    return "n/a";
  }
  return Number(value).toFixed(digits);
}


function escapeHtml(text) {
  return String(text)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");
}


function tokenText(token) {
  if (!token || token.token_text === undefined || token.token_text === null) {
    return "";
  }
  if (token.token_text === " ") {
    return "[space]";
  }
  if (token.token_text === "\n") {
    return "[newline]";
  }
  return token.token_text;
}


function setActiveTab(tabName) {
  state.activeTab = tabName;
  document.querySelectorAll(".tab-button").forEach((button) => {
    button.classList.toggle("active", button.dataset.tab === tabName);
  });
  document.querySelectorAll(".tab-panel").forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabName);
  });
}


function metricCard(label, value) {
  return `
    <div class="metric">
      <div class="metric-label">${escapeHtml(label)}</div>
      <div class="metric-value">${escapeHtml(String(value))}</div>
    </div>
  `;
}


function usageTable(title, usageBucket) {
  if (!usageBucket) {
    return "";
  }
  const rows = Object.entries(usageBucket).map(([key, item]) => `
    <tr>
      <td>${escapeHtml(key)}</td>
      <td>${item.count ?? 0}</td>
      <td>
        ${pct(item.pct)}
        <div class="bar-track">
          <div class="bar-fill ${key}-color" style="width: ${Math.max(0, (item.pct || 0) * 100)}%"></div>
        </div>
      </td>
    </tr>
  `).join("");
  return `
    <article class="table-card">
      <h3>${escapeHtml(title)}</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Count</th>
            <th>Share</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </article>
  `;
}


function passRateTable(title, rates) {
  if (!rates) {
    return "";
  }
  const rows = Object.entries(rates).map(([edge, value]) => `
    <tr>
      <td>${escapeHtml(edge)}</td>
      <td>${pct(value)}</td>
    </tr>
  `).join("");
  return `
    <article class="table-card">
      <h3>${escapeHtml(title)}</h3>
      <table>
        <thead>
          <tr>
            <th>Edge</th>
            <th>Pass-through</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </article>
  `;
}


function runtimeTable(title, runtimeBucket) {
  if (!runtimeBucket) {
    return "";
  }
  const rows = Object.entries(runtimeBucket).map(([key, item]) => `
    <tr>
      <td>${escapeHtml(key)}</td>
      <td>${fixed(item.propose_calls, 1)}</td>
      <td>${fixed(item.review_calls, 1)}</td>
      <td>${fixed(item.propose_wall_time, 2)}s</td>
      <td>${fixed(item.review_wall_time, 2)}s</td>
      <td>${pct(item.share)}</td>
    </tr>
  `).join("");
  return `
    <article class="table-card">
      <h3>${escapeHtml(title)}</h3>
      <table>
        <thead>
          <tr>
            <th>Model</th>
            <th>Draft calls</th>
            <th>Verify calls</th>
            <th>Draft time</th>
            <th>Verify time</th>
            <th>Runtime share</th>
          </tr>
        </thead>
        <tbody>${rows}</tbody>
      </table>
    </article>
  `;
}


function modelStepWall(delta, key) {
  return delta?.[key]?.total_wall_time ?? null;
}


function resolveRun(data, key) {
  if (!data?.runs) {
    return null;
  }
  if (key === "cascaded") {
    return data.runs.cascaded || data.runs.hierarchical || null;
  }
  return data.runs[key] || null;
}


function runLabel(data, key) {
  if (key === "cascaded" && data?.runs?.hierarchical && !data?.runs?.cascaded) {
    return "Hierarchical";
  }
  if (key === "baseline") {
    return "Baseline";
  }
  if (key === "cascaded") {
    return "Cascaded";
  }
  if (key === "adaptive") {
    return "Adaptive";
  }
  return key;
}


function availableRuns(data) {
  return ["baseline", "cascaded", "adaptive"]
    .map((key) => ({ key, label: runLabel(data, key), run: resolveRun(data, key) }))
    .filter((entry) => entry.run);
}


function runSummarySection(entry) {
  const summary = entry.run?.summary;
  if (!summary) {
    return "";
  }
  return `
    <section class="section-card">
      <h2>${escapeHtml(entry.label)}</h2>
      <div class="metric-grid">
        ${metricCard("Tok/s", fixed(summary.tokens_per_sec, 2))}
        ${metricCard("Benchmark", fixed(summary.benchmark_score, 3))}
        ${metricCard("Avg wall time", `${fixed(summary.avg_wall_time, 2)}s`)}
        ${metricCard("Avg large reviews", fixed(summary.avg_large_review_positions, 1))}
        ${metricCard("Avg middle saves", fixed(summary.avg_mm_saved_positions, 1))}
        ${metricCard("Avg switches", fixed(summary.avg_switch_count, 2))}
        ${metricCard("Window policy", summary.draft_window_policy || "fixed")}
        ${metricCard("Avg small window", fixed(summary.avg_draft_window?.small, 2))}
        ${metricCard("Avg middle window", fixed(summary.avg_draft_window?.middle, 2))}
        ${metricCard("Window changes", fixed(summary.avg_draft_window_changes, 2))}
      </div>
    </section>
  `;
}


function comparisonSections(data) {
  const comparisons = data?.comparisons || {};
  const entries = Object.entries(comparisons);
  if (!entries.length && data?.comparison) {
    entries.push(["comparison", data.comparison]);
  }
  if (!entries.length) {
    return "";
  }
  return entries.map(([name, comparison]) => `
    <section class="section-card">
      <h2>${escapeHtml(name.replaceAll("_", " "))}</h2>
      <div class="metric-grid">
        ${metricCard("Throughput delta", fixed(comparison.throughput_delta, 2))}
        ${metricCard("Speedup", fixed(comparison.throughput_speedup, 3))}
        ${metricCard("Benchmark delta", fixed(comparison.benchmark_delta, 3))}
        ${metricCard("Wall time delta", `${fixed(comparison.avg_wall_time_delta, 2)}s`)}
        ${metricCard("Large call delta", fixed(comparison.avg_ml_forward_calls_delta, 2))}
        ${metricCard("Middle save delta", fixed(comparison.avg_mm_saved_positions_delta, 2))}
        ${metricCard("Candidate small window", fixed(comparison.avg_draft_window?.small, 2))}
        ${metricCard("Candidate middle window", fixed(comparison.avg_draft_window?.middle, 2))}
        ${metricCard("Window changes", fixed(comparison.avg_draft_window_changes, 2))}
      </div>
      <div class="table-grid">
        ${passRateTable("Candidate Pass Rates", comparison.candidate_pass_rates || comparison.hierarchical_pass_rates)}
        ${passRateTable("Baseline Pass Rates", comparison.baseline_pass_rates)}
      </div>
    </section>
  `).join("");
}


function renderOverview(data) {
  const runs = availableRuns(data);
  if (!runs.length) {
    overviewPanel.innerHTML = "";
    return;
  }

  const usageSections = runs.map((entry) => {
    const summary = entry.run.summary || {};
    const usage = summary.usage || {};
    return `
      <section class="section-card">
        <h2>${escapeHtml(entry.label)} Usage</h2>
        <div class="metric-grid">
          ${metricCard("Small drafter steps", fixed(summary.avg_drafter_steps?.small, 2))}
          ${metricCard("Middle drafter steps", fixed(summary.avg_drafter_steps?.middle, 2))}
          ${metricCard("Middle review positions", fixed(summary.avg_middle_review_positions, 1))}
          ${metricCard("Middle forward calls", fixed(summary.avg_mm_forward_calls, 2))}
          ${metricCard("Avg small window", fixed(summary.avg_draft_window?.small, 2))}
          ${metricCard("Avg middle window", fixed(summary.avg_draft_window?.middle, 2))}
          ${metricCard("Window changes", fixed(summary.avg_draft_window_changes, 2))}
        </div>
        <div class="table-grid">
          ${usageTable(`${entry.label} Final Token Share`, usage.final_source_counts)}
          ${usageTable(`${entry.label} Draft Token Share`, usage.draft_generated_counts)}
          ${usageTable(`${entry.label} Verification Positions`, usage.verification_positions)}
          ${passRateTable(`${entry.label} Pass-through`, usage.edge_pass_rates)}
          ${runtimeTable(`${entry.label} Runtime Breakdown`, summary.avg_model_runtime)}
        </div>
      </section>
    `;
  }).join("");

  overviewPanel.innerHTML = `
    <div class="overview-grid">
      <div class="summary-row">
        ${runs.map(runSummarySection).join("")}
      </div>
      ${comparisonSections(data)}
      ${usageSections}
    </div>
  `;
}


function renderTokenStream(tokens) {
  if (!tokens || !tokens.length) {
    return '<div class="muted">No tokens recorded.</div>';
  }
  return `
    <div class="token-stream">
      ${tokens.map((token) => `
        <div class="token-chip ${token.source_model} ${token.verified_by_middle ? "verified-middle" : ""} ${token.verified_by_large ? "verified-large" : ""}">
          ${escapeHtml(tokenText(token))}
          <span class="token-meta">
            ${escapeHtml(token.source_model)} | p${token.position}
            ${token.verified_by_middle ? " | M ok" : ""}
            ${token.verified_by_large ? " | L ok" : ""}
          </span>
        </div>
      `).join("")}
    </div>
  `;
}


function renderTokenPanel(title, tokens) {
  if (!tokens || !tokens.length) {
    return "";
  }
  return `
    <div class="token-panel">
      <h3>${escapeHtml(title)}</h3>
      ${renderTokenStream(tokens)}
    </div>
  `;
}


function renderStepTrace(steps) {
  if (!steps || !steps.length) {
    return '<div class="muted">No step trace recorded.</div>';
  }
  return `
    <div class="detail-grid">
      ${steps.map((step) => `
        <article class="detail-card">
          <h3>Step ${step.step_index}</h3>
          <div class="metric-grid">
            ${metricCard("Drafter", step.drafter || "n/a")}
            ${metricCard("Next drafter", step.next_drafter || "n/a")}
            ${metricCard("Requested", step.requested_tokens ?? "n/a")}
            ${metricCard("Middle accepted", step.middle_accepted_count ?? "n/a")}
            ${metricCard("Middle saved", step.middle_saved_positions ?? "n/a")}
            ${metricCard("Large accepted", step.large_accepted_count ?? "n/a")}
            ${metricCard("Large generated", step.large_generated_count ?? "n/a")}
            ${metricCard("Alpha before", fixed(step.alpha_before, 3))}
            ${metricCard("Alpha after", fixed(step.alpha_after, 3))}
            ${metricCard("Switched", step.switched ? "yes" : "no")}
            ${metricCard("Window policy", step.draft_window_policy || "fixed")}
            ${metricCard("Window accept", fixed(step.draft_window_acceptance_ratio, 3))}
            ${metricCard("Small wall", `${fixed(modelStepWall(step.model_runtime_delta, "small"), 3)}s`)}
            ${metricCard("Middle wall", `${fixed(modelStepWall(step.model_runtime_delta, "middle"), 3)}s`)}
            ${metricCard("Large wall", `${fixed(modelStepWall(step.model_runtime_delta, "large"), 3)}s`)}
          </div>
          ${renderTokenPanel("Small Draft", step.small_draft)}
          ${renderTokenPanel("Middle Draft", step.middle_draft)}
          ${renderTokenPanel("Middle Result", step.middle_result)}
          ${renderTokenPanel("Candidate To Large", step.candidate_to_large)}
          ${renderTokenPanel("Final Step", step.final_step)}
        </article>
      `).join("")}
    </div>
  `;
}


function renderSampleUsage(sample) {
  const usage = sample?.usage;
  if (!usage) {
    return "";
  }
  return `
    <div class="table-grid">
      ${usageTable("Final Token Share", usage.final_source_counts)}
      ${usageTable("Draft Token Share", usage.draft_generated_counts)}
      ${usageTable("Verification Positions", usage.verification_positions)}
      ${passRateTable("Pass-through", usage.edge_pass_rates)}
    </div>
  `;
}


function renderSampleExplorer(panel, data, key) {
  const runData = resolveRun(data, key);
  const label = runLabel(data, key);
  if (!runData) {
    panel.innerHTML = '<section class="section-card"><h2>Unavailable</h2><p class="muted">This run was not included in the loaded artifact.</p></section>';
    return;
  }

  const samples = runData.trace_samples || [];
  const summary = runData.summary || {};
  if (!samples.length) {
    panel.innerHTML = `
      <section class="section-card">
        <h2>${escapeHtml(label)}</h2>
        <div class="metric-grid">
          ${metricCard("Tok/s", fixed(summary.tokens_per_sec, 2))}
          ${metricCard("Benchmark", fixed(summary.benchmark_score, 3))}
          ${metricCard("Trace samples", 0)}
        </div>
        <p class="muted">Increase <code>--trace_samples</code> when generating the artifact.</p>
      </section>
    `;
    return;
  }

  panel.innerHTML = `
    <section class="section-card sample-explorer">
      <h2>${escapeHtml(label)}</h2>
      <div class="explorer-toolbar">
        <label>
          <span class="muted">Example sample</span><br>
          <select id="${key}-sample-select" class="select-control">
            ${samples.map((sample, index) => `
              <option value="${index}">
                sample ${sample.sample_index} | ${sample.score.correct ? "correct" : "incorrect"} | score ${fixed(sample.score.score, 2)}
              </option>
            `).join("")}
          </select>
        </label>
      </div>
      <div id="${key}-sample-detail"></div>
    </section>
  `;

  const select = document.getElementById(`${key}-sample-select`);
  const detail = document.getElementById(`${key}-sample-detail`);

  const paint = () => {
    const sample = samples[Number(select.value)];
    const draftWindow = sample.draft_window || {};
    detail.innerHTML = `
      <div class="detail-grid">
        <article class="detail-card">
          <h3>Sample Summary</h3>
          <div class="metric-grid">
            ${metricCard("Benchmark metric", sample.score.metric_name)}
            ${metricCard("Correct", sample.score.correct ? "yes" : "no")}
            ${metricCard("Prediction", sample.score.prediction ?? "n/a")}
            ${metricCard("Gold", sample.score.gold ?? "n/a")}
            ${metricCard("Large calls", sample.ml_forward_calls ?? "n/a")}
            ${metricCard("Middle saves", sample.mm_saved_positions ?? "n/a")}
            ${metricCard("Switch count", sample.switch_count ?? "n/a")}
          </div>
        </article>
        <article class="detail-card">
          <h3>Prompt</h3>
          <div class="text-block">${escapeHtml(sample.prompt)}</div>
        </article>
        <article class="detail-card">
          <h3>Generated Text</h3>
          <div class="text-block">${escapeHtml(sample.generated_text)}</div>
        </article>
        <article class="detail-card">
          <h3>Adaptive State</h3>
          <div class="metric-grid">
            ${metricCard("Alpha trace length", sample.alpha_trace?.length ?? 0)}
            ${metricCard("Acceptance samples", sample.acceptance_trace?.length ?? 0)}
            ${metricCard("Small drafter steps", sample.drafter_steps?.small ?? 0)}
            ${metricCard("Middle drafter steps", sample.drafter_steps?.middle ?? 0)}
            ${metricCard("Window policy", draftWindow.policy || "fixed")}
            ${metricCard("Avg small window", fixed(draftWindow.averages?.small, 2))}
            ${metricCard("Avg middle window", fixed(draftWindow.averages?.middle, 2))}
            ${metricCard("Window changes", draftWindow.change_count ?? 0)}
          </div>
          <p class="callout">Rolling alpha: ${escapeHtml((sample.alpha_trace || []).map((value) => fixed(value, 3)).join(", ") || "n/a")}</p>
        </article>
        <article class="token-panel">
          <h3>Final Token Trace</h3>
          ${renderTokenStream(sample.final_tokens)}
        </article>
        ${renderSampleUsage(sample)}
        ${renderStepTrace(sample.steps)}
      </div>
    `;
  };

  select.addEventListener("change", paint);
  paint();
}


function updateHeader(data, sourceLabel) {
  statusPill.textContent = `Loaded ${sourceLabel}`;
  configSummary.innerHTML = `
    dataset <code>${escapeHtml(data.config?.dataset || "n/a")}</code> |
    samples <code>${escapeHtml(data.config?.n_samples || "n/a")}</code> |
    k_s <code>${escapeHtml(data.config?.k_s || "n/a")}</code> |
    k_m <code>${escapeHtml(data.config?.k_m || "n/a")}</code> |
    tau <code>${escapeHtml(data.config?.tau || "n/a")}</code> |
    window <code>${escapeHtml(data.config?.window_size || "n/a")}</code> |
    draft window <code>${escapeHtml(data.config?.draft_window_policy || "fixed")}</code>
  `;
}


function loadData(data, sourceLabel) {
  state.data = data;
  updateHeader(data, sourceLabel);
  renderOverview(data);
  renderSampleExplorer(baselinePanel, data, "baseline");
  renderSampleExplorer(cascadedPanel, data, "cascaded");
  renderSampleExplorer(adaptivePanel, data, "adaptive");
}


async function fetchServerResults() {
  if (!serverResultSelect || !loadServerResultButton) {
    return [];
  }
  try {
    const response = await fetch("/api/results");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    const results = payload.results || [];
    serverResultSelect.innerHTML = results.map((item) => `
      <option value="${item.name}">${item.name}</option>
    `).join("");
    loadServerResultButton.disabled = results.length === 0;
    return results;
  } catch (error) {
    console.error(error);
    serverResultSelect.innerHTML = '<option value="">No server results found</option>';
    loadServerResultButton.disabled = true;
    return [];
  }
}


async function loadServerResultByName(name) {
  if (!name) {
    return;
  }
  try {
    const response = await fetch(`/api/results/${encodeURIComponent(name)}`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const payload = await response.json();
    loadData(payload, `server artifact ${name}`);
  } catch (error) {
    statusPill.textContent = "Failed to load server result";
    console.error(error);
  }
}


document.querySelectorAll(".tab-button").forEach((button) => {
  button.addEventListener("click", () => setActiveTab(button.dataset.tab));
});


fileInput.addEventListener("change", async (event) => {
  const [file] = event.target.files;
  if (!file) {
    return;
  }
  try {
    const text = await file.text();
    loadData(JSON.parse(text), file.name);
  } catch (error) {
    statusPill.textContent = "Failed to parse JSON";
    console.error(error);
  }
});


loadDemoButton.addEventListener("click", () => {
  loadData(window.DEMO_COMPARISON, "demo artifact");
});


if (loadServerResultButton) {
  loadServerResultButton.addEventListener("click", () => {
    loadServerResultByName(serverResultSelect?.value);
  });
}


loadData(window.DEMO_COMPARISON, "demo artifact");
fetchServerResults().then((results) => {
  if (results.length) {
    loadServerResultByName(results[0].name);
  }
});
