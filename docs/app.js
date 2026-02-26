const DATA_URL = "data/benchmark_summary.json";

const chartPalette = {
  amber: "#d4a74a",
  teal: "#7a9a68",
  coral: "#d97777",
  sun: "#98b384",
  blue: "#7aa2cc",
  mint: "#b8d5a7",
  slate: "#94a3b8",
};

if (window.Chart) {
  Chart.defaults.color = "#52525b";
  Chart.defaults.borderColor = "#e4e4e7";
  Chart.defaults.font.family = "Inter, sans-serif";
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.boxHeight = 10;
}

function pct(value) {
  return `${Number(value).toFixed(1)}%`;
}

function pp(value) {
  const v = Number(value);
  return `${v >= 0 ? "+" : ""}${v.toFixed(1)}pp`;
}

function dateTime(value) {
  return new Date(value).toLocaleString();
}

function metricCard(title, value, detail, tooltip) {
  const tip = tooltip
    ? ` data-tooltip="${escapeHtml(tooltip)}" tabindex="0"`
    : "";
  return `
    <article class="metric-card"${tip}>
      <p class="metric-title">${title}</p>
      <p class="metric-value">${value}</p>
      <p class="metric-detail">${detail}</p>
    </article>
  `;
}

function cellClass(value) {
  if (value === 0) return "cell-zero";
  if (value === 100) return "cell-perfect";
  return "";
}

function rowClass(pctValue) {
  if (pctValue === 100) return "row-perfect";
  if (pctValue < 50) return "row-struggling";
  return "";
}

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str;
  return div.innerHTML;
}

// --- Header ---

function setHeader(data) {
  document.getElementById("generated-at").textContent = `Generated ${dateTime(
    data.generated_at_utc,
  )}`;
  document.getElementById("totals-badge").textContent =
    `${data.totals.rows} rows \u2022 ${data.totals.repeats} repeats \u2022 ${data.totals.questions} questions`;
}

// --- Grading Modes ---

function renderGradingModes(data) {
  const modes = data.commentary.grading_modes;
  const invariant = data.commentary.invariant_note;
  const container = document.getElementById("grading-modes-content");

  const cards = modes
    .map(
      (m) => `
    <div class="mode-card accent-${m.accent}">
      <h3>${escapeHtml(m.name)}</h3>
      <p>${escapeHtml(m.description)}</p>
    </div>
  `,
    )
    .join("");

  container.innerHTML = `
    <div class="grading-modes-grid">${cards}</div>
    <p class="invariant-note">${escapeHtml(invariant)}</p>
  `;
}

// --- Metric Cards (9) ---

function renderCards(data) {
  const overall = data.overall;
  const rescue = data.rescue;
  const consistency = data.consistency;
  const h = data.headline;

  const cardsMarkup = [
    metricCard(
      "Direct",
      pct(overall.direct.pct),
      `${overall.direct.correct}/${overall.direct.total}`,
      "Accuracy when the model answers the question directly (open-ended, no answer choices provided).",
    ),
    metricCard(
      "MCQ with refusal",
      pct(overall.mcq_with_refusal.pct),
      `${overall.mcq_with_refusal.correct}/${overall.mcq_with_refusal.total}`,
      "Accuracy on multiple-choice questions where 'I don't know' is included as an answer option.",
    ),
    metricCard(
      "MCQ without refusal",
      pct(overall.mcq_without_refusal.pct),
      `${overall.mcq_without_refusal.correct}/${overall.mcq_without_refusal.total}`,
      "Accuracy on multiple-choice questions without an 'I don't know' option, forcing a best guess.",
    ),
    metricCard(
      "MCQ Lift",
      pp(h.mcq_lift_pp),
      `Direct \u2192 MCQ w/o refusal`,
      "Percentage-point gain when switching from direct (open-ended) to MCQ without refusal. Shows how much answer choices help the model.",
    ),
    metricCard(
      "Refusal Gap",
      `${h.refusal_gap_pp}pp`,
      `MCQ w/o \u2192 MCQ w/ refusal`,
      "Percentage-point drop from MCQ without refusal to MCQ with refusal. Measures how often the model opts for 'I don't know' when given the chance.",
    ),
    metricCard(
      "MCQ rescue rate",
      pct(rescue.rescued_pct),
      `${rescue.rescued}/${rescue.direct_wrong} direct misses rescued`,
      "Of questions answered wrong in direct mode, the percentage that were answered correctly in MCQ without refusal mode.",
    ),
    metricCard(
      "Best repeat",
      pct(h.best_repeat_pct),
      h.best_repeat_label,
      "The highest MCQ without refusal accuracy achieved by any single repeat run.",
    ),
    metricCard(
      "Always-correct questions",
      pct(consistency.always_correct_pct),
      `${consistency.always_correct}/${data.totals.questions} questions`,
      "Questions answered correctly in MCQ without refusal mode across every single repeat run.",
    ),
    metricCard(
      "Task groups at 100%",
      pct(h.task_groups_at_100_pct),
      `${h.task_groups_at_100}/${h.total_task_groups} task groups`,
      "Task groups where every question was answered correctly in MCQ without refusal across all runs.",
    ),
  ].join("");

  document.getElementById("headline-cards").innerHTML = cardsMarkup;
}

function initMetricTooltips() {
  const cards = Array.from(
    document.querySelectorAll(".metric-card[data-tooltip]"),
  );
  if (!cards.length) return;

  const tip = document.createElement("div");
  tip.id = "metric-tooltip";
  tip.className = "tip";
  tip.setAttribute("role", "tooltip");
  document.body.appendChild(tip);

  const coarsePointerQuery = window.matchMedia("(pointer: coarse)");
  const viewportPadding = 8;
  const edgePadding = 12;
  const gap = 12;
  let activeCard = null;

  function clamp(value, min, max) {
    return Math.min(Math.max(value, min), max);
  }

  function positionTooltip(card) {
    const rect = card.getBoundingClientRect();
    const anchorX = rect.left + rect.width / 2;

    tip.style.left = "0px";
    tip.style.top = "0px";
    const tipRect = tip.getBoundingClientRect();

    const maxLeft = window.innerWidth - tipRect.width - viewportPadding;
    const left = clamp(
      anchorX - tipRect.width / 2,
      viewportPadding,
      Math.max(viewportPadding, maxLeft),
    );
    const anchorInside = clamp(
      anchorX - left,
      edgePadding,
      tipRect.width - edgePadding,
    );
    tip.style.left = `${left}px`;
    tip.style.setProperty("--tip-anchor", `${anchorInside}px`);

    const spaceAbove = rect.top - viewportPadding;
    const spaceBelow = window.innerHeight - rect.bottom - viewportPadding;
    const placeAbove =
      spaceAbove >= tipRect.height + gap || spaceAbove >= spaceBelow;

    if (placeAbove) {
      tip.dataset.placement = "top";
      tip.style.top = `${Math.max(
        viewportPadding,
        rect.top - tipRect.height - gap,
      )}px`;
      return;
    }

    tip.dataset.placement = "bottom";
    tip.style.top = `${Math.min(
      window.innerHeight - tipRect.height - viewportPadding,
      rect.bottom + gap,
    )}px`;
  }

  function showTooltip(card) {
    if (activeCard === card) {
      positionTooltip(card);
      return;
    }
    if (activeCard) {
      activeCard.classList.remove("is-tooltip-active");
    }
    activeCard = card;
    activeCard.classList.add("is-tooltip-active");
    tip.textContent = card.dataset.tooltip;
    tip.classList.add("visible");
    positionTooltip(card);
  }

  function hideTooltip() {
    if (activeCard) {
      activeCard.classList.remove("is-tooltip-active");
      activeCard = null;
    }
    tip.classList.remove("visible");
    tip.removeAttribute("data-placement");
  }

  cards.forEach((card) => {
    card.setAttribute("aria-describedby", "metric-tooltip");

    card.addEventListener("mouseenter", () => {
      showTooltip(card);
    });

    card.addEventListener("mousemove", () => {
      if (activeCard === card) {
        positionTooltip(card);
      }
    });

    card.addEventListener("mouseleave", () => {
      if (activeCard === card) {
        hideTooltip();
      }
    });

    card.addEventListener("focus", () => {
      showTooltip(card);
    });

    card.addEventListener("blur", () => {
      if (activeCard === card) {
        hideTooltip();
      }
    });

    card.addEventListener("click", (event) => {
      if (!coarsePointerQuery.matches) return;
      event.preventDefault();
      if (activeCard === card) {
        hideTooltip();
        return;
      }
      showTooltip(card);
    });
  });

  document.addEventListener("click", (event) => {
    if (!coarsePointerQuery.matches) return;
    if (
      event.target instanceof Element &&
      event.target.closest(".metric-card[data-tooltip]")
    ) {
      return;
    }
    hideTooltip();
  });

  document.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      hideTooltip();
    }
  });

  window.addEventListener(
    "scroll",
    () => {
      if (activeCard) {
        positionTooltip(activeCard);
      }
    },
    true,
  );

  window.addEventListener("resize", () => {
    if (activeCard) {
      positionTooltip(activeCard);
    }
  });
}

// --- Key Takeaways ---

function renderKeyTakeaways(data) {
  const list = document.getElementById("takeaways-list");
  list.innerHTML = data.commentary.key_takeaways
    .map((t) => `<li>${escapeHtml(t)}</li>`)
    .join("");
}

// --- Source Table ---

function renderSourceTable(data) {
  const tableBody = document.getElementById("source-table-body");
  tableBody.innerHTML = data.source_files
    .map(
      (item) => `
      <tr>
        <td><code>${item.filename}</code></td>
        <td>${item.date}</td>
        <td>${item.repeats}</td>
        <td>${item.rows}</td>
      </tr>
    `,
    )
    .join("");

  const links = document.getElementById("csv-links");
  links.innerHTML = data.source_files
    .map(
      (item) =>
        `<li><a href="data/results/${item.filename}" download>${item.filename}</a></li>`,
    )
    .join("");
}

// --- Strengths & Weaknesses ---

function renderStrengthsWeaknesses(data) {
  document.getElementById("strengths-list").innerHTML =
    data.commentary.strengths.map((s) => `<li>${escapeHtml(s)}</li>`).join("");
  document.getElementById("weaknesses-list").innerHTML =
    data.commentary.weaknesses.map((w) => `<li>${escapeHtml(w)}</li>`).join("");
}

// --- Sortable Table Helper ---

function makeSortable(tableId, dataArray, renderRowFn) {
  const table = document.getElementById(tableId);
  if (!table) return;
  const headers = table.querySelectorAll("th[data-sort]");
  let currentSort = null;
  let ascending = true;

  headers.forEach((th) => {
    th.addEventListener("click", () => {
      const key = th.dataset.sort;
      if (currentSort === key) {
        ascending = !ascending;
      } else {
        currentSort = key;
        ascending = true;
      }
      const sorted = [...dataArray].sort((a, b) => {
        const va = a[key];
        const vb = b[key];
        if (typeof va === "number" && typeof vb === "number") {
          return ascending ? va - vb : vb - va;
        }
        return ascending
          ? String(va).localeCompare(String(vb))
          : String(vb).localeCompare(String(va));
      });
      const tbody = table.querySelector("tbody");
      tbody.innerHTML = sorted.map(renderRowFn).join("");
    });
  });
}

// --- All 50 Questions Table ---

function renderQuestionTable(data) {
  const questions = data.question_scores;
  const tbody = document.getElementById("question-table-body");

  function renderRow(q) {
    const rc = rowClass(q.mcq_without_refusal_pct);
    return `<tr class="${rc}">
      <td><code>${q.question_id}</code></td>
      <td>${q.task_group}</td>
      <td class="${cellClass(q.direct_pct)}">${pct(q.direct_pct)}</td>
      <td class="${cellClass(q.mcq_without_refusal_pct)}">${pct(q.mcq_without_refusal_pct)}</td>
      <td class="${cellClass(q.mcq_with_refusal_pct)}">${pct(q.mcq_with_refusal_pct)}</td>
      <td>${q.refusal_gap_pp}pp</td>
      <td>${pp(q.mcq_lift_pp)}</td>
    </tr>`;
  }

  tbody.innerHTML = questions.map(renderRow).join("");
  makeSortable("question-table", questions, renderRow);
}

// --- All 32 Task Groups Table ---

function renderTaskGroupTable(data) {
  const groups = data.task_group_scores;
  const tbody = document.getElementById("taskgroup-table-body");

  function renderRow(g) {
    const rc = rowClass(g.mcq_without_refusal_pct);
    return `<tr class="${rc}">
      <td>${g.task_group}</td>
      <td>${g.questions}</td>
      <td>${g.n}</td>
      <td class="${cellClass(g.direct_pct)}">${pct(g.direct_pct)}</td>
      <td class="${cellClass(g.mcq_without_refusal_pct)}">${pct(g.mcq_without_refusal_pct)}</td>
      <td class="${cellClass(g.mcq_with_refusal_pct)}">${pct(g.mcq_with_refusal_pct)}</td>
    </tr>`;
  }

  tbody.innerHTML = groups.map(renderRow).join("");
  makeSortable("taskgroup-table", groups, renderRow);
}

// --- MCQ Rescue Detail ---

function renderRescueDetail(data) {
  const rescue = data.rescue;
  const lifts = data.rescue_lifts;
  const commentary = data.commentary.rescue_interpretation;
  const container = document.getElementById("rescue-detail-content");

  const statCards = `
    <div class="rescue-grid">
      <div class="rescue-stat">
        <p>Rescued</p>
        <div class="lift-value">${rescue.rescued}</div>
        <p>${pct(rescue.rescued_pct)} of ${rescue.direct_wrong} direct misses</p>
      </div>
      <div class="rescue-stat">
        <p>Lost</p>
        <div class="lift-value" style="color: var(--accent-coral)">${rescue.lost}</div>
        <p>${pct(rescue.lost_pct)} of ${rescue.direct_right} direct correct</p>
      </div>
      <div class="rescue-stat">
        <p>Perfect rescues</p>
        <div class="lift-value">${data.perfect_rescues}</div>
        <p>0% direct \u2192 100% MCQ</p>
      </div>
    </div>
  `;

  const liftsTable = lifts.length
    ? `<h3>Biggest MCQ Lifts (Direct \u2192 MCQ w/o Refusal)</h3>
    <div class="table-scroll"><table>
      <thead><tr><th>Question</th><th>Direct</th><th>MCQ w/o</th><th>Lift</th></tr></thead>
      <tbody>${lifts
        .map(
          (q) =>
            `<tr><td><code>${q.question_id}</code></td><td>${pct(q.direct_pct)}</td><td>${pct(q.mcq_without_refusal_pct)}</td><td>${pp(q.mcq_lift_pp)}</td></tr>`,
        )
        .join("")}</tbody>
    </table></div>`
    : "";

  container.innerHTML = `
    <p class="panel-intro">${escapeHtml(commentary)}</p>
    ${statCards}
    ${liftsTable}
  `;
}

// --- Cross-Run Variability ---

function renderCrossRunVariability(data) {
  const rows = data.cross_run_variability;
  const fileCodes = data.file_codes;
  if (!rows || !rows.length) return;

  const thead = document.getElementById("variability-table-head");
  thead.innerHTML = `<tr><th>Question</th>${fileCodes.map((fc) => `<th>${fc}</th>`).join("")}</tr>`;

  const tbody = document.getElementById("variability-table-body");
  tbody.innerHTML = rows
    .map((r) => {
      const cells = fileCodes
        .map((fc) => {
          const d = r.by_file[fc];
          const cls =
            d.correct === 0
              ? "cell-zero"
              : d.correct === d.total
                ? "cell-perfect"
                : "";
          return `<td class="${cls}">${d.correct}/${d.total}</td>`;
        })
        .join("");
      return `<tr><td><code>${r.question_id}</code></td>${cells}</tr>`;
    })
    .join("");
}

// --- Scientific Reporting ---

function renderScientificReporting(data) {
  const overall = data.overall;
  const mv = data.majority_vote;
  const container = document.getElementById("reporting-content");

  const metrics = ["direct", "mcq_with_refusal", "mcq_without_refusal"];
  const labels = {
    direct: "Direct",
    mcq_with_refusal: "MCQ w/ refusal",
    mcq_without_refusal: "MCQ w/o refusal",
  };

  const primaryItems = metrics
    .map((key) => {
      const m = overall[key];
      return `<li>${labels[key]}: <strong>${pct(m.pct)}</strong> (95% CI: ${m.ci_95.lo}\u2013${m.ci_95.hi}) &mdash; ${m.correct}/${m.total}</li>`;
    })
    .join("");

  const secondaryItems = metrics
    .map((key) => {
      const m = mv[key];
      return `<li>${labels[key]}: <strong>${m.correct}/${m.total}</strong> (${pct(m.pct)})</li>`;
    })
    .join("");

  container.innerHTML = `
    <div>
      <h3>Primary: Per-Instance + Wilson CI</h3>
      <ul class="reporting-list">${primaryItems}</ul>
    </div>
    <div>
      <h3>Secondary: Majority Vote (11-run ensemble)</h3>
      <ul class="reporting-list">${secondaryItems}</ul>
    </div>
  `;
}

// --- Chart Annotations ---

function addChartAnnotations(data) {
  const overallAnnotation = document.getElementById("overall-chart-annotation");
  if (overallAnnotation) {
    overallAnnotation.innerHTML = `<p class="chart-annotation">${escapeHtml(data.commentary.overall_interpretation)}</p>`;
  }
  const gapAnnotation = document.getElementById("gap-chart-annotation");
  if (gapAnnotation) {
    gapAnnotation.innerHTML = `<p class="chart-annotation">${escapeHtml(data.commentary.refusal_gap_interpretation)}</p>`;
  }
}

// --- Best Repeats Table (all 11) ---

function renderRepeatTable(data) {
  const rows = data.by_repeat;
  const tableBody = document.getElementById("repeat-table-body");
  tableBody.innerHTML = rows
    .map(
      (row) => `
      <tr>
        <td>${row.rank_mcq_without_refusal}</td>
        <td><code>${row.file_code}-${row.repeat_label}</code></td>
        <td>${pct(row.metrics.direct.pct)}</td>
        <td>${pct(row.metrics.mcq_with_refusal.pct)}</td>
        <td>${pct(row.metrics.mcq_without_refusal.pct)}</td>
      </tr>
    `,
    )
    .join("");
}

// --- Charts ---

function newChart(id, config) {
  const element = document.getElementById(id);
  return new Chart(element, config);
}

function renderCharts(data) {
  newChart("overall-chart", {
    type: "bar",
    data: {
      labels: ["Direct", "MCQ with refusal", "MCQ without refusal"],
      datasets: [
        {
          label: "Accuracy",
          data: [
            data.overall.direct.pct,
            data.overall.mcq_with_refusal.pct,
            data.overall.mcq_without_refusal.pct,
          ],
          backgroundColor: [
            chartPalette.coral,
            chartPalette.sun,
            chartPalette.teal,
          ],
          borderRadius: 8,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { callback: (value) => `${value}%` },
        },
      },
    },
  });

  newChart("file-chart", {
    type: "bar",
    data: {
      labels: data.by_file.map((row) => row.file_code),
      datasets: [
        {
          label: "Direct",
          data: data.by_file.map((row) => row.metrics.direct.pct),
          backgroundColor: chartPalette.coral,
        },
        {
          label: "MCQ with refusal",
          data: data.by_file.map((row) => row.metrics.mcq_with_refusal.pct),
          backgroundColor: chartPalette.sun,
        },
        {
          label: "MCQ without refusal",
          data: data.by_file.map((row) => row.metrics.mcq_without_refusal.pct),
          backgroundColor: chartPalette.teal,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 0,
          max: 100,
          ticks: { callback: (value) => `${value}%` },
        },
      },
    },
  });

  newChart("repeat-chart", {
    type: "line",
    data: {
      labels: data.by_repeat.map(
        (row) => `${row.file_code}-${row.repeat_label}`,
      ),
      datasets: [
        {
          label: "MCQ without refusal",
          data: data.by_repeat.map(
            (row) => row.metrics.mcq_without_refusal.pct,
          ),
          borderColor: chartPalette.teal,
          pointBackgroundColor: chartPalette.teal,
          pointRadius: 4,
          tension: 0.2,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: {
          min: 80,
          max: 100,
          ticks: { callback: (value) => `${value}%` },
        },
      },
    },
  });

  const hardest = data.hardest_questions.slice(0, 10);
  newChart("hardest-chart", {
    type: "bar",
    data: {
      labels: hardest.map((row) => row.question_id),
      datasets: [
        {
          label: "Direct",
          data: hardest.map((row) => row.direct_pct),
          backgroundColor: chartPalette.slate,
        },
        {
          label: "MCQ without refusal",
          data: hardest.map((row) => row.mcq_without_refusal_pct),
          backgroundColor: chartPalette.blue,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: 0,
          max: 100,
          ticks: { callback: (value) => `${value}%` },
        },
      },
    },
  });

  const refusal = data.refusal_gap_questions.slice(0, 10);
  newChart("gap-chart", {
    type: "bar",
    data: {
      labels: refusal.map((row) => row.question_id),
      datasets: [
        {
          label: "Refusal gap (pp)",
          data: refusal.map((row) => row.refusal_gap_pp),
          backgroundColor: chartPalette.amber,
          borderRadius: 6,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
      },
      scales: {
        x: {
          min: 0,
          ticks: { callback: (value) => `${value}pp` },
        },
      },
    },
  });

  newChart("consistency-chart", {
    type: "doughnut",
    data: {
      labels: ["Always correct", "Mixed", "Always wrong"],
      datasets: [
        {
          data: [
            data.consistency.always_correct,
            data.consistency.mixed,
            data.consistency.always_wrong,
          ],
          backgroundColor: [
            chartPalette.mint,
            chartPalette.amber,
            chartPalette.coral,
          ],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });

  // Task group horizontal grouped bar chart
  const tgData = [...data.task_group_scores].sort(
    (a, b) => a.mcq_without_refusal_pct - b.mcq_without_refusal_pct,
  );
  newChart("taskgroup-chart", {
    type: "bar",
    data: {
      labels: tgData.map((g) => g.task_group),
      datasets: [
        {
          label: "Direct",
          data: tgData.map((g) => g.direct_pct),
          backgroundColor: chartPalette.coral,
        },
        {
          label: "MCQ without refusal",
          data: tgData.map((g) => g.mcq_without_refusal_pct),
          backgroundColor: chartPalette.teal,
        },
      ],
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          min: 0,
          max: 100,
          ticks: { callback: (value) => `${value}%` },
        },
      },
    },
  });
}

// --- Init ---

async function init() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Failed to load ${DATA_URL}`);
  }
  const data = await response.json();
  setHeader(data);
  renderGradingModes(data);
  renderCards(data);
  initMetricTooltips();
  renderKeyTakeaways(data);
  renderSourceTable(data);
  renderCharts(data);
  addChartAnnotations(data);
  renderStrengthsWeaknesses(data);
  renderQuestionTable(data);
  renderTaskGroupTable(data);
  renderRescueDetail(data);
  renderCrossRunVariability(data);
  renderScientificReporting(data);
  renderRepeatTable(data);
}

init().catch((error) => {
  const header = document.getElementById("generated-at");
  header.textContent = "Failed to load benchmark data.";
  console.error(error);
});
