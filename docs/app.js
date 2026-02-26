const DATA_URL = "data/benchmark_summary.json";

const chartPalette = {
  amber: "#ffa630",
  teal: "#27c4a8",
  coral: "#ff6f61",
  sun: "#ffd166",
  blue: "#6fb6ff",
  mint: "#84dcc6",
  slate: "#5e8096",
};

if (window.Chart) {
  Chart.defaults.color = "#d4e3ed";
  Chart.defaults.borderColor = "rgba(255, 255, 255, 0.14)";
  Chart.defaults.font.family = "Sora, sans-serif";
}

function pct(value) {
  return `${Number(value).toFixed(1)}%`;
}

function dateTime(value) {
  return new Date(value).toLocaleString();
}

function metricCard(title, value, detail) {
  return `
    <article class="metric-card">
      <p class="metric-title">${title}</p>
      <p class="metric-value">${value}</p>
      <p class="metric-detail">${detail}</p>
    </article>
  `;
}

function setHeader(data) {
  document.getElementById("generated-at").textContent = `Generated ${dateTime(
    data.generated_at_utc
  )}`;
  document.getElementById(
    "totals-badge"
  ).textContent = `${data.totals.rows} rows • ${data.totals.repeats} repeats • ${data.totals.questions} questions`;
}

function renderCards(data) {
  const overall = data.overall;
  const rescue = data.rescue;
  const consistency = data.consistency;

  const cardsMarkup = [
    metricCard(
      "Direct",
      pct(overall.direct.pct),
      `${overall.direct.correct}/${overall.direct.total}`
    ),
    metricCard(
      "MCQ with refusal",
      pct(overall.mcq_with_refusal.pct),
      `${overall.mcq_with_refusal.correct}/${overall.mcq_with_refusal.total}`
    ),
    metricCard(
      "MCQ without refusal",
      pct(overall.mcq_without_refusal.pct),
      `${overall.mcq_without_refusal.correct}/${overall.mcq_without_refusal.total}`
    ),
    metricCard(
      "MCQ rescue rate",
      pct(rescue.rescued_pct),
      `${rescue.rescued}/${rescue.direct_wrong} direct misses rescued`
    ),
    metricCard(
      "Always-correct questions",
      pct(consistency.always_correct_pct),
      `${consistency.always_correct}/${data.totals.questions} questions`
    ),
  ].join("");

  document.getElementById("headline-cards").innerHTML = cardsMarkup;
}

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
    `
    )
    .join("");

  const links = document.getElementById("csv-links");
  links.innerHTML = data.source_files
    .map(
      (item) =>
        `<li><a href="data/results/${item.filename}" download>${item.filename}</a></li>`
    )
    .join("");
}

function renderRepeatTable(data) {
  const rows = data.by_repeat.slice(0, 5);
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
    `
    )
    .join("");
}

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
          backgroundColor: [chartPalette.coral, chartPalette.sun, chartPalette.teal],
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
      labels: data.by_repeat.map((row) => `${row.file_code}-${row.repeat_label}`),
      datasets: [
        {
          label: "MCQ without refusal",
          data: data.by_repeat.map((row) => row.metrics.mcq_without_refusal.pct),
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
          backgroundColor: [chartPalette.mint, chartPalette.amber, chartPalette.coral],
          borderWidth: 0,
        },
      ],
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
    },
  });
}

async function init() {
  const response = await fetch(DATA_URL);
  if (!response.ok) {
    throw new Error(`Failed to load ${DATA_URL}`);
  }
  const data = await response.json();
  setHeader(data);
  renderCards(data);
  renderSourceTable(data);
  renderRepeatTable(data);
  renderCharts(data);
}

init().catch((error) => {
  const header = document.getElementById("generated-at");
  header.textContent = "Failed to load benchmark data.";
  console.error(error);
});
