const form = document.querySelector("#prediction-form");
const statusNode = document.querySelector("#status");
const resultNode = document.querySelector("#result");
const metricsNode = document.querySelector("#training-metrics");
const navItems = document.querySelectorAll(".nav-item");
const panels = document.querySelectorAll(".tab-panel");

function activateTab(tabId) {
  navItems.forEach((item) => {
    item.classList.toggle("active", item.dataset.tab === tabId);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === tabId);
  });
}

navItems.forEach((item) => {
  item.addEventListener("click", () => activateTab(item.dataset.tab));
});

async function loadMetadata() {
  const response = await fetch("/metadata");
  const metadata = await response.json();
  const notebook = metadata.reported_notebook_metrics;
  const artifact = metadata.artifact_metadata || {};
  const currentMetrics = artifact.cross_validation_metrics_from_current_artifact || {};

  metricsNode.innerHTML = `
    <div><span>Selected dataset</span><strong>${artifact.dataset_used_for_artifact || notebook.dataset}</strong></div>
    <div><span>Selected model</span><strong>${notebook.model}</strong></div>
    <div><span>Artifact accuracy</span><strong>${formatMetric(currentMetrics.accuracy || notebook.accuracy)}</strong></div>
    <div><span>Disease recall</span><strong>${formatMetric(currentMetrics.recall_disease || notebook.recall_disease)}</strong></div>
  `;

  if (!metadata.artifact_available) {
    statusNode.textContent = "Model artifact is not available yet. The interface is ready, but prediction is disabled.";
  }
}

function formatMetric(value) {
  return Number(value).toFixed(2);
}

function formToPayload(formElement) {
  const data = new FormData(formElement);
  return {
    age: Number(data.get("age")),
    sex: data.get("sex"),
    dataset: data.get("dataset"),
    cp: data.get("cp"),
    trestbps: Number(data.get("trestbps")),
    chol: Number(data.get("chol")),
    fbs: data.get("fbs") === "true",
    restecg: data.get("restecg"),
    thalch: Number(data.get("thalch")),
    exang: data.get("exang") === "true",
    oldpeak: Number(data.get("oldpeak")),
    slope: data.get("slope"),
    ca: data.get("ca") === "" ? null : Number(data.get("ca")),
    thal: data.get("thal") || null,
  };
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  resultNode.classList.add("hidden");
  statusNode.classList.remove("error");
  statusNode.textContent = "Running prediction...";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(formToPayload(form)),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || "Prediction could not be computed.");
    }

    const result = await response.json();
    const probability =
      result.probability_disease === null
        ? "No disponible"
        : `${Math.round(result.probability_disease * 100)}%`;

    statusNode.textContent = "Prediction completed.";
    resultNode.innerHTML = `
      <strong>${result.risk_text}</strong>
      <span>Estimated probability of disease: ${probability}</span>
      <small>${result.disclaimer}</small>
    `;
    resultNode.classList.remove("hidden");
  } catch (error) {
    statusNode.classList.add("error");
    statusNode.textContent = error.message;
  }
});

loadMetadata().catch(() => {
  statusNode.textContent = "Model metadata could not be loaded.";
});
