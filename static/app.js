const form = document.getElementById("donorForm");
const resultEmpty = document.getElementById("resultEmpty");
const resultBody = document.getElementById("resultBody");
const resultError = document.getElementById("resultError");

const verdictDot = document.getElementById("verdictDot");
const verdictLabel = document.getElementById("verdictLabel");
const verdictSub = document.getElementById("verdictSub");
const barYes = document.getElementById("barYes");
const barNo = document.getElementById("barNo");
const pctYes = document.getElementById("pctYes");
const pctNo = document.getElementById("pctNo");

form.addEventListener("submit", async (e) => {
  e.preventDefault();

  const data = Object.fromEntries(new FormData(form).entries());

  resultError.hidden = true;
  const submitBtn = form.querySelector(".submit-btn");
  const originalText = submitBtn.textContent;
  submitBtn.textContent = "Reading…";
  submitBtn.disabled = true;

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(data),
    });
    const json = await res.json();

    if (!res.ok) {
      throw new Error(json.error || "Something went wrong.");
    }

    renderResult(json);
  } catch (err) {
    resultEmpty.hidden = true;
    resultBody.hidden = true;
    resultError.hidden = false;
    resultError.textContent = err.message;
  } finally {
    submitBtn.textContent = originalText;
    submitBtn.disabled = false;
  }
});

function renderResult(result) {
  resultEmpty.hidden = true;
  resultError.hidden = true;
  resultBody.hidden = false;

  const isAvailable = result.available;
  const probYes = result.probabilities["Yes"] ?? 0;
  const probNo = result.probabilities["No"] ?? 0;

  verdictDot.className = "verdict-dot " + (isAvailable ? "yes" : "no");
  verdictLabel.textContent = isAvailable ? "Likely available" : "Likely unavailable";
  verdictSub.textContent = isAvailable
    ? "This donor's profile matches people who show up when called."
    : "This donor's profile matches people who tend not to show up right now.";

  barYes.style.width = `${Math.round(probYes * 100)}%`;
  barNo.style.width = `${Math.round(probNo * 100)}%`;
  pctYes.textContent = `${Math.round(probYes * 100)}%`;
  pctNo.textContent = `${Math.round(probNo * 100)}%`;
}
