"""The demo page served at "/": type an applicant ID, see the score, the decision and why.

Plain HTML + a few lines of JavaScript, no external files or libraries, so it works offline.
All text from the API is inserted with textContent, never as HTML.
"""

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Credit risk scorecard</title>
<style>
  :root { --bg:#f6f4ee; --card:#fffdf8; --text:#1c1b18; --muted:#6e6b62; --line:#d9d5c8; --track:#e6e2d6;
          --ok:#1f7a4d; --bad:#b3261e; --accent:#243b73; --on-accent:#ffffff; color-scheme: light; }
  @media (prefers-color-scheme: dark) {
    :root { --bg:#141413; --card:#1c1c1a; --text:#ecebe5; --muted:#9d9a90; --line:#3a3935; --track:#2b2a27;
            --ok:#4cc38a; --bad:#ff7b72; --accent:#9db8ff; --on-accent:#0b1226; color-scheme: dark; } }
  * { box-sizing: border-box; }
  body { margin:0; background:var(--bg); color:var(--text); font:16px/1.5 system-ui, sans-serif; }
  main { max-width:680px; margin:0 auto; padding:36px 20px 72px; }
  .eyebrow { font-size:12px; letter-spacing:.14em; text-transform:uppercase; color:var(--muted); margin:0 0 6px; }
  h1 { font:600 28px/1.2 system-ui, -apple-system, 'Segoe UI', sans-serif; letter-spacing:-0.01em; margin:0 0 6px; }
  p.sub { margin:0 0 24px; color:var(--muted); }
  .card { background:var(--card); border:1px solid var(--line); border-radius:6px; padding:22px; margin-bottom:16px; }
  form { display:flex; gap:8px; flex-wrap:wrap; }
  input { flex:1; min-width:160px; padding:10px 12px; font-size:16px; border:1px solid var(--line);
          border-radius:4px; background:var(--bg); color:var(--text); }
  button { padding:10px 20px; font-size:16px; font-weight:600; border:0; border-radius:4px;
           background:var(--accent); color:var(--on-accent); cursor:pointer; }
  #result { display:none; }
  .head { display:flex; justify-content:space-between; align-items:center; gap:12px; flex-wrap:wrap; }
  .stamp { display:inline-block; padding:4px 14px; border:2px solid currentColor; border-radius:3px;
           font-weight:700; letter-spacing:.16em; text-transform:uppercase; font-size:14px; }
  .stamp.approve { color:var(--ok); } .stamp.decline { color:var(--bad); }
  .big { font:700 52px/1.05 system-ui, -apple-system, 'Segoe UI', sans-serif; letter-spacing:-0.02em; margin:14px 0 2px; font-variant-numeric: lining-nums tabular-nums; }
  .ctx { color:var(--muted); margin:0; }
  .bar { position:relative; height:12px; border-radius:2px; background:var(--track); margin:18px 0 6px; }
  .fill { height:100%; background:var(--accent); width:0; border-radius:2px; }
  .mark { position:absolute; top:-5px; width:2px; height:22px; background:var(--text); }
  .scale { display:flex; justify-content:space-between; font-size:13px; color:var(--muted); }
  h2 { font-size:13px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); margin:26px 0 8px;
       padding-top:18px; border-top:1px solid var(--line); font-weight:600; }
  .bands { display:flex; gap:4px; margin:8px 0 4px; }
  .seg { flex:1; height:12px; background:var(--track); border-radius:2px; }
  .seg.on { background:var(--text); }
  .legend { display:flex; justify-content:space-between; font-size:13px; color:var(--muted); margin:0 0 6px; }
  .frow { margin:12px 0; }
  .lab { display:flex; justify-content:space-between; gap:12px; font-size:15px; }
  .lab span { color:var(--muted); text-align:right; }
  .track { position:relative; height:10px; margin-top:5px; background:var(--track); border-radius:2px; }
  .axis { position:absolute; left:50%; top:-3px; width:1px; height:16px; background:var(--muted); }
  .b { position:absolute; top:0; height:100%; }
  .b.up { left:50%; background:var(--bad); } .b.down { right:50%; background:var(--ok); }
  .tabs { display:flex; gap:20px; border-bottom:1px solid var(--line); margin:0 0 16px; }
  .tab { background:none; border:0; border-bottom:2px solid transparent; padding:6px 0 8px; font-size:15px; font-weight:600;
         color:var(--muted); cursor:pointer; border-radius:0; }
  .tab.on { color:var(--text); border-bottom-color:var(--text); }
  .grid { display:grid; grid-template-columns:repeat(2, 1fr); gap:10px 12px; margin-bottom:12px; }
  .grid label { font-size:13px; color:var(--muted); display:block; }
  .grid input, .grid select { width:100%; min-width:0; margin-top:3px; padding:8px 10px; font-size:15px;
        border:1px solid var(--line); border-radius:4px; background:var(--bg); color:var(--text); }
  .err { color:var(--bad); }
  .note { font-size:14px; color:var(--muted); }
</style>
</head>
<body>
<main>
  <p class="eyebrow">Home Credit dataset | portfolio project</p>
  <h1>Credit risk scorecard</h1>
  <p class="sub">Score an invented applicant, or look one up, to see the estimated chance of default, the decision, and what moved the score.</p>

  <div class="card">
    <div class="tabs">
      <button type="button" class="tab on" id="tabCustom">Try a made-up applicant</button>
      <button type="button" class="tab" id="tabLookup">Look up an applicant ID</button>
    </div>

    <form id="fc">
      <div class="grid">
        <label>External credit score 1 (0 to 1)<input id="c_ext_source_1" inputmode="decimal" value="0.50"></label>
        <label>External credit score 2 (0 to 1)<input id="c_ext_source_2" inputmode="decimal" value="0.60"></label>
        <label>External credit score 3 (0 to 1)<input id="c_ext_source_3" inputmode="decimal" value="0.50"></label>
        <label>Age (years)<input id="c_age_years" inputmode="decimal" value="35"></label>
        <label>Years in current job<input id="c_employment_years" inputmode="decimal" value="5"></label>
        <label>Education
          <select id="c_education_level">
            <option value="">unknown</option><option value="0">Lower secondary</option><option value="1" selected>Secondary</option>
            <option value="2">Incomplete higher</option><option value="3">Higher education</option><option value="4">Academic degree</option>
          </select></label>
        <label>Yearly income<input id="c_income_total" inputmode="decimal" value="150000"></label>
        <label>Loan amount<input id="c_credit_amount" inputmode="decimal" value="500000"></label>
        <label>Price of goods financed<input id="c_goods_price" inputmode="decimal" value="450000"></label>
        <label>Loan repayment per year<input id="c_annuity_amount" inputmode="decimal" value="25000"></label>
      </div>
      <button type="submit">Score</button>
    </form>
    <p class="note" id="customNote">Every value here is invented, and no real applicant is looked up. Anything you clear is treated as unknown.
       The starting values are round example numbers.</p>

    <form id="f" style="display:none">
      <input id="id" inputmode="numeric" placeholder="Applicant ID, e.g. 100002" value="100002" aria-label="Applicant ID">
      <button type="submit">Score</button>
    </form>
    <p class="note" id="lookupNote" style="display:none">Looks up a record from the local feature store, so it only works on a machine that has the data.</p>
    <p class="note" id="status"></p>
  </div>

  <div class="card" id="result">
    <div class="head"><span class="stamp" id="decision"></span><span class="note" id="who"></span></div>
    <div class="big" id="prob"></div>
    <p class="ctx" id="context"></p>
    <div class="bar"><div class="fill" id="fill"></div><div class="mark" id="mark"></div></div>
    <div class="scale"><span>0%</span><span id="markLabel"></span><span>40%+</span></div>

    <div id="bandBlock">
      <h2>Where this applicant sits</h2>
      <div class="bands" id="bands"></div>
      <div class="legend"><span>1 safest</span><span id="bandText"></span><span>10 riskiest</span></div>
      <p class="note">The average is pulled up by a smaller group of high-risk applicants, so most applicants score below it.</p>
    </div>

    <h2>What moved the score</h2>
    <div class="legend"><span>&larr; lowers risk</span><span>raises risk &rarr;</span></div>
    <div id="factors"></div>
    <p class="note">Each bar shows how far one fact pushed this applicant's score up or down compared with a typical applicant. Longer means a bigger push.</p>
  </div>

  <p class="note">Not a real lending decision. The decline threshold rests on an assumed cost ratio.
     API docs: <a href="/docs" style="color:var(--accent)">/docs</a>.</p>
</main>
<script>
const $ = (id) => document.getElementById(id);
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }

function render(d) {
  const pct = d.default_probability * 100, thr = d.threshold * 100, scaleMax = 40;
  $("decision").textContent = d.decision === "decline" ? "Decline" : "Approve";
  $("decision").className = "stamp " + d.decision;
  $("who").textContent = (d.sk_id_curr == null ? "Made-up applicant" : "Applicant " + d.sk_id_curr) + " | model v" + d.model_version;
  $("prob").textContent = pct.toFixed(1) + "%";
  const avg = d.average_default_probability;
  $("context").textContent = "estimated probability of default" + (avg != null ? " (the average applicant: " + (avg * 100).toFixed(1) + "%)" : "");
  $("fill").style.width = Math.min(pct / scaleMax * 100, 100) + "%";
  $("mark").style.left = Math.min(thr / scaleMax * 100, 100) + "%";
  $("markLabel").textContent = "decline at " + thr.toFixed(1) + "%";

  const bandBlock = $("bandBlock");
  if (d.risk_band != null) {
    bandBlock.style.display = "block";
    const bands = $("bands"); bands.textContent = "";
    for (let i = 1; i <= 10; i++) bands.appendChild(el("div", "seg" + (i === d.risk_band ? " on" : "")));
    $("bandText").textContent = "band " + d.risk_band + " of 10: riskier than " + d.riskier_than_pct + "% of applicants";
  } else { bandBlock.style.display = "none"; }

  const box = $("factors"); box.textContent = "";
  const maxAbs = Math.max(...d.factors.map((f) => Math.abs(f.contribution)), 1e-9);
  d.factors.forEach((f) => {
    const row = el("div", "frow");
    const lab = el("div", "lab"); lab.appendChild(el("b", "", f.label)); lab.appendChild(el("span", "", f.value_text));
    const track = el("div", "track"); track.appendChild(el("div", "axis"));
    const bar = el("div", "b " + (f.direction === "raises" ? "up" : "down"));
    bar.style.width = (Math.abs(f.contribution) / maxAbs * 50) + "%";
    track.appendChild(bar); row.appendChild(lab); row.appendChild(track); box.appendChild(row);
  });
  $("result").style.display = "block";
}

function setMode(custom) {
  $("fc").style.display = custom ? "block" : "none"; $("customNote").style.display = custom ? "block" : "none";
  $("f").style.display = custom ? "none" : "flex"; $("lookupNote").style.display = custom ? "none" : "block";
  $("tabCustom").className = "tab" + (custom ? " on" : ""); $("tabLookup").className = "tab" + (custom ? "" : " on");
  $("result").style.display = "none"; $("status").textContent = "";
}
$("tabCustom").addEventListener("click", () => setMode(true));
$("tabLookup").addEventListener("click", () => setMode(false));

const CUSTOM_FIELDS = ["ext_source_1", "ext_source_2", "ext_source_3", "age_years", "employment_years",
                       "income_total", "credit_amount", "goods_price", "annuity_amount", "education_level"];
async function post(url, body) {
  const status = $("status"); status.className = "note"; status.textContent = "Scoring...";
  $("result").style.display = "none";
  try {
    const r = await fetch(url, { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
    const d = await r.json();
    if (!r.ok) { status.className = "note err";
      status.textContent = typeof d.detail === "string" ? d.detail : "Some value is outside its allowed range (scores 0 to 1, age 18 to 100)."; return; }
    status.textContent = ""; render(d);
  } catch (err) { status.className = "note err"; status.textContent = "Could not reach the scoring service."; }
}
$("fc").addEventListener("submit", (e) => {
  e.preventDefault();
  const body = {};
  CUSTOM_FIELDS.forEach((k) => {
    const raw = $("c_" + k).value.trim();
    const v = raw === "" ? null : Number(raw.replace(/,/g, ""));
    body[k] = v === null || Number.isNaN(v) ? null : v;
  });
  post("/score-custom", body);
});

$("f").addEventListener("submit", async (e) => {
  e.preventDefault();
  const status = $("status"); status.className = "note"; status.textContent = "Scoring...";
  $("result").style.display = "none";
  const id = parseInt($("id").value, 10);
  if (!Number.isInteger(id) || id <= 0) { status.className = "note err"; status.textContent = "Enter a positive whole number."; return; }
  try {
    const r = await fetch("/score", { method: "POST", headers: {"Content-Type": "application/json"},
                                      body: JSON.stringify({ sk_id_curr: id }) });
    const d = await r.json();
    if (!r.ok) { status.className = "note err";
                 status.textContent = typeof d.detail === "string" ? d.detail : "Request was rejected."; return; }
    status.textContent = "";
    render(d);
  } catch (err) {
    status.className = "note err"; status.textContent = "Could not reach the scoring service.";
  }
});
</script>
</body>
</html>
"""
