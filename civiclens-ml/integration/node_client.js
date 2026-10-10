// Drop this file into the Node backend (Node 18+, no extra packages needed).
// Environment variables on Vercel:  ML_SERVICE_URL=https://<your-ml-host>   ML_API_KEY=<same key as the ML service>

const ML_URL = process.env.ML_SERVICE_URL;
const ML_KEY = process.env.ML_API_KEY || "";

async function mlPost(path, body, timeoutMs = 25000) {
  if (!ML_URL) throw new Error("ML_SERVICE_URL is not set");
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), timeoutMs); // free hosts can take ~30 s to wake up
  try {
    const res = await fetch(`${ML_URL}${path}`, {
      method: "POST",
      headers: { "Content-Type": "application/json", ...(ML_KEY ? { "X-API-Key": ML_KEY } : {}) },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) throw new Error(`ML service ${path} failed: ${res.status} ${await res.text()}`);
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// 1) New report -> triage. Returns the full AI result (status, severity, ETA, materials, ...).
const analyzeReport = ({ imageUrl, userLat, userLng, submittedAt, distanceToDepotKm, trafficLevel, rainMm, nearSensitive, userTrust }) =>
  mlPost("/analyze", {
    image_url: imageUrl,
    user_lat: userLat,
    user_lng: userLng,
    submitted_at: submittedAt || new Date().toISOString(),
    distance_to_depot_km: distanceToDepotKm,
    traffic_level: trafficLevel,
    rain_mm: rainMm,
    near_sensitive: nearSensitive || [],
    user_trust: userTrust,
    return_annotated: false, // set true if the UI should show the boxed (privacy-blurred) image
  });

// 2) Worker uploads the "after" photo -> verify before closing the ticket.
const verifyRepair = (afterImageUrl, issueType) =>
  mlPost("/verify-repair", { after_image_url: afterImageUrl, issue_type: issueType });

// 3) Merge duplicates (run on a schedule, e.g. every 5 minutes, over recent OPEN reports).
const clusterReports = (reports, radiusM = 50, windowMinutes = 60) =>
  mlPost("/cluster", { reports, radius_m: radiusM, window_minutes: windowMinutes });

// 4) Reporter trust score (call when creating/updating a user's stats).
const trustScore = (stats) => mlPost("/trust-score", stats);

// 5) Admin dashboard heatmap + forecast.
const hotspots = (reports, cellSizeM = 500) => mlPost("/hotspots", { reports, cell_size_m: cellSizeM });

// 6) Free-text / multilingual complaint -> category + urgency.
const classifyComplaint = (text) => mlPost("/complaint/classify", { text });

module.exports = { analyzeReport, verifyRepair, clusterReports, trustScore, hotspots, classifyComplaint };

/* Example Express usage:

const ml = require("./node_client");

app.post("/api/reports", auth, async (req, res) => {
  const report = await db.createReport({ ...req.body, status: "PENDING_AI" });
  try {
    const ai = await ml.analyzeReport({ imageUrl: report.imageUrl, userLat: req.body.lat, userLng: req.body.lng });
    await db.saveTriage(report.id, ai);                       // see ai_triage.sql
    await db.updateReportStatus(report.id, ai.status);         // AUTO_DISPATCH | NEEDS_REVIEW | NEEDS_SECOND_REPORT | REJECTED
    res.json({ reportId: report.id, ai });
  } catch (err) {
    await db.updateReportStatus(report.id, "NEEDS_REVIEW");    // ML down? never lose the citizen's report
    res.status(202).json({ reportId: report.id, note: "Queued for manual review" });
  }
});
*/
