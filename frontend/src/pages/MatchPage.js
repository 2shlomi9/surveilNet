import { useEffect, useState } from "react";
import { Link } from "react-router-dom";

/**
 * MatchPage
 * - Displays all saved matches (from /api/matches_feed) in a clean two-column layout.
 * - Left: reference image (person_main_image) + person name
 * - Right: matched frame from video + details (score, place, time, video, frame)
 */
function MatchPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState("");

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch("http://localhost:5000/api/matches_feed");
        const data = await res.json();
        if (!res.ok) {
          setMsg(data.error || "Failed to load matches");
          return;
        }
        setItems(data.matches || []);
      } catch (err) {
        setMsg(String(err.message || err));
      } finally {
        setLoading(false);
      }
    };
    load();
  }, []);

  return (
    <div className="main-page">
      <div className="card" style={{ maxWidth: 1100 }}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
          <h2>🔎 Matches</h2>
          <Link to="/"><button className="btn secondary">⬅ Back</button></Link>
        </div>

        {loading && <p>Loading…</p>}
        {msg && <p style={{ color: "salmon" }}>{msg}</p>}

        {!loading && items.length === 0 && <p>No matches yet.</p>}

        <div className="matches-grid">
          {items.map((m) => {
            // frame image URL (prefer frame_url; if missing, fallback to api)
            const frameUrl = m.frame_url
              ? `http://localhost:5000${m.frame_url}`
              : m.frame_image
                ? `http://localhost:5000/api/frame_image?path=${encodeURIComponent(m.frame_image)}`
                : null;

            const personImg = m.person_main_image || null;

            const ts = m.ts ? new Date(m.ts * 1000).toLocaleString() : "";
            const timeStr = m.time || "";

            return (
              <div className="match-card" key={m.id}>
                {/* Left column: uploaded / reference image */}
                <div className="col">
                  <div className="img-wrap">
                    {personImg ? (
                      <img src={personImg} alt={m.person_name || "reference"} />
                    ) : (
                      <div className="img-placeholder">No reference image</div>
                    )}
                  </div>
                  <div className="meta">
                    <div className="title">{m.person_name || "Unknown person"}</div>
                    <div className="sub">Saved: {ts || "-"}</div>
                  </div>
                </div>

                {/* Right column: matched video frame */}
                <div className="col">
                  <div className="img-wrap">
                    {frameUrl ? (
                      <img src={frameUrl} alt="matched frame" />
                    ) : (
                      <div className="img-placeholder">Frame unavailable</div>
                    )}
                  </div>
                  <div className="meta">
                    <div className="title">Best frame</div>
                    <div className="kv">
                      <span>Score</span><span>{(m.score ?? "").toFixed ? m.score.toFixed(3) : m.score}</span>
                    </div>
                    <div className="kv">
                      <span>Place</span><span>{m.place || "-"}</span>
                    </div>
                    <div className="kv">
                      <span>Time</span><span>{timeStr || "-"}</span>
                    </div>
                    <div className="kv">
                      <span>Video</span><span>{m.video || "-"}</span>
                    </div>
                    <div className="kv">
                      <span>Frame</span><span>{m.frame_idx ?? "-"}</span>
                    </div>
                  </div>
                </div>
              </div>
            );
          })}
        </div>

      </div>
    </div>
  );
}

export default MatchPage;
