import { useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import "./MatchPage.css";

/**
 * MatchPage
 * - Loads only best-per-person matches from /api/matches_feed?best_per_person=1
 * - Displays compact, separated cards in a 3-column responsive grid
 * - Each card shows: reference image, matched frame, details, and a delete button
 */
function MatchPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState("");

  const fetchFeed = useCallback(async () => {
    setLoading(true);
    setMsg("");
    try {
      const res = await fetch("http://localhost:5000/api/matches_feed?best_per_person=1");
      const data = await res.json();
      if (!res.ok) {
        setMsg(data.error || "Failed to load matches");
        setItems([]);
        return;
      }
      setItems(data.matches || []);
    } catch (err) {
      setMsg(String(err.message || err));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => { fetchFeed(); }, [fetchFeed]);

const handleDelete = async (id) => {
  if (!id) {
    alert("No match id");
    return;
  }
  try {
    const url = (`http://localhost:5000/api/matches_feed/${encodeURIComponent(id)}`).trim(); // <-- trim!
    const res = await fetch(url, { method: "DELETE" });
    const data = await res.json().catch(() => ({}));
    if (!res.ok) {
      console.error("Delete error:", res.status, data);
      alert(data.error || `Failed to delete (HTTP ${res.status})`);
      return;
    }
    setItems((prev) => prev.filter((x) => x.id !== id));
  } catch (err) {
    console.error("Network error:", err);
    alert("Network error while deleting (see console)");
  }
};

  return (
    <div className="main-page">
      <div className="card" style={{ maxWidth: 1280 }}>
        <div className="header-row">
          <h2>🔎 Matches (best per person)</h2>
          <div className="header-actions">
            <button className="btn secondary" onClick={fetchFeed}>↻ Refresh</button>
            <Link to="/"><button className="btn secondary">⬅ Back</button></Link>
          </div>
        </div>

        {loading && <p>Loading…</p>}
        {msg && <p className="error">{msg}</p>}

        {!loading && items.length === 0 && <p>No matches yet.</p>}

        <div className="cards-grid">
          {items.map((m) => {
            const refImg = m.person_main_image || null;
            const frameUrl = m.frame_url
              ? `http://localhost:5000${m.frame_url}`
              : m.frame_image
                ? `http://localhost:5000/api/frame_image?path=${encodeURIComponent(m.frame_image)}`
                : null;

            const scoreStr = (m.score ?? "").toFixed ? m.score.toFixed(3) : m.score;
            const when = m.time || (m.ts ? new Date(m.ts * 1000).toLocaleString() : "-");

            return (
              <div className="match-card compact" key={m.id}>
                <div className="card-head">
                  <div className="person-name" title={m.person_name || ""}>
                    {m.person_name || "Unknown person"}
                  </div>
                  <button className="icon-btn danger" onClick={() => handleDelete(m.id)} title="Delete">
                    ✖
                  </button>
                </div>

                <div className="imgs-row">
                  <div className="img-tile">
                    {refImg ? (
                      <img src={refImg} alt="reference" />
                    ) : (
                      <div className="img-placeholder">No reference</div>
                    )}
                    <div className="img-caption">Reference</div>
                  </div>
                  <div className="img-tile">
                    {frameUrl ? (
                      <img src={frameUrl} alt="matched frame" />
                    ) : (
                      <div className="img-placeholder">No frame</div>
                    )}
                    <div className="img-caption">Matched frame</div>
                  </div>
                </div>

                <div className="meta-grid">
                  <div className="kv"><span>Score</span><span>{scoreStr ?? "-"}</span></div>
                  <div className="kv"><span>Place</span><span>{m.place || "-"}</span></div>
                  <div className="kv"><span>Time</span><span>{when}</span></div>
                  <div className="kv"><span>Video</span><span>{m.video || "-"}</span></div>
                  <div className="kv"><span>Frame</span><span>{m.frame_idx ?? "-"}</span></div>
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
