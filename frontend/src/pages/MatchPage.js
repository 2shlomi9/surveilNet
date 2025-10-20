import { useEffect, useState, useCallback } from "react";
import { Link } from "react-router-dom";
import "./MatchPage.css";

/**
 * MatchPage
 * - Loads best-per-person matches from /api/matches_feed?best_per_person=1
 * - 3-column responsive grid of compact cards
 * - Each card shows: reference image, matched frame, meta, delete, and "Watch 10s clip"
 * - Clip modal supports MP4 or GIF (server returns {kind: "video"|"gif", url: ...})
 */
function MatchPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(true);
  const [msg, setMsg] = useState("");

  // Clip modal
  const [clipOpen, setClipOpen] = useState(false);
  const [clipUrl, setClipUrl] = useState("");
  const [clipKind, setClipKind] = useState("video"); // "video" | "gif"
  const [clipLoading, setClipLoading] = useState(false);
  const [clipErr, setClipErr] = useState("");

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
    if (!id) return;
    try {
      const url = `http://localhost:5000/api/matches_feed/${encodeURIComponent(id)}`.trim();
      const res = await fetch(url, { method: "DELETE" });
      const data = await res.json().catch(() => ({}));
      if (!res.ok) {
        alert(data.error || "Failed to delete match");
        return;
      }
      setItems((prev) => prev.filter((x) => x.id !== id));
    } catch (err) {
      alert(String(err.message || err));
    }
  };

  const openClip = async (m) => {
    setClipErr("");
    setClipLoading(true);
    setClipUrl("");
    setClipKind("video");
    setClipOpen(true);

    try {
      const params = new URLSearchParams({
        video: m.video || "",
        frame_idx: String(m.frame_idx ?? ""),
        window: "5",
        annotate: "1",
      });
      if (typeof m.fps === "number") params.set("fps", String(m.fps));
      if (Array.isArray(m.box) && m.box.length === 4) params.set("box", m.box.join(","));

      const res = await fetch(`http://localhost:5000/api/video_snippet?${params.toString()}`);
      const ct = res.headers.get("content-type") || "";
      const isJson = ct.includes("application/json");
      const data = isJson ? await res.json() : await res.text();

      if (!res.ok) {
        setClipErr((isJson ? data.error : String(data)) || `HTTP ${res.status}`);
        return;
      }

      const absolute = `http://localhost:5000${data.url}`;
      setClipKind(data.kind || "video");
      setClipUrl(absolute);
    } catch (err) {
      setClipErr(String(err.message || err));
    } finally {
      setClipLoading(false);
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
                  <div style={{ display: "flex", gap: 6 }}>
                    <button className="btn secondary" onClick={() => openClip(m)} title="Watch 10s clip">▶ Watch</button>
                    <button className="icon-btn danger" onClick={() => handleDelete(m.id)} title="Delete">✖</button>
                  </div>
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

        {/* Clip modal */}
        {clipOpen && (
          <div
            style={{
              position: "fixed",
              inset: 0,
              background: "rgba(0,0,0,0.6)",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              padding: 16,
              zIndex: 1000,
            }}
            onClick={() => setClipOpen(false)}
          >
            <div
              style={{ maxWidth: 900, width: "100%", background: "rgba(20,20,20,0.95)", borderRadius: 12, padding: 12 }}
              onClick={(e) => e.stopPropagation()}
            >
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h3 style={{ margin: 0 }}>🎬 10s clip</h3>
                <button className="icon-btn" onClick={() => setClipOpen(false)}>✖</button>
              </div>

              <div style={{ marginTop: 10 }}>
                {clipLoading && <p>Preparing clip…</p>}
                {clipErr && <p className="error">{clipErr}</p>}

                {clipUrl && clipKind === "video" && (
                  <video
                    src={clipUrl}
                    controls
                    autoPlay
                    style={{ width: "100%", borderRadius: 8, outline: "none" }}
                  />
                )}

                {clipUrl && clipKind === "gif" && (
                  <img
                    src={clipUrl}
                    alt="snippet gif"
                    style={{ width: "100%", borderRadius: 8, display: "block" }}
                  />
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default MatchPage;
