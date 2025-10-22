import React, { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import "./MatchPage.css";

function MatchCard({ item, onDelete, onWatch }) {
  const score = typeof item.score === "number" ? item.score.toFixed(3) : item.score;

  const frameImg = item.frame_url
    ? `http://localhost:5000${item.frame_url}`
    : item.frame_image
    ? `http://localhost:5000/api/frame_image?path=${encodeURIComponent(item.frame_image)}`
    : null;

  return (
    <div className="mp-card">
      <div className="mp-card-head">
        <div className="mp-title">{item.person_name || "Match"}</div>
        <div className="mp-actions">
          <button className="btn secondary" onClick={() => onWatch(item)}>▶ Watch</button>
          <button className="icon-btn danger" onClick={() => onDelete(item.id)}>✕</button>
        </div>
      </div>

      <div className="mp-media">
        <div className="mp-media-tile">
          {item.person_main_image ? (
            <img src={item.person_main_image} alt="reference" />
          ) : (
            <div className="mp-img-placeholder">Reference</div>
          )}
          <div className="mp-caption">Reference</div>
        </div>
        <div className="mp-media-tile">
          {frameImg ? (
            <img src={frameImg} alt="matched frame" />
          ) : (
            <div className="mp-img-placeholder">Matched frame</div>
          )}
          <div className="mp-caption">Matched frame</div>
        </div>
      </div>

      <div className="mp-meta">
        <div className="kv"><span title="Score">Score</span><span title={score ?? "—"}>{score ?? "—"}</span></div>
        <div className="kv"><span title="Place">Place</span><span title={item.place || "—"}>{item.place || "—"}</span></div>
        <div className="kv"><span title="Time">Time</span><span title={item.time || "—"}>{item.time || "—"}</span></div>
        <div className="kv"><span title="Video">Video</span><span title={item.video || "—"}>{item.video || "—"}</span></div>
        <div className="kv"><span title="Frame">Frame</span><span title={String(item.frame_idx ?? "—")}>{item.frame_idx ?? "—"}</span></div>
      </div>
    </div>
  );
}

export default function MatchPage() {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState("");
  const [clipOpen, setClipOpen] = useState(false);
  const [clipUrl, setClipUrl] = useState("");
  const [clipKind, setClipKind] = useState("video");
  const [clipLoading, setClipLoading] = useState(false);
  const [clipErr, setClipErr] = useState("");
  const scrollRef = useRef(null);

  useEffect(() => {
    const prev = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = prev;
    };
  }, []);

  const load = async () => {
    setLoading(true);
    setErr("");
    try {
      const res = await fetch("http://localhost:5000/api/matches_feed?best_per_person=1");
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setItems(data.matches || []);
    } catch (e) {
      setErr(String(e.message || e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { load(); }, []);

  const handleDelete = async (id) => {
    try {
      const res = await fetch(`http://localhost:5000/api/matches_feed/${id}`, { method: "DELETE" });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || `HTTP ${res.status}`);
      setItems((prev) => prev.filter((x) => x.id !== id));
    } catch (e) {
      alert(`Delete failed: ${e.message || e}`);
    }
  };

  const handleWatch = async (item) => {
    setClipOpen(true);
    setClipLoading(true);
    setClipUrl("");
    setClipErr("");
    setClipKind("video");
    try {
      const params = new URLSearchParams({
        video: item.video || "",
        frame_idx: String(item.frame_idx ?? ""),
        window: "5",
        annotate: "1",
      });
      if (typeof item.fps === "number") params.set("fps", String(item.fps));
      if (Array.isArray(item.box) && item.box.length === 4) params.set("box", item.box.join(","));
      const res = await fetch(`http://localhost:5000/api/video_snippet?${params.toString()}`);
      const ct = res.headers.get("content-type") || "";
      const isJson = ct.includes("application/json");
      const payload = isJson ? await res.json() : await res.text();
      if (!res.ok) throw new Error((isJson ? payload.error : String(payload)) || `HTTP ${res.status}`);
      setClipKind(payload.kind || "video");
      setClipUrl(`http://localhost:5000${payload.url}`);
    } catch (e) {
      setClipErr(String(e.message || e));
    } finally {
      setClipLoading(false);
    }
  };

  const scrollTop = () => {
    if (scrollRef.current) scrollRef.current.scrollTo({ top: 0, behavior: "smooth" });
  };

  return (
    <div className="matches-page">
      <div className="matches-toolbar">
        <div className="left">
          <h2>🔎 Matches (best per person)</h2>
        </div>
        <div className="right">
          <button className="btn secondary" onClick={load} disabled={loading}>⟳ Refresh</button>
          <button className="btn secondary" onClick={scrollTop}>⬆ Back to top</button>
          <Link to="/"><button className="btn secondary">⬅ Back</button></Link>
        </div>
      </div>

      <div className="matches-scroll" ref={scrollRef}>
        {loading && <p className="hint">Loading…</p>}
        {err && <p className="error">{err}</p>}

        <div className="matches-grid">
          {items.map((it) => (
            <MatchCard
              key={it.id}
              item={it}
              onDelete={handleDelete}
              onWatch={handleWatch}
            />
          ))}
        </div>
      </div>

      {clipOpen && (
        <div className="mp-modal" onClick={() => setClipOpen(false)}>
          <div className="mp-modal-inn" onClick={(e) => e.stopPropagation()}>
            <div className="mp-modal-head">
              <h3 style={{ margin: 0 }}>🎬 10s clip</h3>
              <button className="icon-btn" onClick={() => setClipOpen(false)}>✖</button>
            </div>
            <div style={{ marginTop: 10 }}>
              {clipLoading && <p>Preparing clip…</p>}
              {clipErr && <p className="error">{clipErr}</p>}
              {clipUrl && clipKind === "video" && (
                <video src={clipUrl} controls autoPlay style={{ width: "100%", borderRadius: 8 }} />
              )}
              {clipUrl && clipKind === "gif" && (
                <img src={clipUrl} alt="snippet gif" style={{ width: "100%", borderRadius: 8 }} />
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}