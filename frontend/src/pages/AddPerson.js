import React, { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import "./AddPerson.css";

function AddPerson() {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName]   = useState("");
  const [age, setAge]             = useState("");

  const [images, setImages]       = useState([]);   // File[]
  const [previews, setPreviews]   = useState([]);   // object URLs
  const [message, setMessage]     = useState("");
  const [submitting, setSubmitting] = useState(false);

  // Best match returned by backend on save (or null)
  const [best, setBest] = useState(null);
  const [notFound, setNotFound] = useState(false);  // “no match found” flag

  // Clip modal
  const [clipOpen, setClipOpen] = useState(false);
  const [clipUrl, setClipUrl]   = useState("");
  const [clipKind, setClipKind] = useState("video");
  const [clipLoading, setClipLoading] = useState(false);
  const [clipErr, setClipErr] = useState("");

  // handle files
  const onFiles = (e) => {
    const files = Array.from(e.target.files || []);
    setImages(files);
    setBest(null);
    setNotFound(false);
  };

  // build previews from selected files
  useEffect(() => {
    previews.forEach((u) => URL.revokeObjectURL(u));
    if (!images || images.length === 0) {
      setPreviews([]);
      return;
    }
    const urls = images.map((f) => URL.createObjectURL(f));
    setPreviews(urls);
    return () => urls.forEach((u) => URL.revokeObjectURL(u));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [images]);

  // first preview acts as the reference visual
  const referenceSrc = useMemo(() => (previews[0] ? previews[0] : null), [previews]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    setMessage("");

    if (!firstName || !lastName || images.length === 0) {
      setMessage("Please fill first/last name and select at least one image.");
      return;
    }

    setSubmitting(true);
    setBest(null);
    setNotFound(false);

    try {
      const form = new FormData();
      form.append("first_name", firstName);
      form.append("last_name", lastName);
      if (age) form.append("age", age);
      images.forEach((f) => form.append("images", f));

      const res = await fetch("http://localhost:5000/api/people", {
        method: "POST",
        body: form,
      });
      const data = await res.json();

      if (!res.ok) {
        setMessage(data.error || "Failed to add person");
        return;
      }

      setMessage(data.message || "Person added");
      const lastSeen = data.last_seen || null;
      setBest(lastSeen);
      setNotFound(!lastSeen);
    } catch (err) {
      setMessage(String(err.message || err));
    } finally {
      setSubmitting(false);
    }
  };

  // open 10s clip for current best
  const openClip = async () => {
    if (!best) return;
    setClipErr("");
    setClipLoading(true);
    setClipUrl("");
    setClipKind("video");
    setClipOpen(true);

    try {
      const params = new URLSearchParams({
        video: best.video || "",
        frame_idx: String(best.frame_idx ?? ""),
        window: "5",
        annotate: "1",
      });
      if (typeof best.fps === "number") params.set("fps", String(best.fps));
      if (Array.isArray(best.box) && best.box.length === 4) params.set("box", best.box.join(","));

      const res = await fetch(`http://localhost:5000/api/video_snippet?${params.toString()}`);
      const ct = res.headers.get("content-type") || "";
      const isJson = ct.includes("application/json");
      const payload = isJson ? await res.json() : await res.text();

      if (!res.ok) {
        setClipErr((isJson ? payload.error : String(payload)) || `HTTP ${res.status}`);
        return;
      }

      const absolute = `http://localhost:5000${payload.url}`;
      setClipKind(payload.kind || "video");
      setClipUrl(absolute);
    } catch (err) {
      setClipErr(String(err.message || err));
    } finally {
      setClipLoading(false);
    }
  };

  // helpers for meta view
  const val = (x) => (x === 0 || x ? x : "—");
  const scoreStr = best?.score?.toFixed ? best.score.toFixed(3) : best?.score;

  return (
    <div className="ap-shell">
      <div className="ap-card">
        <div className="ap-header">
          <h2>Add Person</h2>
          <div className="ap-actions">
            <Link to="/"><button className="btn secondary" type="button">⬅ Back</button></Link>
          </div>
        </div>

        {/* form */}
        <form onSubmit={handleSubmit} className="ap-form">
          <div className="ap-field">
            <label>First name</label>
            <input value={firstName} onChange={(e) => setFirstName(e.target.value)} />
          </div>
          <div className="ap-field">
            <label>Last name</label>
            <input value={lastName} onChange={(e) => setLastName(e.target.value)} />
          </div>
          <div className="ap-field">
            <label>Age (optional)</label>
            <input value={age} onChange={(e) => setAge(e.target.value)} />
          </div>
          <div className="ap-field">
            <label>Images</label>
            <input type="file" accept="image/*" multiple onChange={onFiles} />
          </div>
          <div className="ap-buttons">
            <button className="btn" type="submit" disabled={submitting}>
              {submitting ? "Saving..." : "Save Person"}
            </button>
          </div>
        </form>

        {message && <p className="ap-msg">{message}</p>}

        {/* Always-visible match container */}
        <div className="ap-match">
          <div className="ap-match-head">
            <div className="ap-match-title">Best match</div>
            <button
              className="btn secondary"
              onClick={openClip}
              disabled={!best}
              title={best ? "Watch 10s clip" : "No match yet"}
            >
              ▶ Watch
            </button>
          </div>

          <div className="ap-media">
            {/* Reference (left) */}
            <div className="ap-media-tile">
              {referenceSrc ? (
                <img src={referenceSrc} alt="reference" />
              ) : (
                <div className="ap-img-placeholder">Upload an image to preview here</div>
              )}
              <div className="ap-caption">Reference</div>
            </div>

            {/* Matched frame (right) */}
            <div className="ap-media-tile">
              {best?.frame_url ? (
                <img src={`http://localhost:5000${best.frame_url}`} alt="matched frame" />
              ) : best?.frame_image ? (
                <img
                  src={`http://localhost:5000/api/frame_image?path=${encodeURIComponent(best.frame_image)}`}
                  alt="matched frame"
                />
              ) : (
                <div className="ap-img-placeholder">
                  {notFound ? "No match found in processed videos" : "No match yet"}
                </div>
              )}
              <div className="ap-caption">Matched frame</div>
            </div>
          </div>

          {/* meta row */}
          <div className="ap-meta">
            <div className="ap-kv"><span>Score</span><span>{best ? val(scoreStr) : "—"}</span></div>
            <div className="ap-kv"><span>Place</span><span>{best ? val(best.place) : "—"}</span></div>
            <div className="ap-kv"><span>Time</span><span>{best ? val(best.time_iso) : "—"}</span></div>
            <div className="ap-kv"><span>Video</span><span>{best ? val(best.video) : "—"}</span></div>
            <div className="ap-kv"><span>Frame</span><span>{best ? val(best.frame_idx) : "—"}</span></div>
          </div>
        </div>
      </div>

      {/* Clip modal */}
      {clipOpen && (
        <div className="ap-modal" onClick={() => setClipOpen(false)}>
          <div className="ap-modal-inner" onClick={(e) => e.stopPropagation()}>
            <div className="ap-modal-head">
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

export default AddPerson;
