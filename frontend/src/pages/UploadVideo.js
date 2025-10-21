import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

/**
 * UploadVideo with server-side processing progress + Cancel in both phases
 * - Upload phase: Cancel aborts XHR and removes partial file on server.
 * - Processing phase: Tries /api/process_cancel?job_id=... (if exists), otherwise stops polling gracefully.
 */
function UploadVideo() {
  const [video, setVideo] = useState(null);
  const [message, setMessage] = useState("");
  const [processing, setProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const [startTime, setStartTime] = useState("");
  const [location, setLocation] = useState("");

  // Processing progress
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);

  // Cancel handles
  const uploadXhrRef = useRef(null);  // holds the xhr during upload
  const canceledRef = useRef(false);

  // Polling handle
  const pollRef = useRef(null);

  useEffect(() => {
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  const uploadWithCancel = (file, meta, xhrRef) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "http://localhost:5000/api/upload_video");

      xhr.onload = () => {
        const status = xhr.status;
        const text = xhr.responseText || "{}";
        let resp = {};
        try { resp = JSON.parse(text); } catch {}
        if (status >= 200 && status < 300) return resolve(resp);
        if (status === 499) return reject(new Error("Upload canceled by user"));
        return reject(new Error(resp.error || `HTTP ${status}`));
      };

      xhr.onerror = () => reject(new Error("Network error"));
      xhr.onabort = () => reject(new Error("Upload canceled by user"));

      const formData = new FormData();
      formData.append("video", file);
      formData.append("start_time", meta.start_time);
      formData.append("location", meta.location);

      xhrRef.current = xhr;     // keep xhr so we can abort later
      xhr.send(formData);
    });

  const startAsyncProcess = async (filename) => {
    const res = await fetch("http://localhost:5000/api/process_video_async", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    const data = await res.json();
    if (!res.ok) throw new Error(data.error || "Failed to start processing");

    setJobId(data.job_id);
    setProgress(0);
    setProcessing(true);
    setMessage("Processing video…");

    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const st = await fetch(
          `http://localhost:5000/api/process_status?job_id=${encodeURIComponent(data.job_id)}`
        );
        const sd = await st.json();
        if (st.ok) {
          const pct = typeof sd.percent === "number" ? sd.percent : 0;
          setProgress(pct);
          if (sd.done) {
            clearInterval(pollRef.current);
            pollRef.current = null;
            setProgress(100);
            setProcessing(false);
            setMessage(`Success: processing complete (${sd.filename || filename})`);
          }
        } else {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setProcessing(false);
          setMessage(sd.error || "Failed to read progress");
        }
      } catch (err) {
        clearInterval(pollRef.current);
        pollRef.current = null;
        setProcessing(false);
        setMessage(String(err.message || err));
      }
    }, 500);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (!video) { setMessage("Please select a video"); return; }
    if (!startTime || !location) { setMessage("Please fill required fields: start time and location"); return; }

    canceledRef.current = false;
    setMessage("Uploading video…");
    setIsUploading(true);
    setProcessing(false);
    setJobId(null);
    setProgress(0);

    try {
      // 1) Upload (with cancel support)
      const uploadData = await uploadWithCancel(
        video,
        { start_time: new Date(startTime).toISOString(), location },
        uploadXhrRef
      );

      if (canceledRef.current) {
        setMessage("Upload canceled by user");
        return;
      }

      // Upload finished -> hide "cancel upload" state now
      setIsUploading(false);
      uploadXhrRef.current = null;

      // 2) Start async processing and show progress bar
      await startAsyncProcess(uploadData.filename);

      // Reset file input after successful start
      setVideo(null);
    } catch (err) {
      if (String(err.message || "").includes("Upload canceled")) {
        setMessage("Upload canceled by user");
      } else {
        setMessage(`Error: ${err.message}`);
      }
      setIsUploading(false);
      uploadXhrRef.current = null;
    }
  };

  // Cancel behavior:
  // - If currently uploading: abort XHR.
  // - Else if currently processing: try server-side cancel (if endpoint exists), otherwise stop polling gracefully.
  const handleCancel = async () => {
    if (isUploading && uploadXhrRef.current) {
      try { uploadXhrRef.current.abort(); } catch {}
      setIsUploading(false);
      setProcessing(false);
      setJobId(null);
      setProgress(0);
      setMessage("Upload canceled by user");
      uploadXhrRef.current = null;
      return;
    }

    if (processing && jobId) {
      // Try to cancel processing on server (optional endpoint)
      try {
        const res = await fetch(
          `http://localhost:5000/api/process_cancel?job_id=${encodeURIComponent(jobId)}`,
          { method: "POST" }
        );
        // Even אם אין endpoint, ניפול חינני ללוגיקה למטה
        if (res.ok) {
          setMessage("Processing canceled");
        } else {
          const data = await res.json().catch(() => ({}));
          setMessage(data.error || "Cancel request sent (server did not confirm)");
        }
      } catch {
        setMessage("Cancel request sent (client stopped polling)");
      } finally {
        if (pollRef.current) { clearInterval(pollRef.current); pollRef.current = null; }
        setProcessing(false);
        setJobId(null);
        setProgress(0);
      }
    }
  };

  return (
    <div className="main-page">
      <div className="card">
        <h2>🎥 Upload & Process Video</h2>

        <form
          onSubmit={handleSubmit}
          style={{ gap: 16, display: "flex", flexDirection: "column", alignItems: "stretch" }}
        >
          <label style={{ textAlign: "left", fontWeight: 600 }}>Start Time (required):</label>
          <input
            type="datetime-local"
            value={startTime}
            onChange={(e) => setStartTime(e.target.value)}
            required
          />

          <label style={{ textAlign: "left", fontWeight: 600 }}>Location (required):</label>
          <input
            type="text"
            placeholder="e.g., Main Gate / North Entrance"
            value={location}
            onChange={(e) => setLocation(e.target.value)}
            required
          />

          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideo(e.target.files?.[0] || null)}
          />

          <div style={{ display: "flex", gap: 8, justifyContent: "center" }}>
            <button type="submit" className="btn" disabled={processing || isUploading}>
              {processing ? "Processing..." : isUploading ? "Uploading…" : "Upload & Process"}
            </button>

            {/* Keep cancel visible both when uploading and when processing */}
            {(isUploading || processing) && (
              <button type="button" className="btn secondary" onClick={handleCancel}>
                ✋ {isUploading ? "Cancel Upload" : "Cancel Processing"}
              </button>
            )}
          </div>
        </form>

        {/* Processing progress bar */}
        {jobId && (
          <div style={{ marginTop: 12 }}>
            <div
              style={{
                height: 12,
                borderRadius: 8,
                background: "rgba(255,255,255,0.15)",
                overflow: "hidden",
                width: 420,
                maxWidth: "100%",
                margin: "0 auto",
              }}
            >
              <div
                style={{
                  height: "100%",
                  width: `${progress}%`,
                  background: "#22c55e",
                  transition: "width 0.2s linear",
                }}
              />
            </div>
            <div style={{ marginTop: 6, fontSize: 12 }}>{progress}%</div>
          </div>
        )}

        {message && <p style={{ marginTop: 10 }}>{message}</p>}

        <Link to="/"><button className="btn secondary">⬅ Back</button></Link>
      </div>
    </div>
  );
}

export default UploadVideo;
