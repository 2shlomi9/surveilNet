import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";

/**
 * UploadVideo (async processing with progress + cancel)
 *
 * Flow:
 *  1) Upload file to /api/upload_video (cancelable via xhr.abort()).
 *  2) Start async processing via /api/process_video_async -> returns {job_id}.
 *  3) Poll /api/process_status?job_id=... every 500ms and update the bar.
 *  4) Optional: Cancel processing via /api/process_cancel (best-effort).
 *
 * Notes:
 *  - Progress polling is tolerant to different server field names/types.
 *  - The "Cancel Upload" button remains visible while the XHR is in flight.
 *  - The "Cancel Processing" button appears during server-side processing.
 */
function UploadVideo() {
  const [video, setVideo] = useState(null);
  const [message, setMessage] = useState("");
  const [processing, setProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const [startTime, setStartTime] = useState("");
  const [location, setLocation] = useState("");

  // Async-processing progress state
  const [jobId, setJobId] = useState(null);
  const [progress, setProgress] = useState(0);

  // Cancel handles
  const cancelUploadRef = useRef(null);
  const canceledRef = useRef(false);

  // Polling handle
  const pollRef = useRef(null);

  useEffect(() => {
    // Cleanup polling on unmount
    return () => {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
    };
  }, []);

  const uploadWithCancel = (file, meta, onCancelRef) =>
    new Promise((resolve, reject) => {
      const xhr = new XMLHttpRequest();
      xhr.open("POST", "http://localhost:5000/api/upload_video");

      xhr.onload = () => {
        const status = xhr.status;
        const text = xhr.responseText || "{}";
        let resp = {};
        try {
          resp = JSON.parse(text);
        } catch {
          // ignore parse error; treat as generic HTTP failure below
        }
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

      xhr.send(formData);
      onCancelRef.current = () => xhr.abort();
    });

  const startAsyncProcess = async (filename) => {
    // Start server-side processing
    const res = await fetch("http://localhost:5000/api/process_video_async", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    const data = await res.json();
    if (!res.ok) {
      throw new Error(data.error || "Failed to start processing");
    }

    setJobId(data.job_id);
    setProgress(0);
    setProcessing(true);
    setMessage("Processing video…");

    // Poll every 500ms (tolerant to different server field names/types)
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const st = await fetch(
          `http://localhost:5000/api/process_status?job_id=${encodeURIComponent(
            data.job_id
          )}`
        );
        const sd = await st.json();

        if (!st.ok) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setProcessing(false);
          setMessage(sd.error || "Failed to read progress");
          return;
        }

        // Accept percent/progress/pct/percentage and number/string
        const rawPct =
          sd.percent ?? sd.progress ?? sd.pct ?? sd.percentage ?? 0;
        const pct = Math.max(0, Math.min(100, Number(rawPct) || 0));
        setProgress(pct);

        // Accept done/finished/status=done or percent>=100
        const status = (sd.status || "").toString().toLowerCase();
        const done =
          sd.done === true ||
          sd.finished === true ||
          status === "done" ||
          status === "finished" ||
          pct >= 100;

        if (done) {
          clearInterval(pollRef.current);
          pollRef.current = null;
          setProgress(100);
          setProcessing(false);
          setMessage(
            `Success: processing complete (${sd.filename || filename})`
          );
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

    if (!video) {
      setMessage("Please select a video");
      return;
    }
    if (!startTime || !location) {
      setMessage("Please fill required fields: start time and location");
      return;
    }

    canceledRef.current = false;
    setMessage("Uploading video…");
    setIsUploading(true);
    setProcessing(false);
    setJobId(null);
    setProgress(0);

    try {
      // 1) Upload (cancelable)
      const uploadData = await uploadWithCancel(
        video,
        { start_time: new Date(startTime).toISOString(), location },
        cancelUploadRef
      );

      if (canceledRef.current) {
        setMessage("Upload canceled by user");
        return;
      }

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
    } finally {
      setIsUploading(false);
      cancelUploadRef.current = null;
    }
  };

  const handleCancelUpload = () => {
    // Abort the XHR immediately
    canceledRef.current = true;
    cancelUploadRef.current?.();
    setIsUploading(false);
    setProcessing(false);
    setJobId(null);
    setProgress(0);
    setMessage("Upload canceled by user");
  };

  const handleCancelProcessing = async () => {
    // Best-effort cancel on the server
    try {
      if (jobId) {
        await fetch("http://localhost:5000/api/process_cancel", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ job_id: jobId }),
        });
      }
    } catch {
      // ignore network errors on cancel
    } finally {
      if (pollRef.current) {
        clearInterval(pollRef.current);
        pollRef.current = null;
      }
      setProcessing(false);
      setJobId(null);
      setProgress(0);
      setMessage("Processing canceled by user");
    }
  };

  return (
    <div className="main-page">
      <div className="card">
        <h2>🎥 Upload & Process Video</h2>

        <form
          onSubmit={handleSubmit}
          style={{
            gap: 16,
            display: "flex",
            flexDirection: "column",
            alignItems: "stretch",
          }}
        >
          <label style={{ textAlign: "left", fontWeight: 600 }}>
            Start Time (required):
          </label>
          <input
            type="datetime-local"
            value={startTime}
            onChange={(e) => setStartTime(e.target.value)}
            required
          />

          <label style={{ textAlign: "left", fontWeight: 600 }}>
            Location (required):
          </label>
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
            <button
              type="submit"
              className="btn"
              disabled={processing || isUploading}
            >
              {processing
                ? "Processing..."
                : isUploading
                ? "Uploading…"
                : "Upload & Process"}
            </button>

            {/* Show cancel button during upload */}
            {isUploading && (
              <button
                type="button"
                className="btn secondary"
                onClick={handleCancelUpload}
              >
                ✋ Cancel Upload
              </button>
            )}

            {/* Show cancel button during processing */}
            {processing && jobId && (
              <button
                type="button"
                className="btn secondary"
                onClick={handleCancelProcessing}
              >
                🖐 Cancel Processing
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
                width: 520,
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

        <Link to="/">
          <button className="btn secondary">⬅ Back</button>
        </Link>
      </div>
    </div>
  );
}

export default UploadVideo;
