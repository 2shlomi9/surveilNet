import { useRef, useState } from "react";
import { Link } from "react-router-dom";

/**
 * UploadVideo (no progress bar)
 * - Required metadata: start_time (datetime-local), location (string)
 * - Real cancel during upload (xhr.abort())
 * - After successful upload, triggers /api/process_video
 * - No progress UI/percentages
 */
function UploadVideo() {
  const [video, setVideo] = useState(null);
  const [message, setMessage] = useState("");
  const [processing, setProcessing] = useState(false);
  const [isUploading, setIsUploading] = useState(false);

  const [startTime, setStartTime] = useState("");
  const [location, setLocation] = useState("");

  // Cancel handles
  const cancelUploadRef = useRef(null);
  const canceledRef = useRef(false);

  const uploadWithCancel = (file, meta, onCancelRef) =>
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

      xhr.send(formData);
      onCancelRef.current = () => xhr.abort();
    });

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

    try {
      // 1) Upload (no progress UI)
      const uploadData = await uploadWithCancel(
        video,
        { start_time: new Date(startTime).toISOString(), location },
        cancelUploadRef
      );

      if (canceledRef.current) {
        setMessage("Upload canceled by user");
        return;
      }

      // 2) Process video (store embeddings only on server side)
      setProcessing(true);
      setMessage("Processing video…");

      const res = await fetch("http://localhost:5000/api/process_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: uploadData.filename }),
      });
      const data = await res.json();

      if (res.ok) {
        setMessage(`Success: ${data.message} (${data.video})`);
        setVideo(null);
      } else {
        setMessage(`Error processing video: ${data.error || "Unknown error"}`);
      }
    } catch (err) {
      if (String(err.message || "").includes("Upload canceled")) {
        setMessage("Upload canceled by user");
      } else {
        setMessage(`Error: ${err.message}`);
      }
    } finally {
      setIsUploading(false);
      setProcessing(false);
      cancelUploadRef.current = null;
    }
  };

  const handleCancelUpload = () => {
    canceledRef.current = true;
    cancelUploadRef.current?.();
    setIsUploading(false);
    setProcessing(false);
    setMessage("Upload canceled by user");
  };

  return (
    <div className="main-page">
      <div className="card">
        <h2>🎥 Upload Video</h2>

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
              {processing ? "Processing..." : "Process Video"}
            </button>
            {isUploading && (
              <button type="button" className="btn secondary" onClick={handleCancelUpload}>
                ✋ Cancel Upload
              </button>
            )}
          </div>
        </form>

        {message && <p style={{ marginTop: 10 }}>{message}</p>}

        <Link to="/">
          <button className="btn secondary">⬅ Back</button>
        </Link>
      </div>
    </div>
  );
}

export default UploadVideo;
