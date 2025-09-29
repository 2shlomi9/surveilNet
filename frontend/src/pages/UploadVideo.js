
import { useState } from "react";
import { Link } from "react-router-dom";

function UploadVideo() {
  const [video, setVideo] = useState(null);
  const [message, setMessage] = useState("");
  const [processing, setProcessing] = useState(false);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!video) {
      setMessage("Please select a video");
      return;
    }

    setMessage("Uploading video...");
    const formData = new FormData();
    formData.append("video", video);

    try {
      // Step 1: Upload the video
      const uploadRes = await fetch("http://localhost:5000/api/upload_video", {
        method: "POST",
        body: formData,
      });
      const uploadData = await uploadRes.json();
      if (!uploadRes.ok) {
        setMessage(`Error uploading video: ${uploadData.error}`);
        return;
      }

      // Step 2: Process the uploaded video
      setMessage("Processing video...");
      setProcessing(true);
      const processRes = await fetch("http://localhost:5000/api/process_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: uploadData.filename }),
      });
      const processData = await processRes.json();
      if (processRes.ok) {
        setMessage(`Success: ${processData.message} (${processData.video})`);
        setVideo(null); // Clear the input
      } else {
        setMessage(`Error processing video: ${processData.error}`);
      }
    } catch (err) {
      setMessage(`Error: ${err.message}`);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="main-page">
      <div className="card">
        <h2>🎥 Upload Video</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setVideo(e.target.files[0])}
          />
          <button
            type="submit"
            className="btn"
            disabled={processing || !video}
          >
            {processing ? "Processing..." : "Process Video"}
          </button>
        </form>
        {message && <p>{message}</p>} {/* Display success/error message */}
        <Link to="/">
          <button className="btn secondary">⬅ Back</button>
        </Link>
      </div>
    </div>
  );
}

export default UploadVideo;