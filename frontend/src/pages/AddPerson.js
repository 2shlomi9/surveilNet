import { useState, useMemo } from "react";
import { Link } from "react-router-dom";

/**
 * AddPerson
 * - Submits person data + images to /api/people (multipart/form-data).
 * - After success, renders "Last seen" card with place/time and the matched frame.
 * - Prefers server-provided lastSeen.frame_url (served by /frame_store/.. route).
 * - Falls back to /api/frame_image?path=... if frame_url is not available.
 */
function AddPerson() {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [age, setAge] = useState("");
  const [images, setImages] = useState([]);
  const [message, setMessage] = useState("");
  const [saving, setSaving] = useState(false);

  const [result, setResult] = useState(null);
  const [lastSeen, setLastSeen] = useState(null);

  // Preview of first selected image
  const previewUrl = useMemo(() => {
    if (!images || images.length === 0) return null;
    return URL.createObjectURL(images[0]);
  }, [images]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!firstName || !lastName) {
      setMessage("Please fill first and last name");
      return;
    }
    if (!images || images.length === 0) {
      setMessage("Please select at least one image");
      return;
    }

    setSaving(true);
    setMessage("Uploading person…");
    setResult(null);
    setLastSeen(null);

    try {
      const formData = new FormData();
      formData.append("first_name", firstName);
      formData.append("last_name", lastName);
      if (age) formData.append("age", age);
      images.forEach((file) => formData.append("images", file));

      const res = await fetch("http://localhost:5000/api/people", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();

      if (!res.ok) {
        setMessage(data.error || "Failed to add person");
        setSaving(false);
        return;
      }

      setMessage(`Success: ${data.message}`);
      setResult(data);
      setLastSeen(data.last_seen || null);
    } catch (err) {
      setMessage(`Error: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  // Prefer server's frame_url; fallback to /api/frame_image?path=...
  const frameImageUrl = lastSeen?.frame_url
    ? `http://localhost:5000${lastSeen.frame_url}`
    : lastSeen?.frame_image
      ? `http://localhost:5000/api/frame_image?path=${encodeURIComponent(
          lastSeen.frame_image
        )}`
      : null;

  return (
    <div className="main-page">
      <div className="card" style={{ maxWidth: 900 }}>
        <h2>➕ Add Person</h2>

        <form
          onSubmit={handleSubmit}
          style={{ display: "flex", flexDirection: "column", gap: 12 }}
        >
          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
            <div>
              <label style={{ fontWeight: 600 }}>First name</label>
              <input
                type="text"
                placeholder="First name"
                value={firstName}
                onChange={(e) => setFirstName(e.target.value)}
                required
              />
            </div>
            <div>
              <label style={{ fontWeight: 600 }}>Last name</label>
              <input
                type="text"
                placeholder="Last name"
                value={lastName}
                onChange={(e) => setLastName(e.target.value)}
                required
              />
            </div>
          </div>

          <div>
            <label style={{ fontWeight: 600 }}>Age (optional)</label>
            <input
              type="number"
              min="0"
              placeholder="Age"
              value={age}
              onChange={(e) => setAge(e.target.value)}
            />
          </div>

          <div>
            <label style={{ fontWeight: 600 }}>Images (at least one)</label>
            <input
              type="file"
              accept="image/*"
              multiple
              onChange={(e) => setImages(Array.from(e.target.files || []))}
            />
          </div>

          <div style={{ display: "flex", gap: 16, alignItems: "center" }}>
            <button type="submit" className="btn" disabled={saving}>
              {saving ? "Saving..." : "Save Person"}
            </button>
            <Link to="/">
              <button type="button" className="btn secondary">⬅ Back</button>
            </Link>
          </div>
        </form>

        {message && <p style={{ marginTop: 10 }}>{message}</p>}

        {(previewUrl || lastSeen) && (
          <div
            style={{
              marginTop: 18,
              display: "grid",
              gridTemplateColumns: "1fr 1fr",
              gap: 16,
              alignItems: "start",
            }}
          >
            {/* Left: uploaded reference preview */}
            <div className="card" style={{ padding: 12 }}>
              <h3 style={{ marginTop: 0 }}>Reference (uploaded)</h3>
              {previewUrl ? (
                <img
                  src={previewUrl}
                  alt="reference"
                  style={{ width: "100%", borderRadius: 8, objectFit: "cover" }}
                  onLoad={() => URL.revokeObjectURL(previewUrl)}
                />
              ) : (
                <div>No preview</div>
              )}
              <div style={{ marginTop: 8 }}>
                <div><strong>Name:</strong> {firstName} {lastName}</div>
                {age && <div><strong>Age:</strong> {age}</div>}
              </div>
            </div>

            {/* Right: best match from stored frames */}
            <div className="card" style={{ padding: 12 }}>
              <h3 style={{ marginTop: 0 }}>Best match (from videos)</h3>
              {lastSeen ? (
                <>
                  {frameImageUrl ? (
                    <img
                      src={frameImageUrl}
                      alt="matched frame"
                      style={{ width: "100%", borderRadius: 8, objectFit: "cover" }}
                      onError={(e) => { e.currentTarget.style.display = "none"; }}
                    />
                  ) : (
                    <div style={{ opacity: 0.8 }}>
                      Frame image URL not available.
                    </div>
                  )}
                  <div style={{ marginTop: 8 }}>
                    <div><strong>Score:</strong> {lastSeen.score?.toFixed?.(3) ?? lastSeen.score}</div>
                    <div><strong>Place:</strong> {lastSeen.place || "Unknown"}</div>
                    <div><strong>Time:</strong> {lastSeen.time || lastSeen.time_iso || "Unknown"}</div>
                    {lastSeen.video && (
                      <div><strong>Video:</strong> {lastSeen.video} (frame #{lastSeen.frame_idx})</div>
                    )}
                    {lastSeen.label && <div style={{ marginTop: 6 }}>{lastSeen.label}</div>}
                  </div>
                </>
              ) : (
                <div>No matches found in processed videos.</div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AddPerson;
