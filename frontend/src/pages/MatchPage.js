import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import "./MatchPage.css";

function MatchPage() {
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchMatches = async () => {
      try {
        const response = await fetch("http://localhost:5000/api/matches");
        if (!response.ok) {
          throw new Error("Failed to fetch matches");
        }
        const data = await response.json();
        console.log("[DEBUG] Matches received from /api/matches:", data.matches);
        setMatches(data.matches || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchMatches();
  }, []);

  const parseMatchInfo = (filename) => {
    const parts = filename.replace('.jpg', '').split('_');
    if (parts.length < 4) return null;
    const isReference = parts[0] === 'reference';
    const frameNum = isReference ? null : parts[0].replace('frame', '');
    const personId = parts[1].replace('id', '');
    const firstName = parts[2];
    const lastName = parts[3];
    return { frameNum, personId, firstName, lastName, filename, isReference };
  };

  const handleDownload = async (filename) => {
    try {
      const response = await fetch(`http://localhost:5000/matches/${filename}`);
      if (!response.ok) throw new Error("Download failed");
      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = filename;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (err) {
      alert(`Download error: ${err.message}`);
    }
  };

  const handleDelete = async (filename, personId) => {
    try {
      const response = await fetch(`http://localhost:5000/api/matches/${filename}`, {
        method: "DELETE",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to delete match");
      setMatches((prevMatches) => {
        const newMatches = prevMatches.filter((match) => match !== filename);
        console.log("[DEBUG] Updated matches after delete:", newMatches);
        return newMatches;
      });
      alert(`Match ${filename} deleted successfully`);
    } catch (err) {
      alert(`Delete error: ${err.message}`);
    }
  };

  const handleDeleteAll = async () => {
    if (!window.confirm("Are you sure you want to delete all matches?")) {
      return;
    }
    try {
      const response = await fetch("http://localhost:5000/api/matches", {
        method: "DELETE",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to delete all matches");
      setMatches([]);
      alert(data.message || "All matches deleted successfully");
    } catch (err) {
      alert(`Delete all error: ${err.message}`);
    }
  };

  if (loading) return <div className="main-page"><div className="card"><p>Loading matches...</p></div></div>;
  if (error) return <div className="main-page"><div className="card"><p>Error: {error}</p></div></div>;

  const parsedMatches = matches.map(parseMatchInfo).filter(Boolean);
  console.log("[DEBUG] Parsed matches:", parsedMatches);
  const matchGroups = {};
  parsedMatches.forEach((match) => {
    if (!match.isReference) {
      const key = match.personId;
      if (!matchGroups[key]) {
        matchGroups[key] = { match: null, reference: null };
      }
      matchGroups[key].match = match;
    } else {
      const key = match.personId;
      if (!matchGroups[key]) {
        matchGroups[key] = { match: null, reference: null };
      }
      matchGroups[key].reference = match;
    }
  });
  console.log("[DEBUG] Match groups:", matchGroups);

  return (
    <div className="main-page">
      <div className="card">
        <h2>👁️ Found Matches</h2>
        {Object.keys(matchGroups).length === 0 ? (
          <p>No matches found yet. Upload a video and process it!</p>
        ) : (
          <div className="match-container">
            <button
              onClick={handleDeleteAll}
              className="btn delete-btn"
              style={{ marginBottom: "1rem" }}
            >
              🗑️ Delete All Matches
            </button>
            <div className="match-grid">
              {Object.values(matchGroups).map((group) => (
                group.match && (
                  <div key={group.match.personId} className="match-card">
                    <div className="image-pair">
                      <div className="image-container">
                        <h4>Match Image</h4>
                        <img
                          src={`http://localhost:5000/matches/${group.match.filename}`}
                          alt={`${group.match.firstName} ${group.match.lastName} - Frame ${group.match.frameNum}`}
                          className="match-photo"
                        />
                      </div>
                      <div className="image-container">
                        <h4>Reference Image</h4>
                        {group.reference ? (
                          <img
                            src={`http://localhost:5000/matches/${group.reference.filename}`}
                            alt={`${group.match.firstName} ${group.match.lastName} - Reference`}
                            className="match-photo"
                          />
                        ) : (
                          <p>No reference image available</p>
                        )}
                      </div>
                    </div>
                    <div className="match-info">
                      <h3>{group.match.firstName} {group.match.lastName}</h3>
                      <p>ID: {group.match.personId}</p>
                      <p>Frame: {group.match.frameNum}</p>
                    </div>
                    <button onClick={() => handleDownload(group.match.filename)} className="btn">⬇ Download Match</button>
                    {group.reference && (
                      <button onClick={() => handleDownload(group.reference.filename)} className="btn secondary">⬇ Download Reference</button>
                    )}
                    <button
                      onClick={() => handleDelete(group.match.filename, group.match.personId)}
                      className="btn delete-btn"
                    >
                      🗑️ Delete Match
                    </button>
                  </div>
                )
              ))}
            </div>
          </div>
        )}
        <Link to="/">
          <button className="btn secondary">⬅ Back to Home</button>
        </Link>
      </div>
    </div>
  );
}

export default MatchPage;