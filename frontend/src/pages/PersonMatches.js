import { useState, useEffect } from "react";
import { useParams, Link } from "react-router-dom";

function PersonMatches() {
  const { personId } = useParams();
  const [matches, setMatches] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchPersonMatches = async () => {
      try {
        const response = await fetch(`http://localhost:5000/api/matches?person_id=${personId}`);
        if (!response.ok) {
          throw new Error("Failed to fetch person matches");
        }
        const data = await response.json();
        setMatches(data.matches || []);
      } catch (err) {
        setError(err.message);
      } finally {
        setLoading(false);
      }
    };

    fetchPersonMatches();
  }, [personId]);

  const parseMatchInfo = (filename) => {
    const parts = filename.replace('.jpg', '').split('_');
    if (parts.length < 4) return null;
    const frameNum = parts[0].replace('frame', '');
    const personId = parts[1].replace('id', '');
    const firstName = parts[2];
    const lastName = parts[3];
    return { frameNum, personId, firstName, lastName, filename };
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

  const handleDelete = async (filename) => {
    try {
      const response = await fetch(`http://localhost:5000/api/matches/${filename}`, {
        method: "DELETE",
      });
      const data = await response.json();
      if (!response.ok) throw new Error(data.error || "Failed to delete match");
      setMatches((prevMatches) => prevMatches.filter((match) => match !== filename));
      alert(`Match ${filename} deleted successfully`);
    } catch (err) {
      alert(`Delete error: ${err.message}`);
    }
  };

  if (loading) return <div className="main-page"><div className="card"><p>Loading person matches...</p></div></div>;
  if (error) return <div className="main-page"><div className="card"><p>Error: {error}</p></div></div>;

  const parsedMatches = matches.filter((filename) => !filename.startsWith('reference')).map(parseMatchInfo).filter(Boolean);

  return (
    <div className="main-page">
      <div className="card">
        <h2>Matches for Person ID: {personId}</h2>
        {parsedMatches.length === 0 ? (
          <p>No matches found for this person.</p>
        ) : (
          <div className="match-grid">
            {parsedMatches.map((match) => (
              <div key={match.filename} className="match-card">
                <img
                  src={`http://localhost:5000/matches/${match.filename}`}
                  alt={`Frame ${match.frameNum}`}
                  className="match-photo"
                />
                <div className="match-info">
                  <p>Frame: {match.frameNum}</p>
                </div>
                <button onClick={() => handleDownload(match.filename)} className="btn">⬇ Download</button>
                <button onClick={() => handleDelete(match.filename)} className="btn delete-btn">🗑️ Delete</button>
              </div>
            ))}
          </div>
        )}
        <Link to="/matches">
          <button className="btn secondary">⬅ Back to Matches</button>
        </Link>
      </div>
    </div>
  );
}

export default PersonMatches;