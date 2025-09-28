import React, { useEffect, useState } from "react";
import "./App.css";

function App() {
  const [people, setPeople] = useState([]);
  const [videos, setVideos] = useState([]);
  const [matches, setMatches] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState("");
  const [processing, setProcessing] = useState(false);
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [age, setAge] = useState("");
  const [images, setImages] = useState([]);
  const [addPersonMessage, setAddPersonMessage] = useState("");

  // Load people
  useEffect(() => {
    fetch("http://localhost:5000/api/people")
      .then(res => res.json())
      .then(data => setPeople(data.people || []))
      .catch(err => {
        console.error("Error fetching people:", err);
        setAddPersonMessage("Error fetching people");
      });
  }, []);

  // Load videos
  useEffect(() => {
    fetch("http://localhost:5000/api/videos")
      .then(res => res.json())
      .then(data => setVideos(data.videos || []))
      .catch(err => {
        console.error("Error fetching videos:", err);
        setAddPersonMessage("Error fetching videos");
      });
  }, []);

  // Load matches
  const fetchMatches = () => {
    fetch("http://localhost:5000/api/matches")
      .then(res => res.json())
      .then(data => setMatches(data.matches || []))
      .catch(err => {
        console.error("Error fetching matches:", err);
        setAddPersonMessage("Error fetching matches");
      });
  };

  // Initial fetch for matches
  useEffect(() => {
    fetchMatches();
  }, []);

  // Add person
  const addPerson = async (e) => {
    e.preventDefault();
    if (!firstName || !lastName) {
      setAddPersonMessage("First name and last name are required");
      return;
    }
    setAddPersonMessage("Adding person...");

    const formData = new FormData();
    formData.append("first_name", firstName);
    formData.append("last_name", lastName);
    if (age) formData.append("age", age);
    images.forEach(image => formData.append("images", image));

    try {
      const res = await fetch("http://localhost:5000/api/people", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setAddPersonMessage(`Success: ${data.message} (ID: ${data.id})`);
        setFirstName("");
        setLastName("");
        setAge("");
        setImages([]);
        fetch("http://localhost:5000/api/people")
          .then(res => res.json())
          .then(data => setPeople(data.people || []));
      } else {
        setAddPersonMessage(`Error: ${data.error}`);
      }
    } catch (err) {
      setAddPersonMessage(`Error: ${err.message}`);
    }
  };

  // Process video
  const processVideo = async () => {
    if (!selectedVideo) {
      setAddPersonMessage("Please select a video");
      return;
    }
    setProcessing(true);
    setAddPersonMessage("Processing video...");
    try {
      const res = await fetch("http://localhost:5000/api/process_video", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ filename: selectedVideo }),
      });
      const data = await res.json();
      if (res.ok) {
        setAddPersonMessage(`Success: ${data.message} (${data.video})`);
        fetchMatches();
      } else {
        setAddPersonMessage(`Error: ${data.error}`);
      }
    } catch (err) {
      console.error("Error processing video:", err);
      setAddPersonMessage("Error processing video");
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="App">
      <h1 className="text-3xl font-bold mb-4">Face Recognition Dashboard</h1>

      <section className="mb-8 p-4 bg-white rounded shadow">
        <h2 className="text-xl font-semibold mb-2">Add Person</h2>
        <form onSubmit={addPerson} className="space-y-4">
          <div>
            <label className="block">First Name:</label>
            <input
              type="text"
              value={firstName}
              onChange={e => setFirstName(e.target.value)}
              className="border p-2 w-full"
              required
            />
          </div>
          <div>
            <label className="block">Last Name:</label>
            <input
              type="text"
              value={lastName}
              onChange={e => setLastName(e.target.value)}
              className="border p-2 w-full"
              required
            />
          </div>
          <div>
            <label className="block">Age (Optional):</label>
            <input
              type="number"
              value={age}
              onChange={e => setAge(e.target.value)}
              className="border p-2 w-full"
            />
          </div>
          <div>
            <label className="block">Images:</label>
            <input
              type="file"
              multiple
              accept=".jpg,.png"
              onChange={e => setImages(Array.from(e.target.files))}
              className="border p-2 w-full"
            />
          </div>
          <button type="submit" className="bg-blue-500 text-white p-2 rounded">
            Add Person
          </button>
        </form>
        <p className="mt-2">{addPersonMessage}</p>
      </section>

      <section className="mb-8 p-4 bg-white rounded shadow">
        <h2 className="text-xl font-semibold mb-2">People</h2>
        <ul className="list-disc pl-5">
          {people.map(p => (
            <li key={p.id}>
              {p.first_name} {p.last_name} ({p.age || "N/A"})
            </li>
          ))}
        </ul>
      </section>

      <section className="mb-8 p-4 bg-white rounded shadow">
        <h2 className="text-xl font-semibold mb-2">Videos</h2>
        <select
          value={selectedVideo}
          onChange={e => setSelectedVideo(e.target.value)}
          className="border p-2 w-full mb-2"
        >
          <option value="">Select video</option>
          {videos.map(v => (
            <option key={v} value={v}>{v}</option>
          ))}
        </select>
        <button
          onClick={processVideo}
          disabled={!selectedVideo || processing}
          className={`p-2 rounded ${!selectedVideo || processing ? 'bg-gray-400' : 'bg-blue-500 text-white'}`}
        >
          {processing ? "Processing..." : "Process Video"}
        </button>
      </section>

      <section className="mb-8 p-4 bg-white rounded shadow">
        <h2 className="text-xl font-semibold mb-2">Matches</h2>
        <div className="flex flex-wrap">
          {matches.map(match => (
            <div key={match.filename} className="m-2 text-center">
              <img
                src={`http://localhost:5000/matches/${match.filename}`}
                alt={match.full_name || "Unknown"}
                className="w-32 h-auto border"
              />
              <p className="mt-1">{match.full_name || "Unknown"}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}

export default App;