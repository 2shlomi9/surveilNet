import { useState } from "react";
import { Link } from "react-router-dom";
import "./AddPerson.css"; // Import the new CSS file

function AddPerson() {
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [age, setAge] = useState("");
  const [images, setImages] = useState([]);
  const [message, setMessage] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!firstName || !lastName) {
      setMessage("First name and last name are required");
      return;
    }
    setMessage("Adding person...");

    const formData = new FormData();
    formData.append("first_name", firstName);
    formData.append("last_name", lastName);
    if (age) formData.append("age", age);
    images.forEach((image) => formData.append("images", image));

    try {
      const res = await fetch("http://localhost:5000/api/people", {
        method: "POST",
        body: formData,
      });
      const data = await res.json();
      if (res.ok) {
        setMessage(`Success: ${data.message} (ID: ${data.id})`);
        setFirstName("");
        setLastName("");
        setAge("");
        setImages([]);
      } else {
        setMessage(`Error: ${data.error}`);
      }
    } catch (err) {
      setMessage(`Error: ${err.message}`);
    }
  };

  return (
    <div className="main-page">
      <div className="card">
        <h2>➕ Add Person</h2>
        <form onSubmit={handleSubmit}>
          <input
            type="text"
            placeholder="First Name"
            value={firstName}
            onChange={(e) => setFirstName(e.target.value)}
            required
          />
          <input
            type="text"
            placeholder="Last Name"
            value={lastName}
            onChange={(e) => setLastName(e.target.value)}
            required
          />
          <input
            type="number"
            placeholder="Age (optional)"
            value={age}
            onChange={(e) => setAge(e.target.value)}
          />
          <input
            type="file"
            multiple
            accept="image/*"
            onChange={(e) => setImages([...e.target.files])}
          />
          <button type="submit" className="btn">Save Person</button>
        </form>
        {message && <p>{message}</p>}
        <Link to="/">
          <button className="btn secondary">⬅ Back</button>
        </Link>
      </div>
    </div>
  );
}

export default AddPerson;