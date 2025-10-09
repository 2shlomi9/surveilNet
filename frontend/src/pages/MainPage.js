import { Link } from "react-router-dom";
import "./MainPage.css";

export default function MainPage() {
  return (
    <div className="main-page">
      {/* Background video */}
      <video autoPlay loop muted playsInline className="video-background">
        <source src="/bg.mp4" type="video/mp4" />
      </video>

      {/* Top bar: logo only */}
      <nav className="navbar">
        <img src="/logo2.png" alt="SurveilNet logo" className="nav-logo" />
      </nav>

      {/* Centered hero only */}
      <section className="hero">
        <div className="hero-card">
          <h1 className="hero-title">SurveilNet — AI Video & Face Recognition</h1>
          <p className="hero-subtitle">
            Advanced AI that processes 3D video streams using cutting-edge CV and image processing.
          </p>

          <div className="hero-buttons">
            <Link to="/add-person" className="btn">➕ Add Person</Link>
            <Link to="/upload-video" className="btn">🎥 Upload Video</Link>
            <Link to="/matches" className="btn">👁️ View Matches</Link>
          </div>

          <div className="hero-more">
            <Link to="/about-us" className="more-link">About Us →</Link>
          </div>
        </div>
      </section>
    </div>
  );
}