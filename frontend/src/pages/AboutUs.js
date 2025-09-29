import { Link } from "react-router-dom";
import "./AboutUs.css";

export default function AboutUs() {
  return (
    <div className="about-page">
      <video autoPlay loop muted playsInline className="about-bg">
        <source src="/bg.mp4" type="video/mp4" />
      </video>

      <nav className="about-nav">
        <Link to="/" className="about-home">← Back to Home</Link>
      </nav>

      <header className="about-hero">
        <h1>About Us</h1>
        <p>
          SurveilNet brings modern computer vision and AI together to deliver
          accurate, fast recognition on live and recorded video sources.
        </p>
      </header>

      <section className="about-grid">
        <article className="about-card">
          <h3>🚀 Real-time 3D Detection</h3>
          <p>
            Process live 3D video streams instantly using advanced AI-driven
            computer vision and image processing.
          </p>
        </article>

        <article className="about-card">
          <h3>🧠 Smart Recognition</h3>
          <p>
            Accurately identify individuals from stored images and detect
            unfamiliar faces with a privacy-first library.
          </p>
        </article>

        <article className="about-card">
          <h3>⚡ Instant Alerts</h3>
          <p>
            Receive real-time alerts when a match is found so you can act fast.
          </p>
        </article>
      </section>
    </div>
  );
}
