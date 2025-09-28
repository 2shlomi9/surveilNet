import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainPage from "./pages/MainPage";
import AddPerson from "./pages/AddPerson";
import UploadVideo from "./pages/UploadVideo";
import AboutUs from "./pages/AboutUs"; // ← NEW

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/add-person" element={<AddPerson />} />
        <Route path="/upload-video" element={<UploadVideo />} />
        <Route path="/about-us" element={<AboutUs />} /> {/* ← NEW */}
      </Routes>
    </Router>
  );
}
