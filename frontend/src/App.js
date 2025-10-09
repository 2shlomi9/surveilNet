import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MainPage from "./pages/MainPage";
import AddPerson from "./pages/AddPerson";
import UploadVideo from "./pages/UploadVideo";
import MatchPage from "./pages/MatchPage";
import AboutUs from "./pages/AboutUs";

export default function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<MainPage />} />
        <Route path="/add-person" element={<AddPerson />} />
        <Route path="/upload-video" element={<UploadVideo />} />
        <Route path="/matches" element={<MatchPage />} />
        <Route path="/about-us" element={<AboutUs />} />
      </Routes>
    </Router>
  );
}