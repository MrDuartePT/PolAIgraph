import React from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import HomePage from "./pages/HomePage";
import VideoPlayer from "./pages/VideoPlayer"; // Placeholder for the upload page
import "./App.css";

export default function App() {
  return (
    <Router>
      <div className="app-container">
        <Routes>
          <Route path="/" element={<HomePage />} />
          <Route path="/video-player" element={<VideoPlayer />} />
        </Routes>
      </div>
    </Router>
  );
}
