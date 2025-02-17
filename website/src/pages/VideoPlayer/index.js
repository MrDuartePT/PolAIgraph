import React from "react";
import { useLocation } from "react-router-dom";
import "./style.css";

export default function VideoPlayer() {
  const location = useLocation();
  const videoUrl = location.state?.videoUrl;

  return (
    <div className="video-player-container">
      <h1 className="video-title">Uploaded Debate Video</h1>
      {videoUrl ? (
        <video className="video-element" controls>
          <source src={videoUrl} type="video/mp4" />
          Your browser does not support the video tag.
        </video>
      ) : (
        <p className="video-error">No video uploaded. Please upload a video first.</p>
      )}
    </div>
  );
}
