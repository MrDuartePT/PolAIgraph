// VideoPlayer.js
import React from 'react';
import { useLocation, useNavigate } from 'react-router-dom';

function VideoPlayer() {
  const location = useLocation();
  const navigate = useNavigate();
  const { videoUrl } = location.state || {};

  // If no video URL is provided, show a message or redirect back.
  if (!videoUrl) {
    return (
      <div>
        <p>No video to play. Please upload a video first.</p>
        <button onClick={() => navigate('/')}>Go Back</button>
      </div>
    );
  }

  return (
    <div>
      <h2>Video Player</h2>
      <video src={videoUrl} controls width="600" />
    </div>
  );
}

export default VideoPlayer;
