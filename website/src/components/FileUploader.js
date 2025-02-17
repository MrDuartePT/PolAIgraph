// FileUploader.js
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';

function FileUploader() {
  const [file, setFile] = useState(null);
  const navigate = useNavigate();

  // Handle file selection and validate that it's an MP4 file
  const handleFileChange = (event) => {
    const selectedFile = event.target.files[0];
    if (selectedFile && selectedFile.type === 'video/mp4') {
      setFile(selectedFile);
    } else {
      alert("Please select a valid MP4 video file.");
      setFile(null);
    }
  };

  // Simulate an upload process and navigate to the video player page
  const handleUpload = () => {
    if (!file) {
      alert("Please select a video file first.");
      return;
    }

    // Here you might perform an API call to upload the file.
    // For this example, we simulate this by creating an object URL.
    const videoUrl = URL.createObjectURL(file);

    // Navigate to the video player page, passing the video URL in the router state.
    navigate('/video-player', { state: { videoUrl } });
  };

  return (
    <div>
      <h2>Upload an MP4 Video</h2>
      <input type="file" accept="video/mp4" onChange={handleFileChange} />
      <button onClick={handleUpload}>Upload &amp; Play</button>
    </div>
  );
}

export default FileUploader;
