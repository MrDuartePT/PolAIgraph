import { motion, useInView } from "framer-motion";
import React, { useState, useRef, use } from 'react';
import { useNavigate } from 'react-router-dom';
import "./style.css";
import logo from "../../assets/images/polaigraph_logo_white.png";
import placeholder from "../../assets/images/avatar_placeholder.png";
import jaime from "../../assets/images/jaime.jpg";
import goncalo from "../../assets/images/goncalo.jpg";
import joana from "../../assets/images/joana.jpg";
import background_video from "../../assets/videos/background_video.mp4";


const HomePage = () => {
  const [file, setFile] = useState(null);
  const [fileName, setFileName] = useState('No file selected');
  const navigate = useNavigate();

  const handleFileSelection = () => {
    const fileInput = document.createElement('input');
    fileInput.type = 'file';
    fileInput.accept = 'video/mp4';

    fileInput.onchange = (event) => {
      const selectedFile = event.target.files[0];
      if (selectedFile && selectedFile.type === 'video/mp4') {
        setFile(selectedFile);
        setFileName(selectedFile.name);
      } else {
        alert("Please select a valid MP4 video file.");
        setFile(null);
        setFileName('No file selected');
      }
    };

    fileInput.click();
  };

  const handleUpload = () => {
    if (!file) {
      alert("Please select a video file.");
      return;
    }
    const videoUrl = URL.createObjectURL(file);
    navigate('/video-player', { state: { videoUrl } });
  };

  // Refs and Visibility Tracking
  const numbersRef = useRef(null);
  const teamRef = useRef(null);

  const numbersInView = useInView(numbersRef, { triggerOnce: true, margin: "-100px 0px" });
  const teamInView = useInView(teamRef, { triggerOnce: true, margin: "-100px 0px" });

  return (
    <div className="home-page-container">

    <div className="background-home-video">
        <div className="video-player-container">
          <video className='video-player' autoPlay loop muted>
            <source src= {background_video} type="video/mp4" />
          </video>
        </div>
      </div>

      <div className="overlay"></div>

      {/* Header Section */}
      <motion.div
        className="header"
        initial={{ opacity: 0, y: -50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        <img src={logo} alt="Logo" className="logo" />
        <p className="intro-text">
        Our mission is to empower viewers with the ability to identify and challenge misinformation in real time during TV debates. By leveraging advanced AI technology, we have developed a tool that automatically analyzes statements made during debates, cross-referencing them with reliable sources to provide instant fact-checking. Our goal is to promote transparency, foster informed discussions, and combat the spread of fake news, ensuring that the public can make well-informed decisions based on accurate information.
        </p>
      </motion.div>

      {/* Numbers Section */}
      <motion.div
        ref={numbersRef}
        className="numbers-section-container"
        initial={{ opacity: 0, y: 50 }}
        animate={numbersInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 1.5 }}
      >
        <h2 className="text-header numbers">The World we live in</h2>
        <div className="percentages-row">
          {[ 
            { num: "94%", text: "Of journalists see made-up news as a significant problem.", delay: 0 },
            { num: "38.2%", text: "Of U.S. news consumers unknowingly shared fake news on social media", delay: 0.5 },
            { num: "500000", text: "Estimated deepfakes shared on social media in 2023", delay: 1 },
          ].map((item, index) => (
            <motion.div
              key={index}
              className="percentage"
              initial={{ opacity: 0, x: index === 0 ? -100 : index === 2 ? 100 : 0 }}
              animate={numbersInView ? { opacity: 1, x: 0 } : {}}
              transition={{ duration: 1, delay: item.delay }}
            >
              <span className="percentage-number">{item.num}</span>
              <p className="percentage-text">{item.text}</p>
            </motion.div>
          ))}
        </div>
      </motion.div>

      {/* Team Section */}
      <motion.div
        ref={teamRef}
        className="team-section-container"
        initial={{ opacity: 0, y: 50 }}
        animate={teamInView ? { opacity: 1, y: 0 } : {}}
        transition={{ duration: 2 }}
      >
        <h2 className="text-header team">Meet Our Team</h2>
        <div className="team-row">
          {[
            { name: "Jaime Ferreira", position: "Developer", img: jaime, link: "https://www.linkedin.com/in/johndoe", delay: 0.5 },
            { name: "Gonçalo Duarte", position: "Developer", img: goncalo, link: "https://www.linkedin.com/in/janesmith", delay: 1 },
            { name: "Joana Santos", position: "Developer", img: joana, link: "https://www.linkedin.com/in/mikejohnson", delay: 1.5 },
            { name: "Ana", position: "Developer", img: placeholder, link: "https://www.linkedin.com/in/mikejohnson", delay: 2 },

          ].map((member, index) => (
            <motion.div
              key={index}
              className="team-member"
              initial={{ opacity: 0, y: 50 }}
              animate={teamInView ? { opacity: 1, y: 0 } : {}}
              transition={{ duration: 1, delay: member.delay }}
            >
              <img src={member.img} alt={`Team Member ${index + 1}`} className="team-member-img" />
              <p className="team-member-name">{member.name}</p>
              <p className="team-member-position">{member.position}</p>
              <a href={member.link} className="team-member-link">LinkedIn</a>
            </motion.div>
          ))}
        </div>
      </motion.div>
      
      <motion.button
        className="floating-button"
        onClick={navigate.bind(null, '/video-player')}
        initial={{ opacity: 0, y: 50 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 1 }}
      >
        Try It Now
      </motion.button>
    </div>
  );
};

export default HomePage;
