import React, { useMemo, useEffect, useState } from 'react';
import Particles from 'react-tsparticles'; // Import the Particles component from react-tsparticles
import { loadFull } from 'tsparticles'; // Import loadFull function to load all particles features

const ParticlesComponent = (props) => {
  const [init, setInit] = useState(false);

  // This effect runs only once to initialize particles
  useEffect(() => {
    loadFull().then(() => {
      setInit(true); // Set the init state to true after particles are loaded
    });
  }, []);

  // Particle configuration, memoized to avoid unnecessary re-renders
  const options = useMemo(() => ({
    particles: {
      number: {
        value: 80,
        density: {
          enable: true,
          value_area: 800,
        },
      },
      color: {
        value: "#6e6e6e",
      },
      shape: {
        type: "circle",
        stroke: {
          width: 0,
          color: "#000000",
        },
        polygon: {
          nb_sides: 5,
        },
      },
      opacity: {
        value: 0.662884018971109,
        random: false,
        anim: {
          enable: false,
          speed: 1,
          opacity_min: 0.1,
          sync: false,
        },
      },
      size: {
        value: 4.008530152163807,
        random: true,
        anim: {
          enable: false,
          speed: 40,
          size_min: 0.1,
          sync: false,
        },
      },
      move: {
        enable: true,
        speed: 3.1565905665290903,
        direction: "none",
        random: false,
        straight: false,
        out_mode: "out",
        bounce: false,
      },
    },
    interactivity: {
      detect_on: "canvas",
      events: {
        onhover: {
          enable: true,
          mode: "repulse",
        },
        onclick: {
          enable: true,
          mode: "push",
        },
        resize: true,
      },
      modes: {
        grab: {
          distance: 400,
          line_linked: {
            opacity: 1,
          },
        },
        bubble: {
          distance: 400,
          size: 40,
          duration: 2,
          opacity: 8,
          speed: 3,
        },
        repulse: {
          distance: 200,
          duration: 0.4,
        },
        push: {
          particles_nb: 4,
        },
        remove: {
          particles_nb: 2,
        },
      },
    },
    retina_detect: true,
  }), []);

  return <Particles id={props.id} options={options} />; 
};

export default ParticlesComponent;
