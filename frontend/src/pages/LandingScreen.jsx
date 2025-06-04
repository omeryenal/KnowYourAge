import React from "react";
import { useNavigate } from "react-router-dom";
import { motion } from "framer-motion";

/**
 * LandingScreen
 * The homepage with title, subtitle, start button, and social icons.
 */
export default function LandingScreen() {
  const navigate = useNavigate();

  return (
    <motion.div
      className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-teal-400 flex flex-col items-center justify-center text-center text-white px-6 pt-24"
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, ease: "easeOut" }}
    >
      {/* Title */}
      <h1 className="font-quicksand text-5xl md:text-6xl lg:text-7xl font-bold mb-4 drop-shadow-xl">
        Know Your Age
      </h1>

      {/* Subtitle */}
      <p className="text-2xl md:text-3xl text-white/90 mb-14 font-light">
        AI-powered age prediction from your face
      </p>

      {/* Start Button */}
      <button
        onClick={() => navigate("/predict")}
        className="relative inline-block bg-gradient-to-r from-indigo-500 to-blue-500 text-white font-bold text-2xl py-5 px-20 rounded-3xl overflow-hidden shadow-2xl hover:ring-4 hover:ring-indigo-300 transition-all duration-300 ease-out mb-16"
      >
        <span className="relative z-10">Start</span>
        <span className="absolute inset-0 bg-white/10 backdrop-blur opacity-0 hover:opacity-100 transition-opacity duration-300" />
      </button>

      {/* Social Icons */}
      <div className="flex flex-col items-center">
        <div className="flex space-x-12 mb-6">
          {/* Gmail */}
          <a
            href="mailto:omer.yenal1@gmail.com"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="Gmail"
            className="text-white/80 hover:text-white transition-colors"
          >
            <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 24 24">
              <path d="M20 4H4a2 2 0 00-2 2v12a2 2 0 002 2h16a2 2 0 002-2V6a2 2 0 00-2-2zm0 2v.01L12 13 4 6.01V6h16zM4 18v-9.99l7.446 6.623a1 1 0 001.108 0L20 8.01V18H4z" />
            </svg>
          </a>

          {/* LinkedIn */}
          <a
            href="https://linkedin.com/in/omer-yenal"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="LinkedIn"
            className="text-white/80 hover:text-white transition-colors"
          >
            <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 24 24">
              <path d="M4.98 3.5C4.98 5.43 3.42 7 1.5 7S-1 5.43-1 3.5 0.56 0 2.5 0s2.48 1.57 2.48 3.5zM0 8h5v13H0V8zm7 0h5v2h.07c.69-1.3 2.38-2.67 4.93-2.67 5.27 0 6.25 3.47 6.25 7.97V21h-5v-6.6c0-1.57-.03-3.6-2.2-3.6-2.2 0-2.53 1.72-2.53 3.5V21h-5V8z" />
            </svg>
          </a>

          {/* GitHub */}
          <a
            href="https://github.com/omeryenal/KnowYourAge"
            target="_blank"
            rel="noopener noreferrer"
            aria-label="GitHub"
            className="text-white/80 hover:text-white transition-colors"
          >
            <svg className="w-12 h-12" fill="currentColor" viewBox="0 0 24 24">
              <path d="M12 0C5.37 0 0 5.373 0 12c0 5.302 3.438 9.8 8.205 11.387.6.112.82-.258.82-.577 
              0-.285-.01-1.04-.015-2.04-3.338.725-4.042-1.61-4.042-1.61
              -.546-1.385-1.333-1.753-1.333-1.753
              -1.088-.745.083-.729.083-.729 
              1.205.084 1.84 1.237 1.84 1.237 
              1.07 1.835 2.807 1.305 3.492.997
              .108-.775.42-1.305.763-1.605
              -2.665-.3-5.467-1.334-5.467-5.932 
              0-1.31.467-2.382 1.235-3.22
              -.123-.303-.535-1.523.117-3.176 
              0 0 1.008-.322 3.3 1.23a11.513 
              11.513 0 013.003-.404c1.018.005 
              2.045.138 3.003.404 
              2.29-1.552 3.296-1.23 3.296-1.23
              .654 1.653.242 2.873.12 3.176
              .77.838 1.232 1.91 1.232 3.22 
              0 4.61-2.807 5.628-5.48 5.922
              .43.37.814 1.102.814 2.222 
              0 1.606-.015 2.896-.015 3.286 
              0 .321.216.694.825.576C20.565 
              21.796 24 17.298 24 12
              c0-6.627-5.373-12-12-12z" />
            </svg>
          </a>
        </div>

        <p className="text-lg text-white/80">By Ömer Yenal</p>
      </div>
    </motion.div>
  );
}
