import React, { useRef, useState } from "react";
import WebcamBox from "../components/WebcamBox";
import PredictionResult from "../components/PredictionResult";
import HomeButton from "../components/HomeButton";

/**
 * PredictScreen
 * The main prediction interface with camera, button, result, and home navigation.
 */
export default function PredictScreen() {
  const webcamRef = useRef(null);
  const [age, setAge] = useState(null);
  const [loading, setLoading] = useState(false);

  const handlePredict = async () => {
    if (!webcamRef.current) return;
    const screenshot = webcamRef.current.getScreenshot();
    setLoading(true);

    try {
      const res = await fetch("http://localhost:8000/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: screenshot.split(",")[1] }),
      });

      const data = await res.json();
      setAge(data.predicted_age);
    } catch (err) {
      console.error("Prediction error:", err);
    }

    setLoading(false);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-600 via-blue-600 to-teal-400 flex flex-col items-center px-6 pt-12 text-white text-center">
      <h1 className="text-5xl md:text-6xl font-bold mb-2">Let’s predict your age!</h1>
      <p className="text-xl md:text-2xl text-white/90 mb-8">Please allow camera access below.</p>

      <WebcamBox webcamRef={webcamRef} loading={loading} handlePredict={handlePredict} />
      <PredictionResult age={age} />
      <div className="mb-12">
        <HomeButton />
      </div>
    </div>
  );
}
