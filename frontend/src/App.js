import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import LandingScreen from "./pages/LandingScreen";
import PredictScreen from "./pages/PredictScreen";

/**
 * App
 * Root component with routing between landing and prediction screens.
 */
function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingScreen />} />
        <Route path="/predict" element={<PredictScreen />} />
      </Routes>
    </Router>
  );
}

export default App;
