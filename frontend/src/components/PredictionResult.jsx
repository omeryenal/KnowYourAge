import React from "react";

/**
 * PredictionResult
 * Displays the predicted age once it is received from the backend.
 */
export default function PredictionResult({ age }) {
  // If no prediction yet, don't render anything
  if (!age) return null;

  return (
    <p className="text-3xl md:text-4xl font-semibold mb-6">
      You look like you’re <span className="font-bold">{age}</span>!
    </p>
  );
}
