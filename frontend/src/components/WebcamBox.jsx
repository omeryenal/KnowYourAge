import React from "react";
import Webcam from "react-webcam";

/**
 * WebcamBox
 * Renders the webcam preview and the "Predict Age" button.
 * Also includes a loading spinner when prediction is in progress.
 */
export default function WebcamBox({ webcamRef, loading, handlePredict }) {
  return (
    <div className="bg-white bg-opacity-10 backdrop-blur-md rounded-2xl p-4 shadow-xl mb-10">
      {/* Webcam Preview */}
      <Webcam
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className="rounded-xl w-[340px] h-[255px] md:w-[440px] md:h-[330px] lg:w-[520px] lg:h-[390px] shadow-lg"
      />
      <p className="text-sm mt-3 text-white/80">Align your face with the center</p>

      {/* Predict Button */}
      <div className="mt-4 flex justify-center">
        <button
          onClick={handlePredict}
          disabled={loading}
          className="bg-white text-blue-600 font-semibold py-3 px-10 text-lg rounded-2xl shadow-lg hover:bg-blue-100 transition flex items-center justify-center min-w-[150px]"
        >
          {/* Spinner shown when loading */}
          {loading ? (
            <div className="w-5 h-5 border-4 border-white border-t-blue-500 rounded-full animate-spin" />
          ) : (
            "Predict Age"
          )}
        </button>
      </div>
    </div>
  );
}
