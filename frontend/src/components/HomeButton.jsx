import React from "react";
import { useNavigate } from "react-router-dom";
import { HomeIcon } from "lucide-react";

/**
 * HomeButton
 * Returns to the landing screen when clicked.
 * Uses React Router's `navigate("/")` to perform the redirect.
 */
export default function HomeButton() {
  const navigate = useNavigate();

  return (
    <button
      onClick={() => navigate("/")}
      className="flex items-center gap-2 bg-white text-blue-600 py-3 px-6 text-lg font-medium rounded-xl shadow hover:bg-blue-100 transition"
    >
      <HomeIcon className="w-5 h-5" />
      Home
    </button>
  );
}
