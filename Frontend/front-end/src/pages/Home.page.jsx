import React from "react";
import { useSelector } from "react-redux";
import { twMerge } from "tailwind-merge";
import DiseaseDetector from "../components/DiseaseDetector/DiseaseDetector";
// import SihLogo from "../assets/Images/SihLogo.png";

const Home = () => {
  const isDarkMode = useSelector((state) => state.ui.isDarkMode);
  return (
    <div className="mx-auto mt-28">
      <div className="flex flex-col items-center justify-center header">
        <div className="flex gap-3">
          <h1 className="mb-2 text-4xl font-semibold text-center text-[color:var(--tertiary-text-color)]">
          </h1>
          <h1 className="text-[color:var(--color-primary)] text-4xl font-semibold mb-2 text-center">
            Derm Insight
          </h1>
        </div>
        <div className="h-[0.30rem] w-12 bg-[color:var(--color-primary)] rounded-full"></div>
        <p
          className={twMerge(
            "mt-4 text-lg",
            isDarkMode ? "text-gray-300" : "text-gray-400"
          )}
        >
          Skin Disease Detection
        </p>
      </div>
      <DiseaseDetector />
    </div>
  );
};

export default Home;
