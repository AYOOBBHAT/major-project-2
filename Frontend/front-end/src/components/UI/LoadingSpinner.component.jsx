import React from "react";

const LoadingSpinner = ({ containerClass }) => {
  return (
    <div className={`flex items-center justify-center ${containerClass}`}>
      <div className="loader animate-spin bg-red w-10 aspect-square border-[0.4rem] border-[color:var(--color-primary)] border-t-[0.4rem] border-t-[color:var(--main-color)] rounded-full"></div>
    </div>
  );
};

export default LoadingSpinner;
