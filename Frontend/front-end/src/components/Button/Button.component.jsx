import React from "react";

const Button = ({ children, className, ...props }) => {
  return (
    <button
      className={` border-2 border-[color:var(--color-primary)]  rounded-lg font-semibold duration-500 ease-in-out  ${className}`}
      {...props}
    >
      {children}
    </button>
  );
};

export default Button;
