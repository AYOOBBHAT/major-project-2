import React, { useEffect } from "react";
import { twMerge } from "tailwind-merge";
import useInput from "../../hooks/useInput.hook";

import { BiPhone } from "react-icons/bi";
import { BsKeyboard } from "react-icons/bs";
import { FaRegEnvelope, FaRegUser } from "react-icons/fa";
import { IoImageOutline } from "react-icons/io5";
import { RiLockPasswordLine } from "react-icons/ri";
import { useSelector } from "react-redux";

const inputValidationFunctions = (inputName) => {
  if (inputName.toLowerCase() === "email") {
    return (value) => value.includes("@");
  }
  return (value) => {
    if (typeof value === "string") return value.trim() !== "";
    return value !== "";
  };
};

const Input = ({
  inputFieldName,
  type,
  setIsInputValid,
  isInputValid,
  data,
  setData,
  ...props
}) => {
  const {
    value: enteredValue,
    isValid: enteredValueIsValid,
    hasError: valueInputHasError,
    valueChangeHandler,
    inputBlurHandler: valueBlurHandler,
  } = useInput(
    inputValidationFunctions(inputFieldName),
    inputFieldName.toLowerCase(),
    data,
    setData,
    type
  );
  const isDarkMode = useSelector((state) => state.ui.isDarkMode);
  useEffect(() => {
    setIsInputValid((isInputValid) => {
      return {
        ...isInputValid,
        [inputFieldName.toLowerCase()]: enteredValueIsValid,
      };
    });
  }, [enteredValueIsValid, inputFieldName, setIsInputValid]);

  let icon = (
      <BsKeyboard className="text-[color:var(--color-primary)] text-xl  mx-1" />
    ),
    message = "Enter details correctly";

  if (inputFieldName.toLowerCase() === "username") {
    icon = (
      <FaRegUser className="text-[color:var(--color-primary)]  mx-1 text-md" />
    );
    message = "Username must not be empty.";
  } else if (inputFieldName.includes("Email")) {
    icon = (
      <FaRegEnvelope className="text-[color:var(--color-primary)] text-lg  mx-1" />
    );
    message = "Please enter a valid email.";
  } else if (inputFieldName.toLowerCase() === "otp") {
    icon = (
      <RiLockPasswordLine className="text-[color:var(--color-primary)]  mx-1 text-xl" />
    );
    message = "OTP must not be empty.";
  } else if (inputFieldName.toLowerCase().includes("phone")) {
    icon = (
      <BiPhone className="text-[color:var(--color-primary)]  mx-1 text-xl" />
    );
    message = "Phone number must not be empty.";
  } else if (inputFieldName.toLowerCase().includes("file")) {
    icon = (
      <IoImageOutline className="text-[color:var(--color-primary)]  mx-1 text-xl" />
    );
    message = "Phone number must not be empty.";
  }

  const invalidContainer = " border-[0.12rem] border-red-400 ";

  return (
    <div className="relative">
      <div
        className={twMerge(
          `flex items-center ${
            isDarkMode ? "" : "bg-gray-100"
          } px-2 py-3 border-[0.12rem] border-gray-100 rounded-md focus-within:border-[color:var(--color-primary)]`,
          `${valueInputHasError ? invalidContainer : ""}`
        )}
      >
        {icon}
        {type === "file" ? (
          <input
            name={inputFieldName}
            className={twMerge(
              `  ml-1 outline-none text-[color:var(--secondary-text-color)] text-sm w-full `,
              isDarkMode ? "bg-transparent text-white" : "bg-gray-100"
            )}
            type={type}
            placeholder={inputFieldName}
            onChange={valueChangeHandler}
            onBlur={valueBlurHandler}
            // value={type === "file" ? enteredValue : ""}
            {...props}
          />
        ) : (
          <input
            name={inputFieldName}
            className={twMerge(
              `  ml-1 outline-none text-[color:var(--secondary-text-color)] text-sm w-full `,
              isDarkMode ? "bg-transparent text-white" : "bg-gray-100"
            )}
            type={type}
            placeholder={inputFieldName}
            onChange={valueChangeHandler}
            onBlur={valueBlurHandler}
            value={enteredValue}
            {...props}
          />
        )}
      </div>
      {valueInputHasError && (
        <p className="text-red-400 text-xs text-left absolute mt-[0.1rem]">
          {message}
        </p>
      )}
    </div>
  );
};

export default Input;
