import React, { useEffect } from "react";
import { twMerge } from "tailwind-merge";
import useInput from "../../hooks/useInput.hook";

import { AiOutlineMessage } from "react-icons/ai";
import { useSelector } from "react-redux";

const inputValidationFunctions = () => {
  return (value) => value.trim() !== "";
};

const TextArea = ({
  inputFieldName,
  type,
  setIsInputValid,
  isInputValid,
  data,
  setData,
}) => {
  const {
    value: enteredValue,
    isValid: enteredValueIsValid,
    hasError: valueInputHasError,
    valueChangeHandler,
    inputBlurHandler: valueBlurHandler,
  } = useInput(
    inputValidationFunctions(),
    inputFieldName.toLowerCase(),
    data,
    setData
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

  const invalidContainer = " border-[0.12rem] border-red-400 ";

  return (
    <div className="relative">
      <div
        className={twMerge(
          `flex  ${
            isDarkMode ? "" : "bg-gray-100"
          } px-2 py-3 border-[0.12rem] border-gray-100 rounded-md focus-within:border-[color:var(--color-primary)]`,
          `${valueInputHasError ? invalidContainer : ""}`
        )}
      >
        <AiOutlineMessage className=" text-xl text-[color:var(--color-primary)]" />
        <textarea
          name={inputFieldName}
          className={twMerge(
            `ml-1 outline-none text-[color:var(--secondary-text-color)] text-sm w-full resize-none`,
            isDarkMode ? "bg-transparent text-white" : "bg-gray-100"
          )}
          type={type}
          placeholder={inputFieldName}
          onChange={valueChangeHandler}
          onBlur={valueBlurHandler}
          value={enteredValue}
          rows={6}
        ></textarea>
      </div>
      {valueInputHasError && (
        <p className="text-red-400 text-xs text-left absolute mt-[0.1rem]">
          Message must not be empty.
        </p>
      )}
    </div>
  );
};

export default TextArea;
