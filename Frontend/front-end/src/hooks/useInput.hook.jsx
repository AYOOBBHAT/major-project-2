import { useEffect, useState } from "react";

const useInput = (validateValue, name, data, setData, type) => {
  const [enteredValue, setEnteredValue] = useState(data);
  const [isTouched, setIsTouched] = useState(false);

  const valueIsValid = validateValue(enteredValue);
  const hasError = !valueIsValid && isTouched;

  useEffect(() => {
    setData(enteredValue);
  }, [enteredValue, setData]);

  const valueChangeHandler = (event) => {
    if (type == "file") {
      setEnteredValue(event.target.files[0]);
    } else setEnteredValue(event.target.value);
  };

  const inputBlurHandler = (event) => {
    setIsTouched(true);
  };

  const reset = () => {
    setEnteredValue("");
    setIsTouched(false);
  };

  return {
    value: enteredValue,
    isValid: valueIsValid,
    hasError,
    valueChangeHandler,
    inputBlurHandler,
    reset,
    setEnteredValue,
  };
};

export default useInput;
