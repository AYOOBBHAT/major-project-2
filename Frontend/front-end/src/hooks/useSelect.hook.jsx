import { useEffect, useState } from "react";
import { useDispatch } from "react-redux";

const useSelect = (data, setData) => {
  const dispatch = useDispatch();
  const [selected, setSelected] = useState(data);
  const [isTouched, setIsTouched] = useState(false);

  useEffect(() => {
    dispatch(setData(selected));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected, dispatch]);

  let valueIsValid = true;

  if (selected === "") {
    valueIsValid = false;
  }

  const hasError = !valueIsValid && isTouched;

  const valueChangeHandler = (event) => {
    setSelected(event.target.value);
  };

  const inputBlurHandler = (event) => {
    setIsTouched(true);
  };

  const reset = () => {
    setSelected("");
    setIsTouched(false);
  };

  return {
    optionSelected: selected,
    isValid: valueIsValid,
    hasError,
    valueChangeHandler,
    inputBlurHandler,
    reset,
    setSelected,
  };
};

export default useSelect;
