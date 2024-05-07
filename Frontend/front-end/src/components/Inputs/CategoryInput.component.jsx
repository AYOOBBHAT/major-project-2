import React, { useEffect } from "react";
import { twMerge } from "tailwind-merge";
import useSelect from "../../hooks/useSelect.hook";
import { BiCategory } from "react-icons/bi";
import { fetchCategories } from "../../store/actions/category.actions";
import { useDispatch, useSelector } from "react-redux";
import { setImages } from "../../store/reducers/category.Reducer";

const CategoryInput = ({ data, setData, isInputValid, setIsInputValid }) => {
  const categoryList = useSelector((state) => state.category.categoryList);
  const isDarkMode = useSelector((state) => state.ui.isDarkMode);
  const dispatch = useDispatch();
  const {
    optionSelected: selectedCategory,
    isValid: selectedCategoryIsValid,
    hasError: selectedCategoryHasError,
    valueChangeHandler: categoryChangeHandler,
    inputBlurHandler: categoryBlurHandler,
  } = useSelect(data, setData);

  useEffect(() => {
    return () => {
      dispatch(setImages([]));
    };
  });
  useEffect(() => {
    dispatch(fetchCategories());
  }, [dispatch]);

  useEffect(() => {
    setIsInputValid((isInputValid) => {
      return {
        ...isInputValid,
        category: selectedCategoryIsValid,
      };
    });
  }, [selectedCategoryIsValid, setIsInputValid]);

  const invalidContainer = " border-[0.12rem] border-red-400";

  const categoryDisplay = categoryList.map((name) => {
    return (
      <option
        value={name.category}
        className="bg-[color:var(--main-color)]"
        key={name.id}
      >
        {name.category}
      </option>
    );
  });

  return (
    <div className="relative">
      <div
        className={twMerge(
          `flex items-center  ${
            isDarkMode ? "" : "bg-gray-100"
          } px-2 border-[0.12rem] border-gray-100 rounded-md focus-within:border-[color:var(--color-primary)] `,
          `${selectedCategoryHasError ? invalidContainer : ""}`
        )}
      >
        <BiCategory className="text-[color:var(--color-primary)] text-lg mx-1" />
        <select
          name="category"
          className={twMerge(
            `text-[color:var(--secondary-text-color)] text-sm w-full outline-none cursor-context-menu py-3`,
            isDarkMode ? "bg-transparent text-white" : "bg-gray-100"
          )}
          onChange={categoryChangeHandler}
          onBlur={categoryBlurHandler}
          value={selectedCategory}
        >
          <option value="" className="bg-[color:var(--main-color)]" key="">
            Select Category
          </option>
          {categoryDisplay}
        </select>
      </div>
      {selectedCategoryHasError && (
        <p className="text-red-400 text-xs absolute text-left mt-[0.1rem] ">
          Please select a category.
        </p>
      )}
    </div>
  );
};

export default CategoryInput;
