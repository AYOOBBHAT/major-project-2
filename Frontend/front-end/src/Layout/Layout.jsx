import React, { useEffect } from "react";
import Navbar from "./Navbar";

import { MdDarkMode } from "react-icons/md";
import { RiSunFill } from "react-icons/ri";

import { useDispatch, useSelector } from "react-redux";
import { setIsDarkMode } from "../store/reducers/ui.Reducer";
import { twMerge } from "tailwind-merge";

const Layout = (props) => {
  const dispatch = useDispatch();
  const isDarkMode = useSelector((state) => state.ui.isDarkMode);

  useEffect(() => {
    if (isDarkMode) {
      document.body.classList.add("dark-mode");
    } else {
      document.body.classList.remove("dark-mode");
    }
  }, [isDarkMode]);

  const darkModeChangeHandler = () => {
    dispatch(setIsDarkMode(!isDarkMode));
  };

  return (
    <React.Fragment>
      <Navbar />
      <div className="w-11/12 md:w-10/12  lg:w-4/5 max-w-[1440px] m-24 mx-auto">
        <main className=" justify-center items-center">{props.children}</main>
      </div>
      <div className="button fixed bottom-8 bg-[color:var(--color-primary)] shadow-lg shadow-[color:var(--shadow-color)] w-16 h-10 rounded-r-full flex justify-end items-center">
        <button
          className={twMerge(
            "p-2 text-[color:var(--main-color)]",
            isDarkMode ? "hover:text-white" : "hover:text-black"
          )}
          onClick={darkModeChangeHandler}
        >
          {isDarkMode && <MdDarkMode className="text-2xl " />}
          {!isDarkMode && <RiSunFill className="text-2xl " />}
        </button>
      </div>
    </React.Fragment>
  );
};

export default Layout;
