import React, { useState } from "react";
import { FaBars } from "react-icons/fa";
import { ImCross } from "react-icons/im";
import { useSelector } from "react-redux";
import { Link, NavLink } from "react-router-dom";
import { twMerge } from "tailwind-merge";
import logo from "../assets/logo/logo.png";
import "./navbar.css";

const Navbar = () => {
  const [dropMenu, setDropMenu] = useState(false);
  const dropMenuHandler = () => {
    setDropMenu((dropMenu) => !dropMenu);
  };

  const isDarkMode = useSelector((state) => state.ui.isDarkMode);

  return (
    <React.Fragment>
      <nav
        className={twMerge(
          " p-4 md:p-3 bg-[color:var(--main-color)]  shadow-lg border-b-[0.3rem] border-[color:var(--color-primary)] fixed top-0 left-0 w-full z-50",
          isDarkMode
            ? "shadow-slate-800"
            : "shadow-[color:var(--shadow-color)] "
        )}
      >
        <div className="w-11/12 md:w-[85%]  lg:w-4/5 mx-auto max-w-[1440px] flex justify-between items-center">
          <div className="logo">
            <h1 className="flex items-center gap-3 text-2xl font-semibold">
              <Link to="/">
                <img src={logo} className="h-12 scale-110" alt="" />
              </Link>
              <span className="text-[color:var(--tertiary-text-color)] ">
                Derm
              </span>{" "}
              <span className="text-[color:var(--color-primary)]">Insight</span>
            </h1>
          </div>
          

          {/* Vissble When Screen Is Greater than 768 pixels */}
          <div className="items-center hidden gap-6 md:flex lg:gap-12">
            <div className="text-[color:var(--tertiary-text-color)] ">
              <ul className="flex gap-6 font-semibold lg:gap-8 text-md">
                <NavLink
                  to="/"
                  className="hover:text-[color:var(--color-primary)]"
                >
                  Home
                </NavLink>
                <NavLink
                  to="/aboutus"
                  className="hover:text-[color:var(--color-primary)]"
                >
                  About
                </NavLink>
                <NavLink
                  to="/contactus"
                  className="hover:text-[color:var(--color-primary)]"
                >
                  Contact
                </NavLink>
              </ul>
            </div>
          </div>
          {/* Burger Menu For Smaller Screen Sizes */}
          <div className="flex items-center justify-center md:hidden">
            <button className="" onClick={dropMenuHandler}>
              {!dropMenu && (
                <FaBars className="text-2xl text-[color:var(--tertiary-text-color)]" />
              )}
              {dropMenu && (
                <ImCross className="text-2xl text-[color:var(--tertiary-text-color)]" />
              )}
            </button>
          </div>
        </div>
      </nav>
      {dropMenu && (
        <div
          className={twMerge(
            "flex flex-col items-center justify-center gap-6 md:hidden fixed z-10 top-20 left-0 bg-[color:var(--main-color)] w-full py-8 shadow-lg",
            isDarkMode
              ? "shadow-gray-800"
              : "shadow-[color:var(--shadow-color)]"
          )}
        >
          <div className="text-center text-[color:var(--tertiary-text-color)] ">
            <ul className="flex flex-col gap-4 text-lg font-semibold">
              <Link
                to="/"
                className="hover:text-[color:var(--color-primary)]"
                onClick={dropMenuHandler}
              >
                Home
              </Link>
              <Link
                to="/aboutus"
                className="hover:text-[color:var(--color-primary)]"
                onClick={dropMenuHandler}
              >
                About
              </Link>
              <Link
                to="/contactus"
                className="hover:text-[color:var(--color-primary)]"
                onClick={dropMenuHandler}
              >
                Contact
              </Link>
            </ul>
          </div>
        </div>
      )}
    </React.Fragment>
  );
};

export default Navbar;
