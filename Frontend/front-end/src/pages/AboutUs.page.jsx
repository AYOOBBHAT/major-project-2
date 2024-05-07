import React from "react";

import ravi from "../assets/members/ravi.jpeg";
import Vicky from "../assets/members/vicky.jpeg";

import { BsFacebook, BsGithub, BsLinkedin } from "react-icons/bs";
import { FaServicestack } from "react-icons/fa";
import { FiTarget } from "react-icons/fi";
import { RiMessage3Fill } from "react-icons/ri";
import { useSelector } from "react-redux";
import { twMerge } from "tailwind-merge";

const members = [
  {
    id: 1,
    name: "Vicky Gupta",
    education: "B.Tech CSE",
    url: Vicky,
    linkedInUrl: "https://www.linkedin.com/in/vickyguptaa7/",
    githubUrl: "https://github.com/vickyguptaa7",
    facebookUrl: "https://www.facebook.com/",
  },
  {
    id: 2,
    name: "Ravi Gowri Jaswanth",
    education: "B.Tech CSE",
    url: ravi,
    linkedInUrl: "https://www.linkedin.com/in/ravi-gowri-jaswanth-578a66202/",
    githubUrl: "https://github.com/Vicky-Guptaa",
    facebookUrl: "https://www.facebook.com/",
  },
];

const AboutUs = () => {
  const isDarkMode = useSelector((state) => state.ui.isDarkMode);
  const allMembers = members.map((member) => {
    return (
      <div
        className=" bg-[color:var(--main-color)] mx-6 mb-12 shadow-[0px_0px_10px_rgba(0,0,0,0.4)] rounded-xl overflow-hidden hover:-translate-y-4 duration-700"
        key={member.id}
      >
        <div className=" border-8 border-[color:var(--color-primary)] bg-[color:var(--color-primary)] rounded-br-[90px] overflow-hidden">
          <img
            src={member.url}
            className="w-56 h-64 object-cover object-top hover:scale-125 duration-[6s] ease-linear"
            alt=""
          />
        </div>
        <div className="p-4 text-center">
          <h3 className="text-left text-[color:var(--tertiary-text-color)] font-bold tracking-wider">
            {member.name.toUpperCase()}
          </h3>
          <h4 className="text-left mt-1 text-[color:var(--color-primary)] text-md font-bold">
            {member.education}
          </h4>
          <div className="flex mt-4 gap-4 text-lg text-[color:var(--tertiary-text-color)]">
            <a href={member.facebookUrl} target="_blank" rel="noreferrer">
              <BsFacebook className="hover:text-[color:var(--color-primary)] hover:scale-110 duration-300  ease-in-out" />
            </a>
            <a href={member.linkedInUrl} target="_blank" rel="noreferrer">
              <BsLinkedin className="hover:text-[color:var(--color-primary)] hover:scale-110 duration-300  ease-in-out" />
            </a>
            <a href={member.githubUrl} target="_blank" rel="noreferrer">
              <BsGithub className="hover:text-[color:var(--color-primary)] hover:scale-110 duration-300  ease-in-out" />
            </a>
          </div>
        </div>
      </div>
    );
  });
  return (
    <div className="mx-auto mt-28">
      <div className="flex flex-col items-center justify-center header">
        <h1 className="text-[color:var(--color-primary)] text-4xl font-semibold mb-2 text-center">
          About Us
        </h1>
        <div className="h-[0.30rem] w-12 bg-[color:var(--color-primary)] rounded-full"></div>
      </div>
      <p
        className={twMerge(
          "tracking-wider mt-6 text-center text-[0.95rem] ",
          isDarkMode ? "text-gray-300" : "text-gray-400"
        )}
      >
        LeafRakshak is an innovative plant leaf disease detection project aimed
        at safeguarding the health and vitality of plant life. Leveraging
        advanced technology and artificial intelligence, LeafRakshak is designed
        to identify and diagnose diseases affecting plant leaves, providing
        timely insights for effective intervention and care.
      </p>

      <div className="flex flex-wrap items-center mt-12 justify-evenly">
        <div className="rounded-xl overflow-hidden w-80 mx-4 mb-8 shadow-[0px_0px_10px_rgba(0,0,0,0.4)] hover:scale-105 duration-700 ease-in-out">
          <div className="bg-[color:var(--color-primary)] px-4 py-3 text-white text-lg flex items-center gap-2">
            <FiTarget className="text-2xl" />
            <h3 className="font-bold">Mission</h3>
          </div>
          <div className="bg-[color:var(--main-color)] p-4">
            <p
              className={twMerge(
                "tracking-wider text-[color:var(--tertiary-text-color)] text-justify text-sm ",
                isDarkMode ? "text-gray-300" : "text-gray-400"
              )}
            >
              LeafRakshak strives to transform plant care through AI-powered
              disease detection. Our goal is to empower individuals, gardeners,
              and farmers with the tools and knowledge to preserve plant health,
              fostering a greener and more sustainable world.
            </p>
          </div>
        </div>
        <div className="rounded-xl overflow-hidden w-80 mx-4 mb-8 shadow-[0px_0px_10px_rgba(0,0,0,0.4)] hover:scale-105 duration-700 ease-in-out">
          <div className="bg-[color:var(--color-primary)] px-4 py-3 text-white text-lg flex items-center gap-2">
            <FaServicestack className="text-2xl" />
            <h3 className="font-bold">Service</h3>
          </div>
          <div className="bg-[color:var(--main-color)] p-4">
            <p
              className={twMerge(
                "tracking-wider text-[color:var(--tertiary-text-color)] text-justify text-sm ",
                isDarkMode ? "text-gray-300" : "text-gray-400"
              )}
            >
              LeafRakshak revolutionizes plant care through AI-driven disease
              detection, personalized treatment advice, real-time monitoring,
              educational resources, and a supportive community. Cultivate
              healthier plants effortlessly with our comprehensive services.
            </p>
          </div>
        </div>
        <div className="rounded-xl overflow-hidden w-80 mx-4 mb-8 shadow-[0px_0px_10px_rgba(0,0,0,0.4)] hover:scale-105 duration-700 ease-in-out">
          <div className="bg-[color:var(--color-primary)] px-4 py-3 text-white text-lg flex items-center gap-2">
            <RiMessage3Fill className="text-2xl" />
            <h3 className="font-bold">Message</h3>
          </div>
          <div className="bg-[color:var(--main-color)] p-4">
            <p
              className={twMerge(
                "tracking-wider  text-[color:var(--tertiary-text-color)] text-sm text-justify ",
                isDarkMode ? "text-gray-300" : "text-gray-400"
              )}
            >
              Our mission is to make disease detection easy and empowering. We provide
              accurate diagnoses, personalized recommendations, and a supportive
              community. Join us in cultivating healthier, greener spaces for a
              sustainable world."
            </p>
          </div>
        </div>
      </div>
      <div className="flex flex-col items-center justify-center mt-6 header">
        <h1 className=" text-[color:var(--color-primary)] text-4xl font-semibold mb-2 text-center">
          Our Team
        </h1>
        <div className="h-[0.30rem] w-12 bg-[color:var(--color-primary)] rounded-full"></div>
      </div>
      <div className="flex flex-wrap items-center mt-12 justify-evenly">
        {allMembers}
      </div>
    </div>
  );
};

export default AboutUs;
