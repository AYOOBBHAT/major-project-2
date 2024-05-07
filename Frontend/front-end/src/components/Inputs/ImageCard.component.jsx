import React from "react";
import "./ImageCard.component.css";
const ImageCard = ({ url, id, onChangeHandler }) => {
  return (
    <div className="shadow-[2px_4px_8px_rgba(0,0,0,0.4)] w-fit justify-self-center rounded-xl hover:-translate-y-2 duration-300 ease-in-out h-fit">
      <label htmlFor={id} className="">
        <input
          type="radio"
          id={id}
          name="cat"
          defaultChecked={parseInt(id) === 0}
          value={id}
        />
        <img
          src={url}
          alt=""
          className="cursor-pointer object-cover object-center w-24 aspect-square rounded-xl"
          onClick={onChangeHandler}
          data-url={url}
          data-id={id}
        />
      </label>
    </div>
  );
};

export default ImageCard;
