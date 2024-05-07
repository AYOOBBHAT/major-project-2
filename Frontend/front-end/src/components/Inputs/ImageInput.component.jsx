import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { setPassImage as setPassImageSignUp } from "../../store/reducers/signUp.Reducer";
import { setPassImage as setPassImageAccountRecovery } from "../../store/reducers/accountRecovery.Reducer";
import ImageCard from "./ImageCard.component";

const ImageInput = ({ isSignUp }) => {
  const images = useSelector((state) => state.category.images);
  console.log(images);
  const dispatch = useDispatch();
  const [selectedImage, setSelectedImage] = useState({});

  useEffect(() => {
    if (images.length !== 0) {
      setSelectedImage(images[0]);
    }
  }, [images]);

  useEffect(() => {
    if (isSignUp) dispatch(setPassImageSignUp(selectedImage));
    else dispatch(setPassImageAccountRecovery(selectedImage));
  }, [selectedImage,isSignUp, dispatch]);

  const onChangeHandler = (event) => {
    setSelectedImage({
      id: event.target.getAttribute("data-id"),
      url: event.target.getAttribute("data-url"),
    });
  };

  const ImageList = images.map((img) => {
    return (
      <ImageCard
        url={img.url}
        id={img.id}
        key={img.id}
        onChangeHandler={onChangeHandler}
      />
    );
  });

  return (
    <div className="mt-8">
      <div className="flex justify-center items-center p-2 mb-4">
        <div className="shadow-[4px_4px_12px_rgba(0,0,0,0.4)] rounded-md overflow-hidden border-white border-8">
          <img
            src={selectedImage.url}
            className="w-[180px] aspect-square object-cover object-center"
            alt="pass_image selected"
          />
        </div>
      </div>
      <div className="list grid grid-cols-2 sm:grid-cols-3 gap-6 overflow-y-scroll h-56 p-2">
        {ImageList}
      </div>
    </div>
  );
};

export default ImageInput;
