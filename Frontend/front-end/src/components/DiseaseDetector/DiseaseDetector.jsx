import axios from "axios";
import React, { useEffect, useState } from "react";
import { useDispatch, useSelector } from "react-redux";
import { toast } from "react-toastify";
import { twMerge } from "tailwind-merge";
import { setIsLoading } from "../../store/reducers/ui.Reducer";
import Button from "../Button/Button.component";
import Input from "../Inputs/Input.component";
import LoadingSpinner from "../UI/LoadingSpinner.component";

const DiseaseDetector = ({}) => {
  const isDarkMode = useSelector((state) => state.ui.isDarkMode);
  const isLoading = useSelector((state) => state.ui.isLoading);
  const [imageType, setImageType] = useState("COLOURED");
  const [isFormValid, setIsFormValid] = useState(false);
  const [file, setFile] = useState("");
  const [detectData, setDetectData] = useState(null);

  const dispatch = useDispatch();

  const [isInputValid, setIsInputValid] = useState({
    file: false,
  });

  useEffect(() => {
    if (isInputValid.file) {
      setIsFormValid(true);
    } else {
      setIsFormValid(false);
    }
  }, [isInputValid]);

  const onSubmitHandler = async (event) => {
    event.preventDefault();
    if (!file) {
      toast.warning("Please select an image!");
      return;
    }
    dispatch(setIsLoading(true));
    const formData = new FormData();
    formData.append("file", file);
    const res = await axios({
      method: "post",
      data: formData,
      url: `/predict?image_type=${imageType}`,
    });
    if (res.status === 200) {
      toast.success("Successfully Detected!");
      dispatch(setIsLoading(false));
      setDetectData({
        // plant: res.data.class.split("___")[0],
        // label: res.data.class.split("___")[1],
        // confidence: res.data.confidence,
        numberPlate: res.data.numberPlate,
        time: res.data.time,
      });
    } else {
      toast.error("Something went wrong!");
      console.log(res);
    }
  };

  const resetForm = () => {
    setFile("");
    setImageType("coloured");
    setIsFormValid(false);
    setIsInputValid({
      file: false,
    });
    setDetectData(null);
  };

  return (
    <div className="mt-6 bg-[color:var(--main-color)] shadow-[2px_4px_12px_rgba(0,0,0,0.2)] max-w-[780px] md:mx-auto shadow-[color:var(--shadow-color)] rounded-xl p-4  flex flex-col-reverse sm:flex-row ">
      <div
        className={twMerge(
          "bg-gradient-to-r from-[color:var(--color-primary)] to-teal-300 right-half p-8 basis-1/2 flex flex-col  rounded-xl justify-between",
          isDarkMode ? "text-black" : "text-white"
        )}
      >
        {file && (
          <img
            src={window.URL.createObjectURL(file)}
            className="w-100 rounded-xl"
          />
        )}
      </div>
      <div className="flex justify-start p-8 basis-1/2 flex-column">
        {!detectData ? (
          <form
            action=""
            className="m-auto"
            onSubmit={onSubmitHandler}
            //   enctype="multipart/form-data"
          >
            <div className="flex flex-col gap-8 body"></div>
            <Input
              data={file}
              accept="image/*"
              setData={setFile}
              inputFieldName="file"
              setIsInputValid={setIsInputValid}
              isInputValid={isInputValid}
              type="file"
            />

            {/* <div
              className={twMerge(
                `flex items-center ${
                  isDarkMode ? "" : "bg-gray-100"
                } px-2 py-2 mt-4 border-[0.12rem] border-gray-100 rounded-md focus-within:border-[color:var(--color-primary)]`
              )}
            >
              <select
                label="Select Version"
                defaultValue={imageType}
                className={twMerge(
                  ` rounded-md outline-none text-[color:var(--secondary-text-color)] text-sm w-full `,
                  isDarkMode ? "bg-transparent text-white" : "bg-gray-100"
                )}
                onChange={(e) => setImageType(e.target.value)}
              >
                <option value="COLOURED">Coloured</option>
                <option value="GRAYSCALE">Grayscale</option>
                <option value="SEGMENTED">Segmented</option>
              </select>
            </div> */}
            <div className="flex justify-around mt-8 footer">
              {isLoading && <LoadingSpinner />}
              {!isLoading && (
                <Button
                  type="submit"
                  className="px-4 py-2 btn-base"
                  disabled={!isFormValid || isLoading}
                >
                  DETECT
                </Button>
              )}
            </div>
          </form>
        ) : (
          <div className="flex flex-col items-center justify-center flex-1">
            <div
              className={twMerge(
                `flex items-center w-full ${
                  isDarkMode ? "" : "bg-gray-100"
                } px-2 py-2 mt-4 border-[0.12rem] border-gray-100 rounded-md focus-within:border-[color:var(--color-primary)]`
              )}
            >
              <h3 className="text-[color:var(--tertiary-text-color)] text-lg font-semibold  text-center ">
                Number Plate - &nbsp;
              </h3>{" "}
              <h3 className="text-[color:var(--color-primary)] text-xl font-semibold  text-center ">
                {detectData?.numberPlate}
              </h3>
            </div>
            <div
              className={twMerge(
                `flex items-center w-full ${
                  isDarkMode ? "" : "bg-gray-100"
                } px-2 py-2 mt-4 border-[0.12rem] border-gray-100 rounded-md focus-within:border-[color:var(--color-primary)]`
              )}
            >
              <h3 className="text-[color:var(--tertiary-text-color)] text-lg font-semibold  text-center ">
                Date - &nbsp;
              </h3>{" "}
              <h3 className="text-[color:var(--color-primary)] text-xl font-semibold  text-center ">
                {new Date(detectData?.time).toDateString()}
              </h3>
            </div>
            <div
              className={twMerge(
                `flex items-center w-full ${
                  isDarkMode ? "" : "bg-gray-100"
                } px-2 py-2 mt-4 border-[0.12rem] border-gray-100 rounded-md focus-within:border-[color:var(--color-primary)]`
              )}
            >
              <h3 className="text-[color:var(--tertiary-text-color)] text-lg font-semibold  text-center ">
                Time - &nbsp;
              </h3>{" "}
              <h3 className="text-[color:var(--color-primary)] text-xl font-semibold  text-center ">
              {new Date(detectData?.time).toLocaleTimeString()}
              </h3>
            </div>
            <div className="flex gap-4">
              <Button
                type="submit"
                className="px-4 py-2 mt-5 btn-inverted"
                onClick={() => setDetectData(null)}
              >
                BACK
              </Button>
              <Button
                type="submit"
                className="px-4 py-2 mt-5 btn-base"
                onClick={resetForm}
              >
                CLEAR
              </Button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default DiseaseDetector;
