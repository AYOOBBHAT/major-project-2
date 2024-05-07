import React, {
  useEffect,
  useRef,
  useState,
  forwardRef,
  useImperativeHandle,
  useMemo,
} from "react";

import { FaAngleDoubleRight } from "react-icons/fa";
import { twMerge } from "tailwind-merge";
import { debounce } from "lodash";

const isTouchDevice = "ontouchstart" in document.documentElement;

const SliderInput = forwardRef(
  (
    { sliderHandler, isFound, text, nextImageSlide, isRightSlide = true },
    ref
  ) => {
    const [isUnlocked, setIsUnlocked] = useState(false);

    // for selecting the slider circle to move the circle
    const slider = useRef(null);

    // to get the width of the container
    const container = useRef(null);

    let isDragging = false,
      startX = 0,
      sliderLeft = 0,
      containerWidth = 0;

    useEffect(() => {
      if (isUnlocked) {
        sliderHandler(isRightSlide, isUnlocked);
      }
    }, [isUnlocked, isRightSlide, sliderHandler]);

    useEffect(() => {
      document.addEventListener("keydown", onkeydown);
      if (isTouchDevice) {
        document.addEventListener("touchmove", onDrag);
        document.addEventListener("touchend", stopDrag);
      } else {
        document.addEventListener("mousemove", onDrag);
        document.addEventListener("mouseup", stopDrag);
      }
      return () => {
        document.removeEventListener("keydown", onkeydown);
        if (isTouchDevice) {
          document.removeEventListener("touchmove", onDrag);
          document.removeEventListener("touchend", stopDrag);
        } else {
          document.removeEventListener("mousemove", onDrag);
          document.removeEventListener("mouseup", stopDrag);
        }
      };
    });

    const updateSliderStyle = () => {
      slider.current.style.width = sliderLeft + 32 + "px";
    };

    const onDrag = (e) => {
      if (isUnlocked) {
        return;
      }
      if (!isDragging) return;

      if (isTouchDevice) {
        sliderLeft = Math.min(
          Math.max(
            0,
            !isRightSlide
              ? -(e.touches[0].clientX - startX)
              : e.touches[0].clientX - startX
          ),
          containerWidth
        );
      } else {
        sliderLeft = Math.min(
          Math.max(
            0,
            !isRightSlide ? -(e.clientX - startX) : e.clientX - startX
          ),
          containerWidth
        );
      }
      updateSliderStyle();
    };

    const stopDrag = () => {
      if (isUnlocked) return;

      if (!isDragging) return;

      isDragging = false;
      if (!isRightSlide) {
        sliderLeft = Math.abs(sliderLeft);
      }
      if (sliderLeft > containerWidth * 0.9) {
        sliderLeft = containerWidth;
        setIsUnlocked(true);
        nextImageSlide();
      } else {
        sliderLeft = 0;
      }

      slider.current.classList.add("duration-500");
      slider.current.classList.add("ease-in-out");
      updateSliderStyle();
    };

    const startDrag = (event) => {
      if (isUnlocked) {
        return;
      }
      isDragging = true;

      if (slider.current.classList.contains("duration-500")) {
        slider.current.classList.remove("duration-500");
        slider.current.classList.remove("ease-in-out");
      }
      if (isTouchDevice) {
        startX = event.touches[0].clientX;
      } else {
        startX = event.clientX;
      }
      containerWidth = container.current.clientWidth - 50;
    };

    // bcoz we need access of reset function in parent therefore we used refs here
    useImperativeHandle(ref, () => ({
      reset() {
        if (!isUnlocked) return;
        setIsUnlocked((isUnlocked) => false);
        sliderLeft = 0;
        if (slider.current.classList.contains("duration-500")) {
          slider.current.classList.remove("duration-500", "ease-in-out");
        }
        updateSliderStyle();
      },
      setSlider() {
        containerWidth = container.current.clientWidth - 50;
        sliderLeft = containerWidth;
        updateSliderStyle();
        setIsUnlocked(true);
      },
    }));

    // Unlocked the slider
    const setSlider = () => {
      containerWidth = container.current.clientWidth - 50;
      sliderLeft = containerWidth;
      updateSliderStyle();
      setIsUnlocked(true);
    };
    // Using debounce concept to remove ambiguity (that sometime both button get unlocked which is avoided by debounce)
    const deb = useMemo(
      () => debounce(() => setSlider(), 200),
      // eslint-disable-next-line react-hooks/exhaustive-deps
      []
    );

    const onkeydown = (event) => {
      if (event.keyCode === 39) {
        //right Yes
        if (isRightSlide && !isUnlocked) {
          deb();
          nextImageSlide();
        }
      }
      if (event.keyCode === 37) {
        //left No
        if (!isRightSlide && !isUnlocked) {
          deb();
          nextImageSlide();
        }
      }
    };

    return (
      <div
        className={twMerge(
          "w-full h-12 rounded-full",
          isUnlocked ? "w-24 duration-500 ease-in-out" : ""
        )}
      >
        <div
          className={twMerge(
            "relative top-9 left-[40%] font-bold text-gray-400 w-12 flex items-center justify-center",
            isUnlocked ? "z-20 text-white left-[25%]" : ""
          )}
        >
          <h1 className="cursor-default text-md">{text}</h1>
        </div>
        <div
          className={twMerge(
            "container bg-white h-full w-full rounded-full flex items-center p-2 overflow-hidden shadow-[2px_4px_12px_rgba(0,0,0,0.2)] ",
            !isRightSlide ? "justify-end" : "justify-start"
          )}
          ref={container}
        >
          <div
            className={twMerge(
              "sliderCircle h-full aspect-square rounded-full bg-[color:var(--color-primary)] flex items-center z-10 cursor-pointer text-white ",
              !isRightSlide ? "justify-start" : "justify-end"
            )}
            ref={slider}
            onMouseDown={startDrag}
            onTouchStart={startDrag}
          >
            {!isUnlocked && (
              <FaAngleDoubleRight
                className={twMerge(" mx-2", !isRightSlide ? "rotate-180" : "")}
              />
            )}
          </div>
        </div>
      </div>
    );
  }
);

export default SliderInput;
