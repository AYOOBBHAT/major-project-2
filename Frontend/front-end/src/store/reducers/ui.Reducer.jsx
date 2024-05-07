import { createSlice } from "@reduxjs/toolkit";

const checkForDarkMode = () => {
  if (localStorage.getItem("isDarkMode")) {
    return localStorage.getItem("isDarkMode")==="true";
  }
  return false;
};

const uiInitialState = {
  isLoading: false,
  isDarkMode: checkForDarkMode(),
};

const uiSlice = createSlice({
  name: "ui",
  initialState: uiInitialState,
  reducers: {
    setIsLoading(state, action) {
      state.isLoading = action.payload;
    },
    setIsDarkMode(state, action) {
      localStorage.setItem("isDarkMode", action.payload);
      state.isDarkMode = action.payload;
    },
  },
});

export const { setIsLoading, setIsDarkMode } = uiSlice.actions;

export default uiSlice.reducer;
