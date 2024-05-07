import { configureStore } from "@reduxjs/toolkit";
import uiReducer from "./reducers/ui.Reducer";

const store = configureStore({
  reducer: {
    ui: uiReducer,
  },
});

export default store;
