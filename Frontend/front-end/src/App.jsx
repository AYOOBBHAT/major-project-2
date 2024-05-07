import { Route, Routes } from "react-router-dom";
import "./App.css";
import Layout from "./Layout/Layout";
import AboutUs from "./pages/AboutUs.page";
import ContactUs from "./pages/ContactUs.page";
import Home from "./pages/Home.page";

import { ToastContainer } from "react-toastify";
import "react-toastify/dist/ReactToastify.css";

import { useSelector } from "react-redux";

function App() {
  const isDarkMode = useSelector((state) => state.ui.isDarkMode);

  return (
    <div className="App hide-scrollbar">
      <Layout>
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/aboutus" element={<AboutUs />} />
          <Route path="/contactus" element={<ContactUs />} />
        </Routes>
      </Layout>
      <ToastContainer
        style={{ zIndex: 1, marginBottom: "20px", alignItems: "end" }}
        bodyClassName="text-gray-500 text-left"
        className=""
        progressClassName="bg-green-400"
        toastClassName="right-[3rem]"
        theme={isDarkMode ? "dark" : "light"}
        position="bottom-right"
        autoClose={5000}
        hideProgressBar={false}
        newestOnTop
        closeOnClick
        rtl={false}
        pauseOnFocusLoss={true}
        draggable
        pauseOnHover
      />
    </div>
  );
}

export default App;
