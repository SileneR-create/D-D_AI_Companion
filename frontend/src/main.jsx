import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App.jsx";
import { AuthProvider } from "./auth/AuthContext.jsx";
import { ActiveCampaignProvider } from "./campaign/ActiveCampaignContext.jsx";
import "./index.css";

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <AuthProvider>
      <ActiveCampaignProvider>
        <App />
      </ActiveCampaignProvider>
    </AuthProvider>
  </React.StrictMode>,
);
