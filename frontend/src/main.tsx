/**
 * main.tsx
 * --------
 * The entry point. This is the very first code the browser runs.
 *
 * It finds the empty <div id="root"> in index.html and tells React to take it
 * over and render our App component inside it.
 */

import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import "./styles.css";

const container = document.getElementById("root");
if (!container) throw new Error("Could not find the #root element in index.html");

createRoot(container).render(
  <StrictMode>
    <App />
  </StrictMode>
);
