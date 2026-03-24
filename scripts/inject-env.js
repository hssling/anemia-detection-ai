#!/usr/bin/env node
// scripts/inject-env.js
// Replaces {{ VAR }} placeholders in frontend/index.html with Netlify env vars at build time.
const fs = require("fs");
const path = require("path");

const TARGET = path.join(__dirname, "..", "frontend", "index.html");
const REPLACEMENTS = {
  HF_SPACE_URL: process.env.HF_SPACE_URL || "",
  BACKUP_API_URL: process.env.BACKUP_API_URL || "",
};

let html = fs.readFileSync(TARGET, "utf8");
for (const [key, value] of Object.entries(REPLACEMENTS)) {
  html = html.replaceAll(`{{ ${key} }}`, value);
}
fs.writeFileSync(TARGET, html);
console.log("inject-env: substituted", Object.keys(REPLACEMENTS).join(", "));
