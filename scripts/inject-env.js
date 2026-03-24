#!/usr/bin/env node
// scripts/inject-env.js
// Copies frontend/ → dist/ then substitutes {{ VAR }} placeholders in dist/index.html
// with Netlify environment variables. Run as Netlify build command — never commit dist/.
const fs = require("fs");
const path = require("path");

const ROOT = path.join(__dirname, "..");
const SRC = path.join(ROOT, "frontend");
const DIST = path.join(ROOT, "dist");

function copyDir(src, dest) {
  fs.mkdirSync(dest, { recursive: true });
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const srcPath = path.join(src, entry.name);
    const destPath = path.join(dest, entry.name);
    if (entry.isDirectory()) {
      copyDir(srcPath, destPath);
    } else {
      fs.copyFileSync(srcPath, destPath);
    }
  }
}

// 1. Copy source to dist
copyDir(SRC, DIST);

// 2. Substitute env var placeholders in dist/index.html only
const TARGET = path.join(DIST, "index.html");
const REPLACEMENTS = {
  HF_SPACE_URL: process.env.HF_SPACE_URL || "",
  BACKUP_API_URL: process.env.BACKUP_API_URL || "",
};

let html = fs.readFileSync(TARGET, "utf8");
for (const [key, value] of Object.entries(REPLACEMENTS)) {
  html = html.replaceAll(`{{ ${key} }}`, value);
}
fs.writeFileSync(TARGET, html);
console.log("inject-env: dist/ built, substituted", Object.keys(REPLACEMENTS).join(", "));
