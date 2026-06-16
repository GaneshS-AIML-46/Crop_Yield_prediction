// Update PRODUCTION_API_URL after deploying backend to Render.com
const PRODUCTION_API_URL = 'https://crop-yield-api.onrender.com';

const API_BASE_URL =
  window.location.hostname === 'localhost' ||
  window.location.hostname === '127.0.0.1'
    ? 'http://localhost:8000'
    : PRODUCTION_API_URL;
