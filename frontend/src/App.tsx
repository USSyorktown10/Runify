import { LandingPage } from './pages/LandingPage.tsx';

import { useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import '@/styles/style.css';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
      </Routes>
    </BrowserRouter>
  )
}