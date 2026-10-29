import { LandingPage } from './pages/LandingPage.tsx';

import { useState } from 'react';
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import { Navbar } from './components/Navbar.tsx';
import '@/styles/style.css';

export default function App() {
  return (
    <BrowserRouter>
      <Navbar />
      <Routes>
        <Route path="/" element={<LandingPage />} />
      </Routes>
    </BrowserRouter>
  )
}