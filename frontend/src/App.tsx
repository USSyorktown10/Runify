import { useState } from 'react'
import { BrowserRouter, Navigate, Route, Routes, useParams } from "react-router-dom";
import './App.css'

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LandingPage />} />
      </Routes>
    </BrowserRouter>
  )
}