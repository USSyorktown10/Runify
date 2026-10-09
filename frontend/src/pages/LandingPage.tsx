import { useState } from 'react';
import { RunHero } from '../components/RunHero.tsx';
import '@/styles/style.css';

export function LandingPage() {
    return (
        <div className="landing-page">
            <h1>Yo</h1>
            <RunHero />
        </div>
    )
}