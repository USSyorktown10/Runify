import { useState } from 'react';
import { Topography}  from '../components/Topography';
import '@/styles/style.css';

export function LandingPage() {
    return (
            <div style={{ width: '100%', height: '100%', position: 'relative' }}>
                <Topography
                    color="#10b981"
                    lineCount={60}
                    lineWidth={0.5}
                    elevation={0}
                    foreground={0.17}
                    tilt={90}
                    speed={0.25}
                    direction="forward"
                    glow={0}
                    depthOfField={0.4}
                    fog={0.35}
                    hoverEffect="lift"
                    clickEffect="pulse"
                    majorLines
                    scale={1}
                    detail={1}
                    morph={0.2}
                    brightness={0.4}
                    markers="none"
                    glitch={0}
                    opacity={1}
                    mouseInteraction
                    mouseRadius={1}
                    mouseStrength={0}
                    trail={0.3}
                    parallax={0.5}
                    intro
                    lightMode={false}
                />
            </div>
    )
}