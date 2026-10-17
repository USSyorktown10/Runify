import { useState } from 'react';
import { Topography}  from '../components/Topography';
import { RunHero } from '../components/RunHero';
import { TechText } from '../components/TechText';
import '@/styles/style.css';
import '@/styles/landingpage.css';

export function LandingPage() {
    return (
        <div className="landing-page">
            <div className="container">
                <TechText
                    text="Runify"
                    fontWeight={600}
                    fontSize={150}
                    reveal="letter"
                    dashLength={4}
                    dashGap={2}
                    specks={15}
                    intro="metrics"
                    introDuration={2}
                    clickKick
                    guides={false}
                    fontFamily=""
                    letterSpacing={-0.05}
                    accentColor="#10b981"
                    reach={200}
                    softness={0.7}
                    lineStyle="dashed"
                    strokeWidth={1.5}
                    speed={1}
                    bounce={0.4}
                    selection
                    labels
                    color="currentColor"
                    align="center"
                    draggable
                    sweep
                />
                <RunHero />
                <p>A cool little thingy</p>
            </div>
            <div className='topo-bg'>
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
                    hoverEffect="light"
                    clickEffect="pulse"
                    majorLines
                    scale={1.9}
                    detail={1}
                    morph={0.2}
                    brightness={0.4}
                    markers="plus"
                    glitch={0}
                    opacity={1}
                    mouseInteraction
                    mouseRadius={0.7}
                    mouseStrength={1}
                    trail={0.3}
                    parallax={0.5}
                    intro
                    lightMode={false}
                />
            </div>
        </div>
    )
}