import '@/styles/hero.css';
import { useState, useEffect } from 'react';

export type PhraseVariant = "stretch" | "larger" | "" | "underline" | "italic" | "heavy" | "spaced" | "highlight" | "runify";

export type Phrases = { word: string; variant: PhraseVariant; };

export const PHRASES: Phrases[] = [
  { word: "Distance", variant: "stretch" },
  { word: "Performance", variant: "underline" },
  { word: "Stats", variant: "larger" },
  { word: "Racing", variant: "highlight" },
  { word: "Times", variant: "heavy" },
  { word: "Speed", variant: "italic" },
  { word: "Runify", variant: "runify" }
];

function StyledWord({ word, variant }: Phrases) {
  return <span className={`phrase-${variant}`}>{word}</span>;
}

const getPrefix = (variant: PhraseVariant) => {
  const isForPrefix = ["stretch", "underline", "highlight", "heavy", "italic"].includes(variant);
  return isForPrefix ? "for" : "with";
};

const getPrep = (variant: PhraseVariant) => {
  const isForPrefix = ["stretch", "underline", "highlight", "heavy", "italic"].includes(variant);
  return isForPrefix ? "for" : "with";
};

export function RunHero({ muted = false, className = "", interval = 3000 }: { muted?: boolean; className?: string; interval?: number; }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  const prevIndexRef = useRef(0);
  const prevIndex = prevIndexRef.current;

  useEffect(() => {
    const timer = setInterval(() => {
      prevIndexRef.current = currentIndex;
      setCurrentIndex((prevIndex) => (prevIndex + 1) % PHRASES.length);
    }, interval);
    return () => clearInterval(timer);
  }, [currentIndex, interval]);

  const plain = muted ? "text-muted" : "text-global-text";
  
  const currentPrep = getPrep(PHRASES[currentIndex].variant);
  const prevPrep = getPrep(PHRASES[prevIndex].variant);
  
  const prepYTransform = currentPrep === prevPrep ? 0 : currentIndex * 100;

  return (
    <div className="run-hero">
      <div className={`phrase-cycler ${className}`}>
        
        {/* 1. Static Prefix: Locked in place */}
        <span className={`${plain} cycler-prefix`}>Train</span> 

        {/* 2. Preposition Track: Animates ONLY when transitioning between for/with */}
        <div className="spinner-window">
          <div 
            className="spinner-track" 
            style={{ 
              transform: `translateY(-${prepYTransform}%)`,
              // Kills the transition snap effect completely when the preposition isn't changing
              transition: currentPrep === prevPrep ? 'none' : 'transform 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.15)' 
            }}
          >
            {PHRASES.map((phrase, idx) => (
              <div key={`prep-${idx}`} className="spinner-item text-global-text" style={{ marginRight: '0.5rem' }}>
                {getPrep(phrase.variant)}
              </div>
            ))}
          </div>
        </div>

        {/* 3. Word Track: Constantly spins on every interval loop */}
        <div className="spinner-window">
          <div 
            className="spinner-track" 
            style={{ transform: `translateY(-${currentIndex * 100}%)` }}
          >
            {PHRASES.map((phrase) => (
              <div key={phrase.word} className="spinner-item">
                <StyledWord word={phrase.word} variant={phrase.variant} />
              </div>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
