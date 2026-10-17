import '@/styles/hero.css'; 
import { useState, useEffect, useRef } from 'react'; 

export type PhraseVariant = "runify" | "stats" | ""; 
export type Phrases = { word: string; variant: PhraseVariant; }; 

export const PHRASES: Phrases[] = [ 
  { word: "Distance", variant: "" }, 
  { word: "Performance", variant: "" }, 
  { word: "Stats", variant: "stats" }, 
  { word: "Racing", variant: "" }, 
  { word: "Times", variant: "" }, 
  { word: "Speed", variant: "" }, 
  { word: "Runify", variant: "runify" } 
]; 

function StyledWord({ word, variant, isActive }: Phrases & { isActive: boolean }) { 
  if (variant === "runify") {
    return (
      <span className={`phrase-runify ${isActive ? 'animate-wave glow-active' : ''}`}>
        {word.split("").map((letter, index) => (
          <span 
            key={index} 
            className="runify-letter" 
            style={{ '--letter-index': index } as React.CSSProperties}
          >
            {letter}
          </span>
        ))}
      </span>
    );
  }
  return <span className={variant ? `phrase-${variant}` : ""}>{word}</span>; 
} 

const getPrep = (variant: PhraseVariant) => { 
  return (variant === "runify" || variant === "stats") ? "with" : "for";
}; 

export function RunHero({ muted = false, className = "", interval = 3000 }: { muted?: boolean; className?: string; interval?: number; }) { 
  const [currentIndex, setCurrentIndex] = useState(0); 
  const prevIndexRef = useRef(0); 
  const prevIndex = prevIndexRef.current; 

  const [activeWordWidth, setActiveWordWidth] = useState<number | null>(null);
  const wordRefs = useRef<(HTMLDivElement | null)[]>([]);

  const isRunifyActive = PHRASES[currentIndex].variant === "runify";
  const currentInterval = isRunifyActive ? 5000 : interval; 

  useEffect(() => { 
    const timer = setInterval(() => { 
      prevIndexRef.current = currentIndex; 
      setCurrentIndex((prevIndex) => (prevIndex + 1) % PHRASES.length); 
    }, currentInterval); 
    return () => clearInterval(timer); 
  }, [currentIndex, currentInterval]); 

  useEffect(() => {
    const currentElement = wordRefs.current[currentIndex];
    if (currentElement) {
      const innerSpan = currentElement.querySelector('span');
      if (innerSpan) {
        const targetSpan = currentElement.querySelector('.phrase-runify') || currentElement.querySelector('span');
        setActiveWordWidth(innerSpan.offsetWidth);
      } else {
        setActiveWordWidth(currentElement.offsetWidth);
      }
    }
  }, [currentIndex]);

  const plain = muted ? "text-muted" : "text-global-text"; 
  const currentPrep = getPrep(PHRASES[currentIndex].variant); 
  const prevPrep = getPrep(PHRASES[prevIndex].variant); 
  const prepYTransform = currentPrep === prevPrep ? 0 : currentIndex * 100; 

  const currentPrepWidth = currentPrep === 'for' ? '55px' : '75px';

  return ( 
    <div className="run-hero"> 
      <div className={`phrase-cycler ${className}`}> 
        <div className={`ambient-glow-layer ${isRunifyActive ? 'glow-active' : ''}`} />
        <span className={`${plain} cycler-prefix`}>Train</span> 
        
        <div className="spinner-window" style={{ width: currentPrepWidth, transition: 'width 0.4s ease' }}> 
          <div 
            className="spinner-track" 
            style={{ 
              transform: `translateY(-${prepYTransform}%)`, 
              transition: currentPrep === prevPrep ? 'none' : 'transform 0.6s cubic-bezier(0.175, 0.885, 0.32, 1.15)'
            }} 
          > 
            {PHRASES.map((phrase, idx) => ( 
              <div key={`prep-${idx}`} className="spinner-item text-global-text"> 
                {getPrep(phrase.variant)} 
              </div> 
            ))} 
          </div> 
        </div> 

        <div 
          className="spinner-window" 
          style={{ 
            width: activeWordWidth ? `${activeWordWidth}px` : 'auto', 
            transition: 'width 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.15)' 
          }}
        > 
          <div 
            className="spinner-track" 
            style={{ transform: `translateY(-${currentIndex * 100}%)` }} 
          > 
            {PHRASES.map((phrase, idx) => ( 
              <div 
                key={phrase.word} 
                className="spinner-item"
                ref={(el) => { wordRefs.current[idx] = el; }} 
              > 
                <StyledWord word={phrase.word} variant={phrase.variant} isActive={idx === currentIndex} /> 
              </div> 
            ))} 
          </div> 
        </div> 
      </div> 
    </div> 
  ); 
}
