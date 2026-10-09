import '@/styles/hero.css';
import { useState, useEffect } from 'react';

export type PhraseVariant =
  | "stretch"
  | "larger"
  | ""
  | "underline"
  | "italic"
  | "heavy"
  | "spaced"
  | "highlight"
  | "runify";

export type Phrases = {
  word: string;
  variant: PhraseVariant;
};

export const PHRASES: Phrases[] = [
    {word: "Distance", variant: "stretch"},
    {word: "Performance", variant: "underline"},
    {word: "Stats", variant: "larger"},
    {word: "Racing", variant: "highlight"},
    {word: "Times", variant: "heavy"},
    {word: "Speed", variant: "italic"},
    {word: "Runify", variant: "runify"}
]

function StyledWord({ word, variant }: Phrases) {
    return (
      <span className={`phrase-${variant}`}>
        {word}
      </span>
    );
}

export function RunHero({ muted = false, className = "", interval = 3000 }: { muted?: boolean; className?: string; interval?: number; }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    const timer = setInterval(() => {
      setCurrentIndex((prevIndex) => (prevIndex + 1) % PHRASES.length);
    }, interval);
    return () => clearInterval(timer);
  }, [interval]);

  const activePhrase = PHRASES[currentIndex];
  const plain = muted ? "text-muted" : "text-global-text";
  console.log(plain);

  const isForPrefix = ["stretch", "underline", "highlight", "heavy", "italic"].includes(activePhrase.variant);
  const prefixText = isForPrefix ? "Train for " : "Train with ";

  return (
    <div className={`phrase-cycler ${className}`}>
      <span className={plain}>{prefixText}</span>
      <span key={activePhrase.word} className="animate-fade-in inline-block">
        <StyledWord word={activePhrase.word} variant={activePhrase.variant} />
      </span>
    </div>
  );
}