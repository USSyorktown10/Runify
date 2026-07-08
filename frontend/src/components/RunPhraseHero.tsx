import { useState, useEffect, type CSSProperties } from "react";

export type PhraseVariant =
  | "stretch"
  | "larger"
  | "uppercase"
  | "underline"
  | "duplicate"
  | "italic"
  | "heavy"
  | "spaced"
  | "highlight"
  | "runify";

export type RunPhraseConfig = {
  word: string;
  variant: PhraseVariant;
  color: string;
};

export const RUN_PHRASES: RunPhraseConfig[] = [
  { word: "further", variant: "stretch", color: "var(--phrase-further)" },
  { word: "more", variant: "larger", color: "var(--phrase-more)" },
  { word: "efficiently", variant: "uppercase", color: "var(--phrase-efficiently)" },
  { word: "smarter", variant: "underline", color: "var(--phrase-smarter)" },
  { word: "together", variant: "duplicate", color: "var(--phrase-together)" },
  { word: "faster", variant: "italic", color: "var(--phrase-faster)" },
  { word: "stronger", variant: "heavy", color: "var(--phrase-stronger)" },
  { word: "longer", variant: "spaced", color: "var(--phrase-longer)" },
  { word: "happier", variant: "highlight", color: "var(--phrase-happier)" },
  { word: "Runify", variant: "runify", color: "var(--phrase-runify)" },
];

const MARQUEE_ROWS = [
  { duration: 55, reverse: false, opacity: 0.12, size: "text-4xl sm:text-5xl" },
  { duration: 42, reverse: true, opacity: 0.08, size: "text-3xl sm:text-4xl" },
  { duration: 68, reverse: false, opacity: 0.14, size: "text-5xl sm:text-6xl" },
  { duration: 38, reverse: true, opacity: 0.06, size: "text-2xl sm:text-3xl" },
  { duration: 50, reverse: false, opacity: 0.11, size: "text-4xl sm:text-5xl" },
  { duration: 62, reverse: true, opacity: 0.09, size: "text-3xl sm:text-4xl" },
  { duration: 45, reverse: false, opacity: 0.07, size: "text-5xl sm:text-6xl" },
];

function WordSuffix({
  tone,
  className = "",
  wordClassName = "",
  underline = false,
  hidePeriod = false,
  children,
}: {
  tone: CSSProperties;
  className?: string;
  wordClassName?: string;
  underline?: boolean;
  hidePeriod?: boolean;
  children: React.ReactNode;
}) {
  return (
    <span className={`inline-flex items-baseline whitespace-nowrap ${className}`} style={tone}>
      <span
        className={`${underline ? "underline decoration-2 underline-offset-[6px]" : ""} ${wordClassName}`}
        style={underline ? { textDecorationColor: tone.color } : undefined}
      >
        {children}
      </span>
      {!hidePeriod && <span className="tracking-normal">.</span>}
    </span>
  );
}

function StyledWord({
  word,
  variant,
  color,
  muted = false,
  hidePeriod = false,
}: {
  word: string;
  variant: PhraseVariant;
  color: string;
  muted?: boolean;
  hidePeriod?: boolean;
}) {
  const tone: CSSProperties = muted ? { color, opacity: 0.35 } : { color };

  switch (variant) {
    case "stretch":
      return (
        <WordSuffix tone={tone} hidePeriod={hidePeriod}>
          <span className="inline-block origin-left scale-x-125 text-[1.15em] mr-[0.25em]">{word}</span>
        </WordSuffix>
      );
    case "larger":
      return (
        <WordSuffix tone={tone} wordClassName="text-[1.25em] font-bold" hidePeriod={hidePeriod}>
          {word}
        </WordSuffix>
      );
    case "uppercase":
      return (
        <WordSuffix tone={tone} wordClassName="uppercase tracking-[0.18em]" hidePeriod={hidePeriod}>
          {word}
        </WordSuffix>
      );
    case "underline":
      return (
        <WordSuffix tone={tone} underline hidePeriod={hidePeriod}>
          {word}
        </WordSuffix>
      );
    case "duplicate":
      return (
        <span className="inline-flex items-baseline gap-2 whitespace-nowrap" style={tone}>
          <span>{word}</span>
          <span style={muted ? { opacity: 0.57 } : { opacity: 0.55 }}>{word}</span>
          {!hidePeriod && <span className="tracking-normal">.</span>}
        </span>
      );
    case "italic":
      return (
        <WordSuffix tone={tone} wordClassName="italic" hidePeriod={hidePeriod}>
          {word}
        </WordSuffix>
      );
    case "heavy":
      return (
        <WordSuffix tone={tone} className="border-2 border-current px-1 font-extrabold" hidePeriod={hidePeriod}>
          {word}
        </WordSuffix>
      );
    case "spaced":
      return (
        <WordSuffix tone={tone} wordClassName="tracking-[0.35em]" hidePeriod={hidePeriod}>
          {word}
        </WordSuffix>
      );
    case "highlight":
      return (
        <WordSuffix tone={tone} hidePeriod={hidePeriod}>
          {word}
        </WordSuffix>
      );
    case "runify":
      return (
        <WordSuffix tone={tone} hidePeriod={hidePeriod}>
          <span className="inline-block origin-left scale-x-110 text-[1.15em] font-extrabold tracking-tight mr-[0.1em]">
            {word}
          </span>
        </WordSuffix>
      );
  }
}

export function RunPhrase({
  phrase,
  muted = false,
  className = "",
  hidePeriod = false,
}: {
  phrase: RunPhraseConfig;
  muted?: boolean;
  className?: string;
  hidePeriod?: boolean;
}) {
  const plain = muted ? "text-muted" : "text-global-text";

  if (phrase.variant === "runify") {
    return (
      <span className={className}>
        <StyledWord word={phrase.word} variant={phrase.variant} color={phrase.color} muted={muted} hidePeriod={hidePeriod} />
      </span>
    );
  }

  return (
    <span className={className}>
      <span className={plain}>Run </span>
      <StyledWord word={phrase.word} variant={phrase.variant} color={phrase.color} muted={muted} hidePeriod={hidePeriod} />
    </span>
  );
}

function MarqueeRow({
  phrases,
  duration,
  reverse,
  className,
  opacity,
  hidePeriod = true,
}: {
  phrases: RunPhraseConfig[];
  duration: number;
  reverse: boolean;
  className: string;
  opacity: number;
  hidePeriod?: boolean;
}) {
  const track = [...phrases, ...phrases];

  return (
    <div className="overflow-hidden whitespace-nowrap" style={{ opacity }}>
      <div
        className={`landing-marquee inline-flex shrink-0 ${className}`}
        style={
          {
            "--marquee-duration": `${duration}s`,
            animationDirection: reverse ? "reverse" : "normal",
          } as CSSProperties
        }
      >
        {track.map((phrase, i) => (
          <span key={`${phrase.word}-${i}`} className="mx-10 shrink-0 font-semibold">
            <RunPhrase phrase={phrase} muted hidePeriod={hidePeriod} />
          </span>
        ))}
      </div>
    </div>
  );
}

export function RunPhraseMarqueeBackground() {
  return (
    <div className="pointer-events-none absolute inset-0 select-none overflow-hidden" aria-hidden="true">
      {MARQUEE_ROWS.map((row, i) => (
        <div key={i} className="absolute w-full" style={{ top: `${6 + i * 13}%` }}>
          <MarqueeRow
            phrases={RUN_PHRASES}
            duration={row.duration}
            reverse={row.reverse}
            className={row.size}
            opacity={row.opacity}
            hidePeriod={true}
          />
        </div>
      ))}
    </div>
  );
}

export function RunPhraseHeroLines({
  className = "",
  lineClassName = "text-xl sm:text-2xl",
}: {
  className?: string;
  lineClassName?: string;
}) {
  return (
    <div className={`space-y-0.5 ${className}`}>
      {RUN_PHRASES.map((phrase) => (
        <span
          key={phrase.word}
          className={`block font-mono font-semibold leading-tight ${lineClassName}`}
        >
          <RunPhrase phrase={phrase} />
        </span>
      ))}
    </div>
  );
}

export function RunPhraseCycler({
  className = "",
  lineClassName = "text-2xl sm:text-3xl md:text-4xl",
  hidePeriod = false,
}: {
  className?: string;
  lineClassName?: string;
  hidePeriod?: boolean;
}) {
  const [index, setIndex] = useState(0);

  useEffect(() => {
    const interval = setInterval(() => {
      setIndex((prev) => (prev + 1) % RUN_PHRASES.length);
    }, 3000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className={`font-mono font-semibold leading-tight ${className}`}>
      <span key={index} className="inline-block animate-fade-in-up">
        <RunPhrase phrase={RUN_PHRASES[index]} className={lineClassName} hidePeriod={hidePeriod} />
      </span>
    </div>
  );
}

