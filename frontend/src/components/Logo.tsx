import { Link } from "react-router-dom";

type LogoProps = {
  showWordmark?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
  to?: string;
};

const sizes = {
  sm: { img: "h-8 w-8", text: "text-lg" },
  md: { img: "h-10 w-10", text: "text-xl" },
  lg: { img: "h-16 w-16", text: "text-3xl" },
};

export function Logo({ showWordmark = true, size = "md", className = "", to = "/feed" }: LogoProps) {
  const s = sizes[size];

  return (
    <Link to={to} className={`inline-flex items-center gap-3 shrink-0 ${className}`}>
      <img src="/logo.svg" alt="" className={`${s.img}`} />
      {showWordmark && (
        <span className={`${s.text} font-mono font-semibold`}>
          Runify
        </span>
      )}
    </Link>
  );
}
