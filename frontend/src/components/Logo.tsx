import { Link } from "react-router-dom";
import '@/styles/logo.css';

type LogoProps = {
  showWordmark?: boolean;
  size?: "sm" | "md" | "lg";
  className?: string;
  to?: string;
};


export function Logo({ showWordmark = true, size = "md", className = "", to = "/feed" }: LogoProps) {
  return (
    <Link to={to} className={`${className} link`}>
      <img src="/logo.svg" alt="" className={`${size}`} />
      {showWordmark && (
        <span className={`${size} logo`}>
          Runify
        </span>
      )}
    </Link>
  );
}