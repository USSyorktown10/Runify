import { Link } from "react-router-dom";
import { RunPhraseCycler, RunPhraseMarqueeBackground } from "@/components/RunPhraseHero";

export function LandingPage() {
  return (
    <section className="relative flex min-h-[calc(100vh-8rem)] w-full flex-col justify-center overflow-hidden">
      <RunPhraseMarqueeBackground />

      <div className="absolute inset-y-0 start-0 z-[1] w-1 bg-accent" aria-hidden="true" />

      <div className="relative z-10 mx-auto w-full max-w-4xl px-4 py-8 sm:px-8 lg:px-12">
        <img src="/logo.svg" alt="" className="mb-4 h-14 w-14 border border-border sm:h-16 sm:w-16" />

        <h1 className="mb-4 min-h-[3rem] flex items-center">
          <RunPhraseCycler lineClassName="text-3xl sm:text-4xl lg:text-5xl" />
        </h1>

        <p className="prose-runify mb-4 max-w-lg text-muted">
            Strava without the paywall. Runify is a free and open-source alternative to Strava that allows you to track your runs, analyze your performance, and connect with other runners. Accounts are always free, and creating one is really easy!
        </p>

        <div className="flex flex-wrap gap-2">
          <Link className="btn-primary px-4 py-2" to="/signup">
            Get started
          </Link>
          <Link className="btn-secondary px-4 py-2" to="/login">
            Log in
          </Link>
        </div>
      </div>
    </section>
  );
}
