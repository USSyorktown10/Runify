import { Link, Outlet } from "react-router-dom";
import { Footer } from "@/components/Footer";
import { Header } from "@/components/Header";
import { Logo } from "@/components/Logo";
import { RunPhraseCycler, RunPhraseMarqueeBackground } from "@/components/RunPhraseHero";

export function AppShell() {
  return (
    <div className="flex min-h-screen w-full flex-col">
      <a className="sr-only focus:not-sr-only focus:fixed focus:start-4 focus:top-4 focus:z-[100] focus:rounded-none focus:border focus:border-border focus:bg-global-bg focus:px-3 focus:py-2" href="#main">
        Skip to content
      </a>
      <Header />
      <main id="main" className="flex-1 w-full px-4 py-3 sm:px-6 lg:px-8">
        <div className="mx-auto w-full max-w-[1600px]">
          <Outlet />
        </div>
      </main>
      <Footer />
    </div>
  );
}

export function AuthLayout() {
  return (
    <div className="relative grid min-h-screen w-full overflow-hidden lg:grid-cols-2">
      <div className="relative hidden overflow-hidden border-e-2 border-border bg-surface lg:flex lg:flex-col lg:justify-between lg:p-8">
        <RunPhraseMarqueeBackground />
        <div className="absolute inset-y-0 start-0 z-[2] w-1 bg-accent" aria-hidden="true" />

        <img src="/logo.svg" alt="" className="relative z-10 h-14 w-14" />
        <div className="relative z-10">
          <h1 className="mb-3 min-h-[2.5rem] flex items-center">
            <RunPhraseCycler lineClassName="text-lg sm:text-xl" />
          </h1>
          <p className="prose-runify max-w-md text-muted">Strava without the paywall.</p>
        </div>
        <p className="relative z-10 text-muted text-xs">© Runify {new Date().getFullYear()}</p>
      </div>

      <div className="relative flex flex-col justify-center px-6 py-12 sm:px-12 lg:px-16">
        <RunPhraseMarqueeBackground />
        <div className="absolute inset-y-0 start-0 z-[1] w-1 bg-accent lg:hidden" aria-hidden="true" />

        <div className="relative z-10 mx-auto w-full max-w-md border border-border bg-global-bg p-6 lg:border-0 lg:bg-transparent lg:p-0">
          <div className="mb-6 lg:hidden">
            <div className="mb-4 flex items-center gap-3">
              <img src="/logo.svg" alt="" className="h-10 w-10" />
              <span className="text-xl font-mono font-semibold">Runify</span>
            </div>
            <div className="min-h-[1.5rem] flex items-center">
              <RunPhraseCycler lineClassName="text-sm" />
            </div>
          </div>
          <Outlet />
        </div>
      </div>
    </div>
  );
}

export function PublicLayout() {
  return (
    <div className="flex min-h-screen w-full flex-col">
      <header className="sticky top-0 z-50 w-full border-b-2 border-border bg-global-bg">
        <div className="mx-auto flex h-16 w-full max-w-[1600px] items-center justify-between px-4 sm:px-6 lg:px-8">
          <Logo size="sm" to="/" />
          <nav className="flex items-center gap-3 text-sm font-semibold">
            <Link className="text-muted hover:text-accent" to="/login">
              Log in
            </Link>
            <Link className="btn-primary text-sm" to="/signup">
              Sign up
            </Link>
          </nav>
        </div>
      </header>
      <main className="flex-1 w-full">
        <Outlet />
      </main>
      <Footer />
    </div>
  );
}
