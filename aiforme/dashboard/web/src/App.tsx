import { CalendarPage } from "./CalendarPage";

/** One page for now; the action log, modes and finance get their own as their phases land. */
export function App() {
  return (
    <div className="app">
      <header className="topbar">
        <span className="logo" aria-hidden>◆</span>
        <span className="name">Амадеус</span>
        <nav>
          <a className="active" href="/">Календарь</a>
        </nav>
      </header>
      <main>
        <CalendarPage />
      </main>
    </div>
  );
}
