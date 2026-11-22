import { CalendarPage } from "./CalendarPage";
import { CommandsPage } from "./CommandsPage";
import { ModeBadge } from "./ModeBadge";
import { lang, setLang, t } from "./i18n";

const PAGES = [
  { path: "/", title: t.pages.calendar, page: CalendarPage },
  { path: "/commands", title: t.pages.commands, page: CommandsPage },
];

/** A page per path; the server answers every path with this app, so plain links are enough for now. */
export function App() {
  const current = PAGES.find((p) => p.path === window.location.pathname) ?? PAGES[0];
  const Page = current.page;
  return (
    <div className="app">
      <header className="topbar">
        <span className="logo" aria-hidden>◆</span>
        <span className="name">{t.assistant}</span>
        <nav>
          {PAGES.map((p) => (
            <a key={p.path} className={p === current ? "active" : ""} href={p.path + window.location.search}>{p.title}</a>
          ))}
        </nav>
        <span className="spacer" />
        <ModeBadge />
        <div className="toggle lang" role="group" aria-label="Language">
          {(["en", "ru"] as const).map((l) => (
            <button key={l} className={l === lang ? "active" : ""} aria-pressed={l === lang}
                    onClick={() => l !== lang && setLang(l)}>{l.toUpperCase()}</button>
          ))}
        </div>
      </header>
      <main>
        <Page />
      </main>
    </div>
  );
}
