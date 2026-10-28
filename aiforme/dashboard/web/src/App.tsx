import { CalendarPage } from "./CalendarPage";
import { CommandsPage } from "./CommandsPage";
import { ModeBadge } from "./ModeBadge";

const PAGES = [
  { path: "/", title: "Календарь", page: CalendarPage },
  { path: "/commands", title: "Команды", page: CommandsPage },
];

/** A page per path; the server answers every path with this app, so plain links are enough for now. */
export function App() {
  const current = PAGES.find((p) => p.path === window.location.pathname) ?? PAGES[0];
  const Page = current.page;
  return (
    <div className="app">
      <header className="topbar">
        <span className="logo" aria-hidden>◆</span>
        <span className="name">Амадеус</span>
        <nav>
          {PAGES.map((p) => (
            <a key={p.path} className={p === current ? "active" : ""} href={p.path}>{p.title}</a>
          ))}
        </nav>
        <span className="spacer" />
        <ModeBadge />
      </header>
      <main>
        <Page />
      </main>
    </div>
  );
}
