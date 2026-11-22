// The dashboard's two languages. English is the default, so anyone can read it; Russian is what the assistant is
// spoken to in. The choice comes from ?lang=ru|en in the address (and is remembered), else from the last choice.

export type Lang = "en" | "ru";

const KEY = "aiforme.lang";

function pick(): Lang {
  const asked = new URLSearchParams(window.location.search).get("lang");
  if (asked === "en" || asked === "ru") {
    try { localStorage.setItem(KEY, asked); } catch { /* private window: the address still decides */ }
    return asked;
  }
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === "en" || saved === "ru") return saved;
  } catch { /* storage blocked: fall back to the default */ }
  return "en";
}

export const lang: Lang = pick();

/** Switch and reload: every string is read once at render, so a reload is the simplest correct switch. */
export function setLang(next: Lang) {
  try { localStorage.setItem(KEY, next); } catch { /* ignore */ }
  const url = new URL(window.location.href);
  url.searchParams.set("lang", next);
  window.location.href = url.toString();
}

/** «1 метку», «2 метки», «5 меток». */
function ruPlural(n: number, one: string, few: string, many: string) {
  const d = Math.abs(n) % 100, u = d % 10;
  if (d > 10 && d < 20) return many;
  return u === 1 ? one : u >= 2 && u <= 4 ? few : many;
}

const en = {
  locale: "en-GB",
  quote: (s: string) => `“${s}”`,
  assistant: "Amadeus",
  pages: { calendar: "Calendar", commands: "Commands" },
  modes: { normal: "Normal mode", coding: "Coding", gaming: "Gaming", meeting: "Meeting" } as Record<string, string>,
  kinds: { meeting: "Meeting", shift: "Shift", other: "Other" },
  intents: {
    open_app: "Open app", close_app: "Close app", find_file: "Find file", dictate: "Dictate text",
    web_search: "Web search", volume_up: "Volume up", volume_down: "Volume down", mute: "Mute",
    media_play_pause: "Play / pause", media_next: "Next track", screenshot: "Screenshot", lock_screen: "Lock screen",
    tell_time: "What time is it", cancel: "Cancel", set_mode: "Switch mode", agenda: "Today's plans",
    out_of_scope: "Not a command",
  } as Record<string, string>,
  slots: { app: "App", query: "Search for", text: "Text", mode: "Mode" } as Record<string, string>,
  calendar: {
    loadFailed: (e: string) => `Could not load the calendar: ${e}`,
    moveFailed: (e: string) => `Could not move it: ${e}`,
  },
  dialog: {
    newEvent: "New event", event: "Event", title: "Title", titlePlaceholder: "e.g. call with the team",
    kind: "Type", start: "Start", end: "End", notes: "Notes",
    shiftNotes: "Rate, place: the finance tracker will use it",
    remove: "Delete", cancel: "Cancel", create: "Create", save: "Save",
  },
  chart: {
    title: "Accuracy by version", test: "Test phrases", live: "My commands", aria: "Accuracy by model version",
    empty: "The chart appears once there are labels or a first retraining.",
    version: (v: number) => `Version ${v}`, outOf: (a: number, b: number) => `${a} of ${b}`,
  },
  commands: {
    loadFailed: (e: string) => `Could not load the commands: ${e}`,
    heading: "Commands", order: "Order of commands", uncertain: "Uncertain first", recent: "Newest first",
    hint: "At the top: unlabelled commands where the model hesitated, its confidence right at the threshold or two " +
      "answers almost tied. A label on these teaches it the most.",
    empty: "Nothing yet: say something to the assistant or send",
    emptyExample: 'python -m aiforme say "open telegram"',
    looksLike: (name: string) => `, looks like “${name}”`,
    or: (name: string, p: number) => `, or “${name}” ${p}%`,
    right: "✓ Right", fix: "✎ Fix", rightDone: "✓ right", fixedDone: "✎ fixed", undo: "Undo",
    inVersion: (v: number) => `in version ${v}`,
    notSaved: (e: string) => `Not saved: ${e}`,
    whatWasIt: "What it was", slotWords: (slot: string) => `${slot}: words from the command`,
    cancel: "Cancel", save: "Save",
  },
  model: {
    heading: "Model", current: (embedder: string) => `current version · ${embedder}`,
    rightOnMine: (c: number, n: number) => `right on my commands (${c} of ${n})`, rightOnMineBare: "right on my commands",
    newLabels: (ok: number, fixed: number) => `new labels: ${ok} right, ${fixed} fixed`,
    retrain: "Retrain", retraining: "Retraining…", needLabel: "Needs at least one new label.",
    auto: (left: number, done: number, every: number) =>
      `Retrains on its own after ${left} more ${left === 1 ? "label" : "labels"} (${done} of ${every} since last time).`,
    failed: (e: string) => `Retraining failed: ${e}`,
    automatic: " (on its own, by label count)",
    lost: (n: number | undefined) => `loses ${n} test phrases (at most one allowed)`,
    worse: (a: string, b: string) => `worse on my commands: ${a} → ${b}`,
    accepted: (id: number, how: string, labels: number, t0: string, t1: string, r0: string, r1: string) =>
      `✓ Version ${id} accepted${how}: ${labels} labels, test phrases ${t0} → ${t1}, my commands ${r0} → ${r1}.`,
    rejected: (id: number, how: string, reasons: string) =>
      `✕ Version ${id} rejected${how}: ${reasons}. The labels are kept: check the fixes and undo the wrong ones.`,
  },
  history: {
    heading: "Retraining history",
    columns: ["Version", "When", "Outcome", "New labels", "Test phrases", "My commands"],
    initial: "initial", accepted: "accepted", rejected: "rejected",
    note: "“My commands” are this version's new labels: before is the old model's accuracy on them, after is the " +
      "new one's, checked on them out of training (by folds). Test phrases never go into training.",
  },
};

export type Strings = typeof en;

const ru: Strings = {
  locale: "ru-RU",
  quote: (s) => `«${s}»`,
  assistant: "Амадеус",
  pages: { calendar: "Календарь", commands: "Команды" },
  modes: { normal: "Обычный режим", coding: "Кодинг", gaming: "Игровой режим", meeting: "Встреча" },
  kinds: { meeting: "Встреча", shift: "Смена", other: "Другое" },
  intents: {
    open_app: "Открыть приложение", close_app: "Закрыть приложение", find_file: "Найти файл",
    dictate: "Надиктовать текст", web_search: "Поиск в интернете", volume_up: "Громче", volume_down: "Тише",
    mute: "Без звука", media_play_pause: "Пауза / продолжить", media_next: "Следующий трек",
    screenshot: "Скриншот", lock_screen: "Заблокировать экран", tell_time: "Который час", cancel: "Отмена",
    set_mode: "Сменить режим", agenda: "Планы на сегодня", out_of_scope: "Не команда",
  },
  slots: { app: "Приложение", query: "Что искать", text: "Текст", mode: "Режим" },
  calendar: {
    loadFailed: (e) => `Не удалось загрузить календарь: ${e}`,
    moveFailed: (e) => `Не удалось перенести: ${e}`,
  },
  dialog: {
    newEvent: "Новое событие", event: "Событие", title: "Название", titlePlaceholder: "Например, созвон с командой",
    kind: "Тип", start: "Начало", end: "Конец", notes: "Заметки",
    shiftNotes: "Ставка, место — пригодится финансовому трекеру",
    remove: "Удалить", cancel: "Отмена", create: "Создать", save: "Сохранить",
  },
  chart: {
    title: "Точность по версиям", test: "Тестовые фразы", live: "Мои команды", aria: "Точность по версиям модели",
    empty: "График появится, когда будут метки или первое переобучение.",
    version: (v) => `Версия ${v}`, outOf: (a, b) => `${a} из ${b}`,
  },
  commands: {
    loadFailed: (e) => `Не удалось загрузить команды: ${e}`,
    heading: "Команды", order: "Порядок команд", uncertain: "Сначала спорные", recent: "По времени",
    hint: "Сверху — неразмеченные команды, где модель колебалась: уверенность у самого порога или два варианта " +
      "почти поровну. Метка на них учит её больше всего.",
    empty: "Пока пусто: скажите ассистенту что-нибудь или отправьте",
    emptyExample: 'python -m aiforme say "открой телеграм"',
    looksLike: (name) => `, похоже на «${name}»`,
    or: (name, p) => `, или «${name}» ${p}%`,
    right: "✓ Верно", fix: "✎ Исправить", rightDone: "✓ верно", fixedDone: "✎ исправлено", undo: "Отменить",
    inVersion: (v) => `в версии ${v}`,
    notSaved: (e) => `Не сохранилось: ${e}`,
    whatWasIt: "Что это было", slotWords: (slot) => `${slot} — слова из команды`,
    cancel: "Отмена", save: "Сохранить",
  },
  model: {
    heading: "Модель", current: (embedder) => `текущая версия · ${embedder}`,
    rightOnMine: (c, n) => `верно на моих командах (${c} из ${n})`, rightOnMineBare: "верно на моих командах",
    newLabels: (ok, fixed) => `новых меток: ${ok} верно, ${fixed} исправлено`,
    retrain: "Переобучить", retraining: "Переобучаю…", needLabel: "Нужна хотя бы одна новая метка.",
    auto: (left, done, every) =>
      `Само переобучится через ${left} ${ruPlural(left, "метку", "метки", "меток")} (${done} из ${every} с прошлого раза).`,
    failed: (e) => `Переобучение упало: ${e}`,
    automatic: " (само, по числу меток)",
    lost: (n) => `теряет ${n} тестовых фраз (можно не больше одной)`,
    worse: (a, b) => `на моих командах хуже: ${a} → ${b}`,
    accepted: (id, how, labels, t0, t1, r0, r1) =>
      `✓ Версия ${id} принята${how}: ${labels} меток, тестовые фразы ${t0} → ${t1}, мои команды ${r0} → ${r1}.`,
    rejected: (id, how, reasons) =>
      `✕ Версия ${id} отклонена${how}: ${reasons}. Метки остались — проверьте исправления и отмените ошибочные.`,
  },
  history: {
    heading: "История переобучений",
    columns: ["Версия", "Когда", "Итог", "Новых меток", "Тестовые фразы", "Мои команды"],
    initial: "исходная", accepted: "принята", rejected: "отклонена",
    note: "«Мои команды» — новые метки этой версии: до — точность прежней модели, после — новой, проверенной на них " +
      "вне обучения (по фолдам). Тестовые фразы в обучение не попадают никогда.",
  },
};

export const t: Strings = lang === "ru" ? ru : en;
