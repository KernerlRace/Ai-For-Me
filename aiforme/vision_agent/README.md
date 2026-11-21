# Vision Agent

Phase 4 of [AI For Me](../../README.md): the assistant finds an element on the screen from a description —
*«нажми на кнопку сохранить»* — and clicks it, in any application, without a command written for that application.

The model runs locally. It is reached over HTTP, so moving it from this PC's RTX 3060 Ti to a separate inference
server with a larger model will only change one address.

## Where it stands

| Step | What | Status |
|---|---|---|
| 1 | A grounding dataset from my own windows, labelled by UI Automation | done |
| 2 | The baseline: UI Automation names, no ML | done |
| 3 | A local VLM (Qwen2.5-VL) on the same tasks | done: 62.7% on commands in my own words, where UI Automation scores 0% |
| 4 | The agent on the bus: command → find → click | |
| 5 | Multi-step tasks, on a larger model | later |

## The dataset: labels for free

`python -m aiforme snapshot --count 5` saves the window in front five times, ten seconds apart: a PNG of it and
every element Windows' [UI Automation](https://learn.microsoft.com/windows/win32/winauto/entry-uiauto-win32)
reports in it — name, control type, box in the image's pixels. UI Automation knows exactly where each button is, so
every box is a ground-truth label that took no hand labelling.

A **task** is one element a command could name: clickable (a button, tab, list item, menu item, …), enabled, wholly
in view, with a name of at most 60 characters that no other element in the window shares — with two *«Закрыть»*
buttons an instruction cannot say which. Its instruction is the element's own name; a click scores if it lands
inside the box.

Everything stays in `data/vision/` — snapshots, images and reports — which git ignores: the names are my chats,
files and tabs.

**What the boxes get wrong.** The process is made per-monitor DPI aware before anything is read, or at 125%
scaling the boxes and the image would disagree by a quarter. Chromium and Electron apps (Chrome, Discord, the
Claude app) expose only their frame until an accessibility client asks for more, so for them the dataset has
few tasks — and those are exactly the windows where a model that reads pixels is needed.

## The baseline

`UiaGrounder` matches the instruction against the names UI Automation reports: the share of words in common, words
like *«кнопку»* or *"button"* dropped from the instruction when that helps, and no click under half a match — a
wrong click costs more than "I couldn't find it".

`python -m aiforme vision-eval` scores it. On seven system windows (Notepad, File Explorer, Calculator, four pages of
Settings — 203 tasks) it hits 100%. That number is a sanity check, not a result: the instructions are the very names it reads, so on
these tasks it is close to an upper bound. The first run hit 98%: the window's *«Закрыть»* lost to the tab's
*«Закрыть вкладку»* because *«вкладку»* was dropped from names as well as from instructions.

What it cannot do is the point of step 3: an icon without a name, a canvas, a game, a web page that hides its tree,
and any description that is not the element's name — *«нажми на шестерёнку»*. So the dataset has a second task set.

**Described tasks.** `data/vision/described.tsv` holds 59 commands in my own words: *«тумблер авиарежима»*,
*«крестик на активной вкладке»*, *«стереть последнюю цифру»*. Each points at one element of a snapshot. I wrote them
by looking at each screenshot, before any model was run on them. On these, UI Automation scores 0%: it gives no
answer to 58 of them and clicks the wrong element for one.

## The VLM grounder

`VlmGrounder` ([`vlm.py`](vlm.py)) sends the screenshot and the instruction to Qwen2.5-VL through Ollama:

- **What it asks for.** The element's box as `bbox_2d`, the format the model was trained to ground in. The click is
  the box's centre.
- **Resizing.** The model sees an image as 28×28 patches. The screenshot is resized on this side to the size the
  model would pick itself: both sides multiples of 28, about a megapixel at most. The answer is then scaled back.
  So the coordinates do not depend on how a given server version would have resized the image.
- **The server's own resizing, measured.** Ollama 0.40 still resizes some images itself. A 1904×364 File Explorer
  window becomes about 2072×392, and the model answers in those pixels. So before the first command on each image
  size, the grounder sends a dark image of that size with a red square at a known place. The ratio between where
  the model puts the square and where it is gives the server's frame. On the other sizes here it is exactly 1.
- **An answer every time** (prompt v2). The first prompt (v1) let the model answer `[]`, "not visible". See below
  for why that was dropped. An empty answer or a box off the image is still no click. A server that cannot be
  reached is an error, not a miss.
- **Where the server is.** `AIFORME_VLM_URL` sets the address, default `http://127.0.0.1:11434`. It is not
  `localhost`: on Windows that tries IPv6 first, and a server listening on IPv4 only then answers late. Moving the
  model to the inference server is a change of this one variable.

Two grounders build on it:

- **`SnapToElement`.** A click that lands on no clickable element is moved to the nearest one, when the window has
  an accessibility tree. A click already on an element is left alone, so this can fix a near miss but never undo a
  hit.
- **`FirstAnswer`**, the hybrid. UI Automation answers first, when an instruction matches an element's name. The
  model looks at the pixels only when it declines.

```bash
ollama pull qwen2.5vl:7b
uv run python -m aiforme vision-eval --grounder vlm --limit 30         # a quick, even sample
uv run python -m aiforme vision-eval --grounder vlm --tasks described  # my own words
uv run python -m aiforme vision-eval --grounder hybrid                 # also: vlm-snap, --prompt v1
uv run python -m aiforme.vision_agent.compare A.json B.json            # paired: Wilson intervals, McNemar
```

Every task's raw model answer is saved in the report next to its click, so each miss can be read.

## Results

Qwen2.5-VL 7B (Q4_K_M, Ollama 0.40.16) on the RTX 3060 Ti, greedy decoding. 203 literal tasks and 59 described
ones on 7 windows. Brackets are 95% Wilson intervals.

| Grounder | Literal: the element's name | Described: my own words | Wrong clicks, described | Median time |
|---|---|---|---|---|
| UI Automation names (baseline) | **100%** [98.1, 100] | 0% [0, 6.1] | 1.7% | < 1 ms |
| VLM, prompt v1 (may answer "not visible") | 37.9% [31.5, 44.8] | 30.5% [20.3, 43.1] | 25.4% | 0.66 s |
| + the server's frame measured | 38.9% [32.5, 45.8] | 33.9% [23.1, 46.6] | 23.7% | 0.71 s |
| VLM, prompt v2 (always a box) | 52.2% [45.4, 59.0] | 55.9% [43.3, 67.8] | 40.7% | 0.80 s |
| + click snapped to the nearest element | 57.1% [50.3, 63.8] | **62.7%** [50.0, 73.9] | 33.9% | 0.82 s |
| **Hybrid**: UI Automation, then the VLM + snap | **100%** [98.1, 100] | 61.0% [48.3, 72.4] | 35.6% | < 1 ms / 0.87 s |

Each step is checked against the one before it on the same tasks, with McNemar's exact test (tasks only the
first got right : tasks only the second got right):

- **v1 → v2:** 1 : 28 on literal tasks (p = 1·10⁻⁷) and 0 : 13 on described ones (p = 0.0002). This is the
  largest effect.
- **Snapping:** 0 : 10 on literal tasks (p = 0.002) and 0 : 4 on described ones (p = 0.12). It never lost a
  hit.
- **Measuring the frame:** 2 : 4 and 0 : 2, not significant overall. It touches one window of seven. On File
  Explorer alone it doubled the hits, from 3 to 6 of 26 literal tasks and from 1 to 3 of 10 described ones.

What the runs taught me:

- **The model declined to answer far more than it failed to see.** With v1 it answered "not visible" on 43% of
  literal tasks, including *«Настройки»* and *«Развернуть»*, which are plainly on screen. It even declined on a
  red square on a blank 2560×1392 image. With the escape removed, accuracy rose by 13 points on literal tasks
  and 22 on described ones.
  - The price is in the last column: a wrong click instead of no click, 41% of the time.
  - The refusals were not noise, though. v2 hits 28% of the literal tasks v1 declined and 70% of those it answered;
    on described tasks the split is 44% against 65%. A refusal is a weak confidence signal, and the plan below uses
    it.
- **Coordinates need a check, not trust.** On the first run every File Explorer click drifted 9% to the right. The
  cause was not the model: Ollama resizes a wide, thin image on its own. A red square at a known position showed it
  in one call. It now runs once per image size, as part of the grounder.
- **Literal tasks are harder for the model than my own words: 57% against 63%.** An element's accessibility name
  is often not written on the screen:
  - a Notepad tab is named *«requirements.txt. Не изменено.»*;
  - scroll-bar arrows are *«Строка вверх»*;
  - a picture is *«Аватар пользователя»*.

  Notepad's literal tasks are 14% hits even with v2, and most of them are names like these. UI Automation reads
  these names directly, which is why the hybrid gives each source the tasks it is good at.
- **Where it still misses (v2, described):**
  - *The window's corner controls*: close and maximize, at the very edge. The model aims short of the edge; that is
    5 of the 26 misses.
  - *Toggles*: the click lands on the label next to the switch. Snapping fixes some of these.
  - *Elements described by position or purpose*: *«стереть последнюю цифру»*, *«ссылка на микрофоны внизу
    страницы»*.

**The limits.**
- 59 described tasks is a small set: the interval around 62.7% is 24 points wide.
- All seven windows are Windows system apps, which have good accessibility trees. Snapping and the hybrid lean on
  that tree. On a web page, a game or a canvas they fall back to the bare model, at 56%.
- Both task sets were written before any model was run. Prompt v2 and snapping were then chosen after reading v1's
  misses on these same tasks. Their gains are measured where they were found, so a fresh set of described tasks
  will be the honest check.

**Next.**
- A wrong click is worse than none, so a 34% wrong-click rate is the problem to solve before the agent is let loose.
  Two candidates, both measurable on the saved runs plus one extra call:
  - ask with both prompts and click only when v1 does not decline. On the saved runs this would cut wrong clicks
    on described tasks from 34% to 14%, but hits would fall from 63% to 44%;
  - a second look: crop around the click and ask the model whether that is the element it meant.
- After that, a fresh set of described tasks, on Chromium windows too.
