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
| 3 | A local VLM (Qwen2.5-VL) on the same tasks | next |
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

`python -m aiforme vision-eval` scores it. On three system windows (Notepad, File Explorer, Settings — 102 tasks)
it hits 100%. That number is a sanity check, not a result: the instructions are the very names it reads, so on
these tasks it is close to an upper bound. The first run hit 98%: the window's *«Закрыть»* lost to the tab's
*«Закрыть вкладку»* because *«вкладку»* was dropped from names as well as from instructions.

What it cannot do is the point of step 3: an icon without a name, a canvas, a game, a web page that hides its tree,
and any description that is not the element's name — *«нажми на шестерёнку»*. Commands in my own words, written
against the same snapshots, come with step 3.
