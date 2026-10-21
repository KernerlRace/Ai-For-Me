# AI For Me

A personal AI layer for my own machine: a voice-controlled orchestrator that also acts on its own — switching the environment between coding and gaming, watching the system for anomalies, turning calendar shifts and receipt screenshots into a finance tracker, and giving a screen-reading agent the ability to act in any application instead of a fixed command set.

Status: planning stage. The architecture, the ML approach for each module and the phased build order are written up in [ROADMAP.md](ROADMAP.md); implementation starts with the voice orchestrator (Phase 1).

## Why

Most "AI assistant" side projects are a thin wrapper around a single LLM call. This one is a set of independent ML subsystems wired together by a shared event bus and a common memory — intent classification with an active-learning feedback loop, a vision-language agent that acts from screenshots, OCR and categorization for expense tracking, anomaly detection for system security, a knowledge graph for long-term memory — each with its own baseline, its own evaluation, and a reason to exist on its own.

## Planned modules

| Module | What it will do |
|---|---|
| `voice-orchestrator` | Wake word + STT + intent classification → executes commands on the machine |
| `mode-autopilot` | Detects context (coding / gaming / meeting via calendar) and switches the environment |
| `vision-agent` | Screenshot + a vision-language model → acts in any application, not just pre-defined intents |
| `active-learning` | Logs misclassified commands, retrains the intent classifier, tracks accuracy over time |
| `security-watchdog` | Anomaly detection over running processes and network connections |
| `password-vault` | Local encrypted password store with breach checks |
| `finance-tracker` | Work shifts in the calendar → pay/tax calculation and savings goals; receipt screenshots → OCR'd, categorized expenses |
| `learning-tracker` | Builds a topic graph from what I study (course material, tutorials), generates spaced-repetition flashcards |
| `memory-graph` | Long-term memory as a knowledge graph (projects, people, habits), shared across modules |
| `universal-search` | Semantic search across code, notes, clipboard history and learning material at once |
| `dashboard` | Shows what the assistant did and why, plus finance/learning/security widgets |
| `mobile-companion` | Push notifications and remote approval when away from the machine |
| `resume-builder` | Drafts resume bullet points from commit history and finished modules |

## Tech stack (planned)

Python · faster-whisper · HuggingFace Transformers · a vision-language model for the screen agent · Tesseract/PaddleOCR · Chroma · SQLite · FastAPI · React

## Project structure

| Path | Purpose |
|---|---|
| `ROADMAP.md` | Full plan: ML approach per module, architecture, tech stack, phased build order |
| `modules/` | One folder per module, built out as each phase starts |
| `.env.example` | Required environment variables — copy to `.env` |

## Roadmap

Ten phases, from a voice-command MVP through the vision agent, finance tracking and the security watchdog to a knowledge-graph memory, a dashboard and a mobile companion. Full detail, including the ML approach for every module, is in [ROADMAP.md](ROADMAP.md).
