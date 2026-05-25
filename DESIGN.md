# Yammer — Design Document

## Concept

Yammer is a language learning app that teaches entirely through listening and speaking — no translation UI, no flashcard decks, no reading exercises. The learner acquires language the way children do: through immersive audio interaction.

---

## Core Principles

- **Immersion-first:** The app communicates with the user in the target language from the very first session.
- **Comprehensible input:** At least 90–95% of vocabulary used in any session is already known to the learner. New words are introduced using known vocabulary.
- **Spaced repetition via conversation:** Vocabulary is reviewed naturally through dialogue, not rote drilling.
- **Adaptive pacing:** Speaking speed adjusts based on observed comprehension over time.

---

## Key Features

### 1. Bootstrap Phrases (Meta-Language Learning)
The very first phrases taught are the tools for learning more language — in the target language itself.

Examples (for French):
- "Comment dit-on ce mot en français?" *(How do you say this word in French?)*
- "Comment dit-on cette phrase en français?" *(How do you say this phrase in French?)*
- "Que signifie ce mot en anglais?" *(What does this word mean in English?)*

This gives the learner agency to drive their own learning from day one.

### 2. CEFR Alignment
For European languages, vocabulary and grammar targets are aligned to CEFR levels:

| Level | Description |
|-------|-------------|
| A1 | Beginner |
| A2 | Elementary |
| B1 | Intermediate |
| B2 | Upper Intermediate |
| C1 | Advanced |
| C2 | Mastery |

The app tracks which CEFR level the learner is operating at and uses this to guide vocabulary introduction pacing.

### 3. Vocabulary Tracker
Each word/phrase the learner has encountered is stored with:
- **Known confidence score** — how reliably the learner recognizes and produces the word
- **Last reviewed** — timestamp of most recent encounter
- **Introduction date** — when first encountered
- **CEFR level** — the word's difficulty classification

This data drives both session content selection and spaced repetition scheduling.

### 4. Adaptive Speaking Cadence
The app's text-to-speech (TTS) output adjusts speaking speed based on learner comprehension signals:
- Starts slow (e.g. 70% of native speed)
- Speeds up incrementally as the learner demonstrates consistent understanding
- Slows back down when errors or hesitation are detected

### 5. 90–95% Known Vocabulary Rule
In any given session:
- At least 90–95% of words used are already in the learner's known vocabulary
- New words are introduced in context, explained using only known vocabulary
- This mirrors natural immersion and keeps sessions comprehensible without translation

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| UI Framework | Flet (Python, cross-platform: Android + PC) |
| Package Manager | uv |
| Speech Output (TTS) | Gemini Live API (native audio output, speed via prompt engineering) |
| Speech Input (STT) | Gemini Live API (native audio input) |
| Dialogue / Comprehension | Gemini Live API |
| Vocabulary Storage | SQLite (local, via Python `sqlite3`) |

---

## Languages

**French only at launch.** Multi-language support is a future consideration.

---

## Dialogue Engine

**Gemini API** drives the conversation:
- Generates responses in French (and English for meta-language bootstrap phrases)
- Receives the STT transcript of the learner's spoken reply
- Analyzes comprehension: did the learner respond correctly? Did they use French or fall back to English?
- Returns both the next dialogue turn and a structured comprehension signal

### Comprehension Assessment
The LLM analyzes the learner's spoken response (via STT transcript):
- Correctness of reply in context
- Use of target language vs. native language fallback
- Confidence scores per word/phrase updated after each interaction

---

## Adaptive Speaking Speed

Speaking speed is controlled via prompt engineering in the Gemini Live API system prompt. Instructions become progressively less restrictive as comprehension improves:

| CEFR Level | Prompt Instruction |
|------------|-------------------|
| A1 | "Speak very slowly. Pause between each word." |
| A2 | "Speak slowly. Pause between phrases." |
| B1 | "Speak at a measured pace." |
| B2 | "Speak at a natural conversational pace." |
| C1+ | "Speak naturally." |

Speed instructions update at the start of each session based on the learner's current level and recent comprehension signals.

---

## Session Structure

**Unstructured.** Sessions are open-ended conversations — no fixed lesson plan or word quota per session. The app naturally steers vocabulary toward the learner's current level and introduces new words opportunistically within the 90–95% known vocabulary constraint.

---

## Connectivity

**Online-first.** All three Google APIs require internet. Vocabulary data and session history are stored locally in SQLite so progress persists between sessions.

---

## Vocabulary Confidence Scoring

Each word/phrase is scored on a **1–5 scale**:

| Score | Meaning |
|-------|---------|
| 1 | Introduced — heard once, no evidence of retention |
| 2 | Familiar — recognized but not reliably produced |
| 3 | Developing — correct most of the time |
| 4 | Strong — consistently correct with little hesitation |
| 5 | Mastered — automatic, no hesitation, long retention |

Spaced repetition review intervals increase with score. Scores drop on failed recall.

---

## Onboarding / First Session

The very first session runs in **English** (the learner's native language):
- Explains the app's method briefly
- Immediately begins teaching the **meta-language bootstrap vocabulary** — the French phrases needed to drive their own learning:
  - "Comment dit-on ___ en français?" *(How do you say ___ in French?)*
  - "Que signifie ___ en anglais?" *(What does ___ mean in English?)*
  - "Répétez, s'il vous plaît." *(Please repeat.)*
  - "Plus lentement, s'il vous plaît." *(More slowly, please.)*
- From the second session onward, the app communicates primarily in French, using only known vocabulary

The bootstrap vocabulary set is **curated** — a fixed list of phrases designed specifically to give the learner the tools to ask questions and control the conversation in French from day one.

---

## Session Topics

At the start of each session the app suggests a topic based on the learner's current vocabulary and CEFR level. The learner can:
- **Accept** the suggested topic and begin
- **Choose their own** topic in French (or English early on)

Topics are used to steer the conversation but not constrain it — the session remains a natural dialogue. The app tracks which topics have been covered to ensure variety over time.
