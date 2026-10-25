# AI Market Analyst (Gemini) and voice reports (ElevenLabs)

Both are optional. The dashboard, its numbers and its decisions never depend on them.

## Setup: keys in a `.env` file

The keys live only on the server, in a `.env` file at the project root (next to `README.md`).
`.env` and `.env.*` are ignored by git; only the empty template `.env.example` is committed.

1. Create `.env` from the template, in PowerShell from the project folder:

   ```powershell
   cd "<path to>\Anomality"
   Copy-Item .env.example .env
   code .env    # or any editor that saves plain UTF-8 (VS Code does)
   ```

   Fill in `GEMINI_API_KEY`, `ELEVENLABS_API_KEY` and `ELEVENLABS_VOICE_ID` as `NAME=value`
   (no quotes, no spaces around `=`). Leave a value empty to keep that feature off. Avoid
   Notepad: it can save `.env.txt`, or add a byte-order mark that corrupts the first name (the
   template starts with comment lines, which also guards against that).

2. Start the server with Uvicorn's `--env-file` option. Uvicorn loads the file before the app
   starts, using `python-dotenv` (installed with `uvicorn[standard]` from `requirements.txt`),
   so no code change is needed. Stop any server already running on the port first (Ctrl+C):

   ```powershell
   .\.venv\Scripts\python.exe -m uvicorn backend.app:app --port 8000 --env-file .env
   ```

   Uvicorn logs only the file name ("Loading environment from '.env'"), never its contents.
   Variables already set in the PowerShell session take precedence over the file.

3. Check the configuration without revealing any key: open
   <http://localhost:8000/api/analyst/status> and <http://localhost:8000/api/voice/status>. Each
   returns only `"configured": true/false` and the model name, never a key.

**No paid call happens on its own.** Loading the page only reads those two status endpoints.
Gemini is called only when a user clicks one of the AI Market Analyst buttons ("Explain
current results", "Explain an alert", "Summarize risks", "Limits and uncertainties"), and
ElevenLabs only when a user clicks "🔊 Listen to report". Identical requests are served from
an in-memory cache (Gemini answers for an hour, audio clips while they fit in the cache).

## AI Market Analyst

Code: `backend/analyst.py`; tests: `tests/test_analyst.py` (a fake HTTP client, no key, no
paid calls); API: `GET /api/analyst/status`, `POST /api/analyst`
`{"kind": "summary" | "alert" | "risks" | "limits", "alert_id"?, "alerts_date"?}`.

**Gemini explains; the backend computes.**

1. The backend collects **facts** from its own results: the decision layer
   (`/api/decision/portfolio`), the significance check, robustness (costs, forecast quality) and
   the alerts, for the day the user is looking at. Each fact has an id (F1, F2, ...) and its
   source endpoint.
2. Gemini receives only those facts, marked as data (not instructions), and must answer in
   four sections: measured facts, interpretation, uncertainties, missing information. Each item
   cites fact ids.
3. The backend **validates** the answer: valid JSON, all four sections, every measured fact
   cites a real fact id, and **every number in the text must appear in the facts** (rounding a
   fact is allowed; any other number is rejected). A number written with a sign must match a
   fact with that sign: "-0.15" is rejected when the fact is "+0.15". An unsigned number may
   match a fact of either sign ("a drop of 30.3%" for "-30.3%").
4. If Gemini is not configured, times out, is rate-limited by Google, rejects the key, blocks the
   request, stops early or fails validation, the response falls back to the **deterministic**
   answer: the facts themselves, sorted into the same sections, with the reason shown.

Nothing is requested on page load: only a click asks. Identical questions over the same facts
are cached for an hour.

**Paid-call protection.**

- Explaining an alert first checks that the alert exists on that day. An unknown alert (or alerts
  that can't be loaded) returns **404** and never reaches Gemini.
- The whole server makes at most **10 Gemini calls per minute** (`backend/ratelimit.py`, a
  sliding window shared by every request, so varying dates or alert ids can't get around it).
  Only actual calls count: cached answers and deterministic answers are free. When the budget is
  used up the endpoint returns **429** with a `Retry-After` header.

### Configuration (server only)

| Variable | Default | |
|---|---|---|
| `GEMINI_API_KEY` | none | Without it the analyst shows the deterministic answer |
| `GEMINI_MODEL` | `gemini-3.5-flash-lite` | A stable model the Gemini docs recommend for new projects (checked Oct 2026) |
| `ANALYST_TIMEOUT` | `20` | Seconds |

The key is sent in the `x-goog-api-key` header to
`https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent`, never in a
URL, the frontend or a log. `GET /api/analyst/status` reports only whether a key is set.

### Limits

- The number check catches invented figures, not every wrong sentence: a statement built only
  from real numbers can still be misleading. Answers are labelled as written by Gemini, and the
  facts are listed under every answer so they can be checked.
- Small whole numbers that happen to appear in a fact (a date, "5 bps") pass the check.
- The check is per number, not per meaning: a number that is a rounding of *another* fact with
  the same sign passes ("+0.4" passes because "+0.39" is a fact, even if the sentence is about
  a different measure).
- Not verified against the live Gemini API in development (no key was available); the request
  format follows the official REST reference, and every failure mode is tested with a fake.

## Voice reports (ElevenLabs)

Code: `backend/voice.py`; tests: `tests/test_voice.py` (fake HTTP client) and the browser smoke
test (voice API faked in the page); API: `GET /api/voice/status`, `POST /api/voice
{"report_id": "..."}` → `audio/mpeg`.

- **Reads only what is on screen.** Every analyst answer gets a `report_id`. The voice endpoint
  accepts that id, never free text, and rebuilds the text on the server from the stored report:
  every displayed item, in order, without the fact ids, plus the disclaimer. The audio cannot
  add a claim, and the paid API cannot be used for arbitrary text.
- **Only on click** ("🔊 Listen to report"). The same text is synthesized once and cached in
  memory (at most 20 clips / 40 MB); no audio file is written to disk.
- **Limits:** empty text and text over 4,000 characters are rejected before any call (the
  default model accepts 10,000). The whole server makes at most **10 ElevenLabs calls per
  minute** (same shared limiter as Gemini, separate budget); cached audio is free. Over the
  budget the endpoint returns **429** with a `Retry-After` header.
- **Failures** (not configured, unknown or expired report, timeout, rate limit, rejected key, a
  non-audio reply) return a clear message; the written report stays on the page. ElevenLabs'
  own error message is shown too (key and voice id redacted, at most 300 characters), e.g.
  "ElevenLabs rejected the API key or voice (HTTP 400): API key ID used as API key ...".
  That one means `ELEVENLABS_API_KEY` holds the key's **ID** from the list of keys; use the
  secret key instead, which starts with `sk_` and is shown only when the key is created or
  rotated, then restart the server.

| Variable | Default | |
|---|---|---|
| `ELEVENLABS_API_KEY` | none | Without it (or the voice id) the button is disabled with a reason |
| `ELEVENLABS_VOICE_ID` | none | A voice from your ElevenLabs library. Use a stock or designed voice, not a clone of a real person without their consent |
| `ELEVENLABS_MODEL` | `eleven_multilingual_v2` | The API's default model |
| `VOICE_TIMEOUT` | `30` | Seconds |

Request: `POST https://api.elevenlabs.io/v1/text-to-speech/{voice_id}?output_format=mp3_44100_128`
with the key in the `xi-api-key` header. Not verified against the live API in development (no
key was available); every failure mode is tested with a fake.
