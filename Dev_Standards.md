# Playground — Engineering & Deployment Standards

### All 5 Phases | Senior Staff Engineer Review

---

## Phase 1 – Prompt Versioning + A/B Testing

### 1. Code Quality Standards

**Testing Coverage**

- Minimum 80% line coverage on backend Python code. Use `pytest-cov` to enforce this in CI — the pipeline fails below the threshold.
- Unit tests are mandatory for: the hash computation function, the diff engine (diff-match-patch wrapper), the parent-chain traversal logic, and the parallel execution `asyncio.gather` wrapper.
- Integration tests are mandatory for: `POST /prompts/{id}/versions` (creates correctly, sets parent pointer, computes hash), `POST /prompts/{id}/versions/{version_id}/rollback` (creates a new version, does not mutate history), and `POST /experiments/{id}/run` (returns results for all variants, handles one variant failing without aborting others).
- Frontend: no coverage gate in Phase 1. Manual QA is acceptable. You do not have time for comprehensive React testing at this stage.

**Linting / Formatting**

- Backend: `ruff` for linting (replaces flake8 + isort), `black` for formatting. Both run in CI. PRs fail if either fails.
- Frontend: ESLint with the Next.js default ruleset plus `eslint-plugin-react-hooks`. Prettier for formatting. Both run as a pre-commit hook via `husky`.
- No exceptions, no `# noqa` suppression without a code comment explaining why.

**Documentation Requirements**

- Every API endpoint must have a docstring describing: what it does, the key parameters, and what it returns. Not Swagger-level prose — a 2–3 line description is sufficient.
- The `prompt_versions` data model (the parent-chain structure, how rollbacks work, what the hash represents) must be documented in a `docs/data-model.md` file before Day 14. This is the spec that prevents future bugs.
- The `README` must include: local setup instructions (`docker compose up`), how to run tests, and what the staging URL is.

**Definition of Done for a Feature**
A feature is done when: (1) the backend endpoint is implemented and passing its integration tests, (2) the frontend is wired up and manually tested against the real backend (not mocked), (3) it is deployed to staging and verified working there, and (4) any new database tables have an Alembic migration that has been run on staging. "Works on my machine" is not done.

---

### 2. Architecture Standards

**Separation of Concerns**

- The diff engine is a pure utility module (`services/diff.py`) with zero FastAPI or database imports. It takes strings, returns structured data. It must be testable in complete isolation.
- The LLM execution engine is a service class (`services/execution.py`) behind a clean interface: `async def execute(prompt, model, temperature, max_tokens) -> ExecutionResult`. No FastAPI request objects inside it. No database calls inside it. The API layer calls it; it doesn't know about the API.
- Database access is in repository functions only (`repositories/versions.py`, `repositories/experiments.py`). No raw SQL or SQLAlchemy queries in route handlers.

**API Design Constraints**

- All endpoints return consistent error envelopes: `{ "error": { "code": "VERSION_NOT_FOUND", "message": "..." } }` with the appropriate HTTP status. No naked 500s with stack traces exposed to the client.
- All list endpoints return paginated responses from day one, even if the frontend only uses page 1. Structure: `{ "items": [...], "total": N, "page": 1, "per_page": 50 }`. Retrofitting pagination later is painful.
- Version IDs in URLs are UUIDs, not auto-increment integers. Auto-increment IDs leak information about your data volume and are awkward to migrate later.

**Database Migration Discipline**

- Every schema change ships as an Alembic migration. No `CREATE TABLE` or `ALTER TABLE` in ad-hoc SQL scripts.
- Migrations must be reversible: every `upgrade()` has a working `downgrade()`. Test the downgrade on local before merging.
- Migration files are reviewed in PR like code. The migration file name must match what it does: `add_prompt_versions_parent_id_fk.py`, not `migration_day3.py`.
- The `prompt_versions` table is insert-only by design. There must be no `UPDATE` statement anywhere in the codebase that touches `prompt_versions.content`. Add a database-level check constraint if your ORM allows it.

**Background Job Reliability**

- Phase 1 has no background jobs. Parallel execution via `asyncio.gather` is in-process and synchronous from the HTTP request's perspective. The client waits. This is acceptable for Phase 1 — do not introduce Celery yet.
- If `asyncio.gather` is used with `return_exceptions=True`, every result must be explicitly checked. A returned exception must be stored as a failed variant result, not silently dropped.

---

### 3. Reliability & Performance Standards

**Error Handling**

- Every LLM API call is wrapped in a try/except that catches at minimum: network timeout, API rate limit (429), and API server error (500+). On failure, return a structured `ExecutionResult` with `status=failed` and `error_message` populated. Never let an unhandled exception from the LLM client surface as a 500 to the frontend.
- The frontend must handle the case where one variant in an A/B run fails: show the error message in that variant's column, not a blank or broken layout.

**Concurrency**

- `asyncio.gather` for parallel variant execution is correct. Ensure you are using the async clients from both OpenAI and Anthropic SDKs — not the sync clients wrapped in `asyncio.run_in_executor`. The latter defeats the purpose.
- Maximum 6 variants per experiment (enforced at the API layer with a 400 error, not just in the UI). This bounds the concurrency and API cost per run.

**Rate Limiting**

- Phase 1: no inbound rate limiting on your own API is required. You are not public-facing yet. But you must respect OpenAI and Anthropic rate limits outbound — if a 429 is received, log it and return a clear error to the user. Do not retry automatically in Phase 1 (add retry logic in Phase 2 when you have Celery).

**Timeouts**

- Set a 30-second timeout on all outbound LLM API calls. If the call exceeds 30 seconds, cancel it and return a timeout error for that variant. Do not let a hung variant block the entire experiment run indefinitely.
- FastAPI route timeout: configure uvicorn with a 60-second worker timeout. Experiments with 6 variants at 30s each technically could hit this — the timeout is acceptable because the slowest variant in practice will be much faster.

**Observability**

- Use Python's `logging` module with structured JSON output (use `python-json-logger`). Every log line must include: `request_id` (generated per HTTP request via middleware), `endpoint`, and `duration_ms` for LLM calls.
- Log at INFO level: every experiment run start and completion with variant count and total latency. Log at ERROR level: any LLM call failure with the model name, error type, and prompt length.
- No distributed tracing in Phase 1. A `request_id` threaded through logs is sufficient.

---

### 4. Security Standards

**API Key Handling**

- LLM API keys live in environment variables only. No `.env` file is ever committed — `.env.example` with blank values is committed instead.
- Keys are never logged. Add a log sanitization check to your CI: `grep -r "sk-" --include="*.py" --include="*.ts"` must return nothing in the codebase.
- On Railway/Render, keys are set via the platform's secret management UI, never hardcoded in `railway.toml` or `render.yaml`.

**Auth Requirements**

- Phase 1 ships without user authentication. This is intentional and acceptable — you are building for internal use or a known set of users. Document this explicitly in the README as a known gap.
- If the staging URL is publicly accessible, add HTTP Basic Auth at the infrastructure level (Vercel password protection or an nginx basic auth layer) so the API is not open to the internet.

**Input Validation**

- All request bodies are validated by Pydantic models. No raw `request.json()` access in route handlers.
- Prompt content length: enforce a maximum of 32,000 characters at the API layer. Return a 400 with a clear message if exceeded. This prevents runaway token costs.
- `branch_name` and `commit_message` fields: strip leading/trailing whitespace, enforce max length of 255 characters, reject if empty after trimming.

**Data Protection**

- Prompt content is sensitive. Postgres connection must use SSL in production (Neon and Supabase enforce this by default — verify the connection string includes `sslmode=require`).
- No prompt content is ever logged. Log prompt IDs and lengths only.

---

### 5. Deployment Standards

**Pre-Deploy Verification**

- Full test suite passes in CI. No manual override of a failing CI gate.
- Alembic migrations have been run on staging and the staging environment is functional.
- A smoke test checklist is run manually on staging before promoting to production: (1) create a project, (2) save a prompt version, (3) branch from a version, (4) diff two versions, (5) create and run an A/B experiment with 2 variants, (6) verify results render correctly.

**Staging vs Production**

- Staging uses a separate database instance from production. Never run migrations against production without first running them against staging.
- Staging can use cheaper/faster models (GPT-3.5, Claude Haiku) to reduce cost during testing. Production uses the full model list.
- Environment variable `APP_ENV` is set to `staging` or `production`. Any code path that behaves differently (e.g., logging verbosity) branches on this value.

**Rollback Readiness**

- Every deploy is tagged in git with `v1.X.Y`. If a production deploy breaks, rollback is: redeploy the previous git tag on Railway/Vercel. This must take less than 5 minutes.
- Database migrations in Phase 1 are all additive (new tables, new columns). No destructive migrations. This means rollback of application code is safe — the old code will still work against the new schema (it just won't use the new columns).
- Confirm this explicitly before every migration: "If we redeploy the previous version of the app, will it break against this new schema?" If the answer is yes, the migration strategy is wrong.

**Migration Safety**

- Never drop a column or table in Phase 1. Add new columns as nullable with defaults. If a column needs renaming, add the new column, migrate data, then deprecate (not drop) the old one.
- Run `alembic upgrade head` on staging, verify the app starts successfully, then run on production. Never apply to production without staging validation.

---

### 6. Product Readiness Standards

**UX Expectations**

- Every button that triggers an async operation (saving a version, running an experiment) must show a loading state and be disabled while in-flight. No double-submit bugs.
- The diff viewer must render correctly for: identical prompts (show "no changes" message), fully rewritten prompts (all red/green), and prompts with only whitespace changes (surface these, do not hide them).
- Version history panel must show at minimum: version number, branch name, commit message, and timestamp. If a user has never written a commit message, show "No message" in muted text — not an empty cell or undefined.

**Failure-State Clarity**

- If an A/B run fails entirely (e.g., invalid API key), show a clear error banner explaining what went wrong and what to check. Not a generic "Something went wrong."
- If a single variant fails, its column must show the error type ("Rate limit exceeded", "Timeout after 30s") — not a spinner or blank box.
- If the user tries to branch from a version that doesn't exist (stale UI), the API returns a 404 and the frontend shows "This version no longer exists. Refresh to see the current history."

**Edge Case Handling**

- Zero versions: if a prompt has no saved versions, the version panel shows an empty state with instructions ("Save your prompt to create the first version").
- Concurrent saves: if two saves happen within the debounce window, only one request is sent. The frontend debounce must not fire two `POST /versions` calls within 500ms.
- Empty prompt: saving an empty prompt is rejected at the API level (400) with message "Prompt content cannot be empty."

**Internal Documentation**

- `docs/data-model.md` exists and explains the version tree, branch model, and rollback semantics before Phase 1 ships.
- A brief runbook (`docs/runbook.md`) covers: how to restart the backend, how to connect to the production database, and how to roll back a deploy. Two people building a system means both people must be able to operate it independently.

---

## Phase 2 – Prompt Scoring

### 1. Code Quality Standards

**Testing Coverage**

- Maintain 80% backend coverage. New code in this phase must not drop it.
- The scoring weight normalization math is the highest-risk logic in this phase. It requires its own unit test file (`tests/unit/test_scoring.py`) with at minimum: correct weighted average computation, handling of weights that don't sum to 100 (should raise a validation error), handling of a rubric with one criterion, and handling of the judge returning a score of 0.
- Integration test for `POST /experiments/{id}/run` with a rubric attached: verify that after the run, `scores` rows exist for each variant and the `total_score` is within expected range given the mock judge response.

**Linting / Formatting**

- Same standards as Phase 1. No relaxation.
- Add a rule: LLM judge prompt strings must live in a dedicated `prompts/` directory as Python string constants, not inline in service functions. This makes them reviewable and editable without touching service logic.

**Documentation Requirements**

- The LLM judge system prompt must have a comment explaining its structure: what it receives, what format it must return, and what happens if it returns malformed JSON.
- The rubric weighting logic must have an inline comment explaining the normalization formula. Any developer reading `scoring.py` in 6 months must understand it without asking you.

**Definition of Done**

- Same as Phase 1, plus: scoring runs asynchronously and does not extend the perceived latency of the experiment run from the user's perspective. If scoring happens in-process after execution, the frontend must not wait for it before rendering results.

---

### 2. Architecture Standards

**Separation of Concerns**

- The LLM judge is its own service (`services/judge.py`), separate from the execution service. It takes: prompt text, output text, rubric object. It returns: a structured `ScoreResult`. It has no awareness of experiments or database rows.
- Scoring is triggered as a post-execution step. In Phase 2, this can still be in-process but must be decoupled logically — the execution service does not call the judge service. The route handler orchestrates: execute → store results → trigger scoring. This sets up the clean Celery extraction in Phase 3.

**API Design Constraints**

- `GET /experiments/{id}/results` must return scores alongside execution results in a single response. No second API call from the frontend to fetch scores separately. If scores are not yet computed, return `score: null` in the variant object.
- Rubric endpoints: `POST /rubrics`, `GET /rubrics`, `GET /rubrics/{id}`, `PUT /rubrics/{id}`. Rubrics are mutable (the user may want to edit weights) but once a rubric is used in a scored run, it must be snapshotted — store the rubric state at time of scoring in the `scores` table's JSONB field, not just a foreign key. If the rubric changes later, historical scores remain interpretable.

**Database Migration Discipline**

- `scoring_rubrics` and `scores` tables are new in this phase. Both added via Alembic migrations.
- The `scores.run_id` FK references `experiment_runs`. If you anticipate this becoming a generic execution reference (for mass prompting scores later), add a comment in the migration noting this — do not refactor it yet, but flag it.

**Background Job Reliability**

- Phase 2 does not require Celery. Scoring runs in-process after execution. However: scoring must be wrapped in a try/except so a judge failure does not fail the entire experiment run. If scoring fails, store `status=failed` on the score row and log the error. The execution results must always be returned to the user regardless of scoring outcome.

---

### 3. Reliability & Performance Standards

**Error Handling**

- The LLM judge can return malformed JSON. This must be handled: wrap the JSON parse in a try/except, log the raw response, and store a failed score with `error_message="Judge returned non-JSON response"`. Never crash the scoring pipeline on a parse error.
- The judge can hallucinate scores outside the 0–10 range. Clamp all criterion scores to [0, 10] after parsing. Log a warning if clamping occurs.

**Concurrency**

- Scoring runs after all variants have executed. If scoring all variants in parallel (one judge call per variant), use `asyncio.gather` the same way as execution. Maximum 6 judge calls in parallel — same bound as variants.

**Rate Limiting**

- Judge calls add to your LLM API usage. A single experiment run with 6 variants now makes up to 12 API calls (6 executions + 6 judge calls). Ensure your OpenAI/Anthropic tier limits accommodate this. If you hit a 429 on the judge call, log it and store a failed score — do not retry in Phase 2.

**Timeouts**

- Apply the same 30-second timeout to judge calls as to execution calls.

**Observability**

- Log at INFO level: every scoring run start and completion with rubric ID, variant count, and individual scores.
- Add a metric (even if just a log line): judge call latency in ms. You want to know if scoring is adding significant latency to the workflow.
- If a score is clamped (out-of-range judge output), log at WARNING with the criterion name and the raw value received.

---

### 4. Security Standards

**Input Validation**

- Rubric criterion weights: validate server-side that weights are positive integers and sum to exactly 100. Return a 400 with specific message if not: "Criterion weights must sum to 100. Current sum: 85."
- Rubric name: max 100 characters, non-empty after trim.
- Criterion description: max 500 characters. This goes into the judge prompt — unconstrained input here is a prompt injection risk.

**Data Protection**

- The judge receives prompt content and output content. Both are sensitive. Same logging rules apply: no content in logs, only IDs and lengths.
- The rubric criteria (including descriptions) go into the judge system prompt. This means user-written text influences LLM behavior. Flag this in internal docs as a known prompt injection surface — the risk is low (users are attacking their own results), but it should be noted.

---

### 5. Deployment Standards

**Pre-Deploy Verification**

- Smoke test additions for Phase 2: (1) create a rubric with 3 criteria summing to 100 weights, (2) attach it to an experiment, (3) run the experiment, (4) verify scores appear for all variants, (5) verify the criterion breakdown is visible in the results UI, (6) verify the leaderboard shows the correct top-scoring version.

**Rollback Readiness**

- New tables (`scoring_rubrics`, `scores`) added in this phase. Rollback of app code to Phase 1 will not break — the old code doesn't reference these tables. Confirm this before deploying.
- If scoring is causing production failures, it can be disabled with a feature flag (`SCORING_ENABLED=false` env var) without rolling back the whole deploy. Wire this flag into the route handler. This is your circuit breaker.

**Migration Safety**

- Same rules as Phase 1. All migrations are additive.

---

### 6. Product Readiness Standards

**UX Expectations**

- Score column in results grid must handle null gracefully — if scoring is still in progress or failed, show a "—" not a 0. A 0 and a failed score are very different things.
- The trend chart (score over time) must handle a single data point gracefully — don't render a line chart with one point; render a message: "Run at least 2 experiments to see a trend."
- Default rubric templates ("Conciseness + Accuracy" etc.) must be preloaded in the database via a seed script, not hardcoded in the frontend. They must be selectable in the rubric builder UI immediately on first use.

**Failure-State Clarity**

- If the judge fails on one variant but succeeds on others, show the score for successful ones and "Scoring failed" for the failed one. Do not hide the failure.
- If the rubric weights don't sum to 100 (caught by API validation), the error message in the UI must name the specific problem: "Weights must sum to 100. You have 85." Not a generic form error.

**Edge Case Handling**

- Rubric with one criterion (weight = 100): must work correctly. The weighted average of one criterion is just that criterion's score.
- Running an experiment without a rubric attached: scoring simply doesn't run. The results UI shows no score column. This is correct behavior, not an error.
- Editing a rubric after it has been used in a run: allowed, but the historical score records remain tied to the snapshot of the rubric at time of scoring. Verify this is working correctly before Phase 2 ships.

---

## Phase 3 – Mass Prompting

### 1. Code Quality Standards

**Testing Coverage**

- The combination generation logic (Cartesian product of variable values) is the most algorithmically risky code in this phase. It needs thorough unit tests: single variable with multiple values, multiple variables, variables appearing multiple times in the template, a variable with one value (degenerate case), and a variable with zero values (should return a validation error).
- The template parser must have unit tests for: basic `{{variable}}` substitution, a variable that appears multiple times, malformed braces (`{variable}` — single brace, should not match), and an empty variable name (`{{}}`).
- Integration test for the batch executor: mock the LLM client, run a mass job with 4 combinations, verify all 4 combination rows end up in `complete` status with output populated.

**Documentation Requirements**

- `docs/mass-prompting.md`: document the template syntax, how Cartesian products are computed, what the concurrency limiter does and why it exists, and what happens when a combination fails mid-batch.
- The batch executor's concurrency configuration (`BATCH_SIZE`, `BATCH_CONCURRENCY`) must be documented with their defaults and the reasoning behind the defaults.

**Definition of Done**

- Same as prior phases, plus: the SSE progress stream must be manually tested and confirmed working in the browser before shipping. SSE has browser-specific behavior that integration tests don't catch.

---

### 2. Architecture Standards

**Separation of Concerns**

- The template parser is a pure utility (`services/template.py`): takes a string, returns a list of variable names. No database, no FastAPI.
- The combination generator is a pure utility: takes variable definitions, returns a list of value dicts. No database.
- The batch executor (`services/batch_executor.py`) is the only component that touches both the database and the LLM execution service. It reads combinations from the DB, calls the execution service, writes results back. It does not handle HTTP.
- **This is the phase where Celery becomes necessary.** Batch runs can take minutes. They must run in a background worker, not in a request-response cycle. Introduce Celery + Redis now. The route handler enqueues a task and returns immediately with a `mass_run_id`. The frontend polls (or SSE) for progress.

**API Design Constraints**

- `POST /mass-runs` returns `202 Accepted` with the `mass_run_id`. It does not wait for the run to complete.
- `GET /mass-runs/{id}/status` returns current status: `pending`, `running`, `complete`, `failed`, plus a `progress` object: `{ "total": 50, "complete": 23, "failed": 2 }`.
- `GET /mass-runs/{id}/results` returns the full combination results. This is paginated — do not return 50 full output objects in one response. Default page size: 20.
- SSE endpoint: `GET /mass-runs/{id}/stream` pushes progress events. Each event is `{ "combination_id": "...", "status": "complete", "score": 87.3 }`. The frontend updates the grid live.

**Database Migration Discipline**

- `mass_runs` and `mass_run_combinations` tables added via Alembic. The `mass_run_combinations.status` column uses a Postgres `ENUM` type (`pending`, `running`, `complete`, `failed`). Define this enum in the migration, not just as a string column.
- Add a database index on `mass_run_combinations(mass_run_id, status)`. You will query "all pending combinations for this run" repeatedly in the batch executor — without this index, it degrades badly at scale.

**Background Job Reliability**

- Celery tasks must be idempotent: if a combination task is retried (e.g., after a worker crash), it checks if the combination already has a result before re-executing. Use the combination's `status` column as the guard.
- Celery task failure handling: if a combination task fails after 3 retries, mark the combination as `failed` with the error message. Do not leave it in `running` state indefinitely (stale lock). Set a task `time_limit=90` seconds in Celery config.
- The Celery worker must be a separate process in Docker Compose, not the FastAPI process. Document how to start it in the README.

---

### 3. Reliability & Performance Standards

**Error Handling**

- If one combination fails (LLM error, timeout), that combination is marked `failed`. The rest of the batch continues. The mass run itself only transitions to `failed` if all combinations fail.
- If the Celery worker crashes mid-run, the combinations left in `running` state are orphaned. Add a recovery mechanism: a scheduled task (or a check on `GET /mass-runs/{id}/status`) that detects combinations stuck in `running` for more than 2 minutes and resets them to `pending`.

**Concurrency**

- The concurrency limiter (batch size of 5) is enforced in the Celery task, not just the executor. Use `asyncio.Semaphore(5)` within the async batch executor function. This is the primary rate limit protection.
- Do not generate more than 100 combinations per mass run. Enforce this at the API layer with a 400 error. This bounds cost and runtime.

**Rate Limiting**

- At 5 concurrent LLM calls, each at ~2s average, a 100-combination run takes ~40 seconds of wall time. At 5 concurrent calls with scoring (5 more judge calls per batch), you're making 10 LLM calls per 5-second window. Verify this is within your API tier limits before shipping.
- If a 429 is received during batch execution, implement exponential backoff for that combination's retry: wait 2s, then 4s, then 8s before marking it failed. This is the first phase where retry logic is worth the complexity.

**Observability**

- Log at INFO level: mass run start (template variable count, combination count), each batch of 5 fired, mass run completion (total combinations, success count, failure count, total duration).
- Add a Celery task failure alert: if more than 20% of combinations in a run fail, log at ERROR level with the run ID and failure rate.

---

### 4. Security Standards

**Input Validation**

- Template variable names: alphanumeric plus underscores only. Reject any template containing variable names with special characters.
- Variable values: each individual value max 1,000 characters. Total value count per variable: max 10. Total combinations: max 100 (enforced server-side).
- Template string: same 32,000 character max as prompt content.
- The rendered prompt (after substitution) must also pass the 32,000 character max check before execution. A template with a large context variable could balloon past this limit.

**Data Protection**

- Mass run combination outputs are sensitive (they contain LLM outputs of user prompts). Same logging rules: no content in logs. Combination IDs only.

---

### 5. Deployment Standards

**Pre-Deploy Verification**

- Celery worker must be deployed and confirmed running before the Phase 3 deploy is considered live. Test: submit a mass run from the UI and verify the Celery worker picks it up within 10 seconds.
- Smoke test: create a template with 2 variables (2 values each = 4 combinations), run it, verify all 4 combinations complete with results, verify the results grid sorts by score correctly, verify "Save as version" from a combination works.
- Redis must be running in production. Verify the Celery broker connection before promoting to production.

**Rollback Readiness**

- Rolling back to Phase 2 is safe at the database level (new tables are additive). Application rollback is safe — Phase 2 code does not reference mass prompting tables.
- If the Celery worker has a bug causing mass runs to hang, the circuit breaker is to kill the Celery worker. Pending mass runs will pause. Existing results are unaffected. Document this in the runbook.

---

### 6. Product Readiness Standards

**UX Expectations**

- The combination count preview ("This will generate X combinations") must update in real-time as the user adds variable values. It must be visually prominent — this is the user's primary cost/time signal.
- The warning at 50+ combinations must be a blocking confirmation dialog, not just a text warning.
- The results grid must be sortable by score in both directions, and filterable by a specific variable value. If a user has 100 combinations, they need to be able to isolate "show me all combinations where tone=formal."

**Failure-State Clarity**

- If a combination fails, its row in the grid shows the error type in the output column — not a blank or spinner.
- If the entire mass run fails (Celery worker crashed, Redis down), the status page shows "Run failed — please try again" with the timestamp of failure. Not a perpetual spinner.
- The SSE stream disconnect: if the user closes the browser tab and reopens, the frontend must re-hydrate from `GET /mass-runs/{id}/status` and `GET /mass-runs/{id}/results` rather than reconnecting the SSE stream from scratch. The results are in the database.

**Edge Case Handling**

- Template with no variables: reject at UI level before submission with message "Your template has no `{{variables}}`. Add at least one variable to use mass prompting."
- A variable with only one value: valid, but warn: "Variable 'tone' has only one value — it won't produce variation. Add more values to explore the space."
- Model as a variable: if `{{model}}` is used and one of the model values is invalid (typo), that combination fails cleanly. The error message in the combination row must say "Invalid model name: 'gpt-4o-turb'" — not a generic API error.

---

## Phase 4 – Prompt Breakdown

### 1. Code Quality Standards

**Testing Coverage**

- The ablation engine is the most critical and most expensive component in the entire product. It must have integration tests using mocked LLM clients. Test: a prompt with 3 segments produces exactly 4 executions (1 full + 3 ablations). Test: if one ablation fails, the others still complete and the result is stored with the failed ablation marked.
- The attribution analyzer (the judge that maps segments to output) is tested with a fixed mock: given a known diff, the judge returns a fixed attribution; verify it is stored correctly in the `attributions` table.
- The SVG connection-line rendering (the bezier curves) cannot be unit tested. Write a visual regression checklist instead: a document with screenshots of correct states that QA is verified against manually before deploy.

**Documentation Requirements**

- `docs/breakdown-design.md` must exist before Day 38 (before any UI work). It must document: the attribution algorithm in plain English, the ablation execution strategy, the data model for `attributions` (including what `confidence` means and how it's computed), and the segment type taxonomy.
- The LLM segmenter system prompt and the LLM attribution analyzer system prompt must both live in `prompts/breakdown_segmenter.py` and `prompts/breakdown_attributor.py` as named string constants with comments.

**Definition of Done**

- Breakdown is done when: auto-segmentation produces sensible results on a real 500+ word prompt, the attribution visualization renders with visible connection lines, and a manual QA run confirms the lines draw/clear correctly on hover/click. The visualization is interactive — "renders" is not sufficient.

---

### 2. Architecture Standards

**Separation of Concerns**

- The segmentation service, the ablation runner, and the attribution analyzer are three separate service modules. None of them calls another directly. The `breakdown_orchestrator.py` module coordinates them in sequence. This keeps each component testable and replaceable.
- The SVG line rendering in the frontend is a self-contained React component (`<BreakdownConnections />`). It receives segment positions and attribution data as props. It has no API calls inside it. Position data comes from `getBoundingClientRect()` calls in the parent, passed down.

**API Design Constraints**

- `POST /breakdown/run` returns `202 Accepted` with a `breakdown_run_id`. This triggers a Celery task — ablation can mean 10+ LLM calls and takes minutes.
- `GET /breakdown/{run_id}` returns the complete data structure for the visualization in a single call: segments, output segments, and attributions. This response can be large (many attribution rows) — cap at 200 attribution records and document this limit.
- Cost estimate endpoint: `POST /breakdown/estimate` takes a prompt and returns `{ "segment_count": N, "estimated_api_calls": N+1, "estimated_cost_usd": X }`. This must be called before the user confirms a breakdown run. Cost estimation is based on average token counts — it does not need to be exact, but must be within 2x of actual cost.

**Database Migration Discipline**

- Four new tables: `breakdown_runs`, `prompt_segments`, `output_segments`, `attributions`. All added via Alembic.
- `attributions` has a composite index on `(prompt_segment_id, output_segment_id)` — you will query "all output segments attributed to this prompt segment" in the visualization data fetch.
- `prompt_segments` and `output_segments` store `start_char` and `end_char` offsets. Add a check constraint: `start_char < end_char` and both are non-negative. Malformed offsets from LLM output cause rendering bugs that are hard to diagnose.

**Background Job Reliability**

- The ablation Celery task runs all N+1 executions. If the task crashes after some ablations complete, on retry it must check which ablations already have results and skip them. This is critical given the cost of each ablation.
- Set `task_time_limit=600` (10 minutes) for breakdown tasks — ablation on a long prompt with many segments is legitimately slow.
- Send a Celery task completion event that the frontend can pick up via SSE to know when the breakdown is ready.

---

### 3. Reliability & Performance Standards

**Error Handling**

- If auto-segmentation returns 0 segments (LLM failure or empty output), the breakdown run must fail immediately with a clear error: "Segmentation failed — the AI could not identify segments in this prompt. Try manual segmentation or simplify the prompt."
- If the attribution analyzer returns character offsets that exceed the length of the output string, clamp them to valid ranges and log a WARNING. Do not crash the attribution storage.
- If more than 50% of ablation runs fail, mark the breakdown run as `failed` and surface this to the user — do not attempt attribution analysis on partial data.

**Concurrency**

- Ablation runs fire in parallel using `asyncio.gather`, but limit concurrency to 5 simultaneous ablations (same Semaphore pattern as mass prompting). A prompt with 15 segments will process ablations in 3 batches of 5.

**Timeouts**

- Individual ablation calls: 30-second timeout (same as standard execution).
- The attribution analyzer call: 60-second timeout — this is a more complex LLM call processing diffs.
- Total Celery task time limit: 600 seconds. If hit, mark the run as `failed` with "Breakdown timed out."

**Observability**

- Log at INFO: breakdown run start (segment count, estimated ablation count), each batch of ablations fired, attribution analysis start/complete.
- Log at WARNING: any clamped character offset, any confidence score outside [0, 1].
- Log at ERROR: breakdown run failure with reason.
- Track ablation cost in logs: total tokens used across all ablation calls per breakdown run. This is the most expensive operation in the product — you need visibility into actual costs.

---

### 4. Security Standards

**Input Validation**

- Manual segment delimiter syntax: parse strictly. If the delimiter syntax is malformed (e.g., unclosed `### [`), return a 400 with the specific line number of the malformation.
- Segment label user input: max 100 characters, alphanumeric plus spaces and hyphens. Reject special characters — this text is rendered in the UI and interpolated into LLM prompts.
- The attribution analyzer receives LLM-generated diffs as input. This is AI-generated content influencing another AI call. Flag this in internal docs as a known chain-of-trust issue — the output of one call is not sanitized before being passed to another. This is acceptable, but document it.

---

### 5. Deployment Standards

**Pre-Deploy Verification**

- Smoke test on staging with a real prompt (minimum 300 words, 4+ natural sections): (1) auto-segmentation produces sensible segments, (2) ablation runs complete, (3) attribution data is stored, (4) visualization renders with connection lines, (5) hover on a prompt segment highlights the correct output segments, (6) "Re-run attribution" after a segment edit completes successfully.
- The cost estimate endpoint must be verified to return plausible numbers before deploy. Run it on a known prompt and manually verify the estimate against your actual API billing.

**Rollback Readiness**

- Phase 4 introduces the most complex new tables. Rollback of application code to Phase 3 is safe (Phase 3 doesn't reference breakdown tables). But the Celery task queue may have in-progress breakdown tasks if you rollback. Document the procedure: flush the Celery queue before rolling back (`celery -A app purge`), then redeploy Phase 3 code.

**Migration Safety**

- The check constraints on `start_char < end_char` must be verified to work correctly on the staging database before production migration. Run a test: attempt to insert a row with `start_char > end_char` and confirm the constraint fires.

---

### 6. Product Readiness Standards

**UX Expectations**

- The cost estimate must be shown in dollars (even if approximate) with a disclaimer: "Estimated cost: ~$0.12. Actual cost may vary." If the estimate exceeds $1.00, require an explicit confirmation: "This breakdown will cost approximately $X. Continue?"
- The visualization must degrade gracefully when there are many segments: if there are more than 10 prompt segments, the connection lines must not become an illegible tangle. Cap the simultaneously visible lines to the hovered segment only — never draw all lines at once.
- The segment editor must provide instant visual feedback when a segment boundary is moved: the prompt text highlights update in real-time.

**Failure-State Clarity**

- If the Celery task fails mid-ablation, the breakdown status page shows: "Breakdown failed after completing X of Y ablations. Reason: [error]." It offers a "Retry" button that resumes from where it left off (idempotent retry).
- If attribution produces no connections for a segment (confidence below threshold for all pairs), show a tooltip on that segment: "No clear attribution found for this segment."

**Edge Case Handling**

- Prompt with one segment (the LLM determined the entire prompt is one unit): breakdown is not useful. Show a message: "Your prompt was detected as a single segment. Breakdown attribution requires at least 2 segments. Try manual segmentation."
- Output that is very short (under 50 characters): attribution to specific output segments may be meaningless. Show a warning: "Output is very short — attribution confidence may be low."
- Manual segment mode with overlapping segment boundaries: reject at the API level with a specific error identifying the overlapping character ranges.

---

## Phase 5 – The Perfect Prompt

### 1. Code Quality Standards

**Testing Coverage**

- The data aggregation layer (`GET /projects/{id}/performance-summary`) is the most important new backend code in this phase. It must have an integration test that seeds the database with known prompt versions, scores, mass run results, and breakdown attributions, then verifies the aggregation query returns the correct top-N results in the correct order.
- The synthesis LLM call has non-deterministic output — it cannot be unit tested for correctness. What can be tested: the synthesis prompt construction function (given a performance summary object, produces a correctly structured string with all expected sections).
- Test: accepting a suggestion creates a new prompt version tagged `perfect_prompt_generated` linked to the performance summary that generated it.

**Documentation Requirements**

- `docs/perfect-prompt-design.md` must document: what data is aggregated and why (the selection criteria for top-10 versions, top-5 A/B results, etc.), the synthesis prompt structure and why it's ordered the way it is, and the provenance chain from suggestion to saved version.
- The synthesis system prompt lives in `prompts/perfect_prompt_synthesizer.py` with a detailed comment explaining each section of the prompt and its intent.

**Definition of Done**

- A Perfect Prompt suggestion is "done" when: it includes source annotations (which sections came from which historical versions), the suggested prompt is editable before saving, and saving creates a version with full provenance stored. A suggestion that cannot be traced back to its sources is not shippable.

---

### 2. Architecture Standards

**Separation of Concerns**

- The aggregation query is in `repositories/performance.py`. It does not contain synthesis logic.
- The synthesis engine is in `services/synthesizer.py`. It takes a `PerformanceSummary` object and a goal string. It returns a `SynthesisResult` with the suggested prompt text and a list of source attributions. It does not touch the database.
- The route handler orchestrates: call aggregation → call synthesizer → return result. It does not contain business logic.

**API Design Constraints**

- `GET /projects/{id}/performance-summary` returns deterministic, cached data. Cache it for 5 minutes in Redis — the aggregation query touches many tables and should not run on every request.
- `POST /projects/{id}/perfect-prompt` accepts a goal statement and optional rubric ID. It calls the aggregation layer and synthesis engine synchronously (the synthesis call is one LLM call — fast enough for a request-response cycle) and returns the suggestion. No Celery task needed.
- The response includes: `suggested_prompt` (the full text), `explanation` (why the prompt is structured as it is), and `sources` (an array of `{ version_id, contribution_description }` objects). All three fields are mandatory. A response missing sources is an API contract violation.

**Background Job Reliability**

- The iteration loop (re-running scoring after accepting a suggestion) reuses the existing A/B testing and scoring infrastructure. No new background job patterns needed.
- The Perfect Prompt history (`GET /projects/{id}/perfect-prompt-history`) is a simple database query — no caching needed in Phase 5.

---

### 3. Reliability & Performance Standards

**Error Handling**

- If the performance summary aggregation returns no data (brand new project, no runs yet), the synthesis endpoint returns a `422 Unprocessable Entity` with message: "Insufficient data. Run at least one scored experiment before generating a Perfect Prompt." Do not call the synthesis LLM with empty context.
- If the synthesis LLM returns a response that does not include source attributions (it hallucinated a prompt without citing sources), log a WARNING and return the suggestion anyway, but flag `sources: []` in the response. The UI must handle this case and show a warning.
- If the goal statement is vague or contradictory, the synthesis LLM may return a poor suggestion. This is acceptable — the system cannot guarantee quality. The UX must set this expectation.

**Performance**

- The aggregation query joins across `prompt_versions`, `scores`, `experiment_runs`, `mass_run_combinations`, and `attributions`. On a project with 100+ versions and 500+ runs, this query can be slow. Add `EXPLAIN ANALYZE` output to `docs/query-analysis.md` for this query before shipping. If query time exceeds 500ms on staging data, add targeted indexes before deploying.
- The synthesis LLM call has a large context window (top-10 prompts verbatim plus scores and breakdowns). Estimate the token count of the synthesis request before sending and log it. If the context exceeds 100K tokens, truncate the historical prompts to their first 2,000 characters each with a note in the system prompt.

**Observability**

- Log at INFO: synthesis request start (project ID, goal statement length, number of source versions included), synthesis complete (response length, source count, latency).
- Log at WARNING: synthesis returned empty sources, context window truncation occurred.

---

### 4. Security Standards

**Input Validation**

- Goal statement: max 1,000 characters, non-empty after trim.
- The goal statement goes directly into the synthesis LLM prompt. This is user-controlled text influencing an LLM call. Treat it as a prompt injection surface. Prepend the goal statement in the synthesis prompt with a structural delimiter that makes injection harder: `<user_goal>{goal}</user_goal>`. Document this in the synthesis prompt file.
- The performance summary is internal data — do not expose raw prompt content from other users' projects in the aggregation endpoint. (Relevant if you add multi-user support in the future — flag this in docs now.)

**Auth Requirements**

- Phase 5 ships with the same auth posture as all prior phases (none or basic auth at infra level). If you have added user auth by this phase, verify that the performance summary endpoint only returns data from the requesting user's projects. Do not assume the project ID in the URL is safe without an ownership check.

---

### 5. Deployment Standards

**Pre-Deploy Verification**

- The aggregation query must be performance-tested on staging data before production deploy. Seed staging with at least 50 prompt versions, 20 experiment runs, and 10 mass runs, then run the query and verify it returns in under 500ms.
- Smoke test: (1) project with existing scored runs generates a suggestion, (2) suggestion includes source annotations, (3) editing the suggestion and saving creates a new version, (4) the new version appears in the prompt version history with the `perfect_prompt_generated` tag, (5) the Perfect Prompt history shows the just-generated suggestion.
- Verify the Redis cache for the performance summary is working: call the endpoint twice and confirm the second call is faster (log the latency difference).

**Final Production Deploy Checklist (Phase 5 = Full System)**
This is the final deploy. Before it goes live, verify the entire system end-to-end on staging:

1. Create project, write prompt, save versions, branch.
2. Run A/B test with 3 variants across 2 models.
3. Score with a custom rubric.
4. Run mass prompting with 3 variables, 3 values each.
5. Save a winning combination as a version.
6. Run breakdown on the winning version.
7. Generate a Perfect Prompt.
8. Score the generated prompt.
9. Verify the full data chain: the Perfect Prompt history shows the suggestion; the version history shows the generated version with provenance; the leaderboard shows the scored result.

If any step fails on staging, fix it before promoting to production. No exceptions.

**Rollback Readiness**

- If the synthesis LLM begins returning unusable results in production (a model degradation event), disable the Perfect Prompt feature with `PERFECT_PROMPT_ENABLED=false` env var. The rest of the product continues working. Document this in the runbook.

---

### 6. Product Readiness Standards

**UX Expectations**

- The suggestion review page must show source annotations inline — not in a separate panel. Each section of the suggested prompt that was derived from a historical version has a visible marker (a colored tag or subtle highlight) that, when hovered, shows: "Derived from Version 7 (Score: 91)" with a link to that version.
- The goal definition field must have a placeholder example: "A prompt that extracts action items from meeting transcripts with high completeness and structured formatting." First-time users will not know what to write without an example.
- The onboarding tour (Day 52) must cover all 5 phases in order and show a real example of each feature, not placeholder screenshots.

**Failure-State Clarity**

- If there is insufficient data to generate a suggestion, the Perfect Prompt page must explain what is needed: "To generate a Perfect Prompt, you need at least one scored experiment run. [Go run an experiment →]." Not an error message — an actionable prompt.
- If the suggestion takes more than 10 seconds to generate, show a progress message: "Analyzing your prompt history…" with a spinner. 10 seconds of silence feels like a hang.

**Edge Case Handling**

- Project with only one scored version: the synthesis runs but the explanation must acknowledge the limited data: "Generated from 1 scored version. Run more experiments for better suggestions." This expectation must be set in the UI, not hidden.
- User edits the suggestion to be empty before saving: reject with "Prompt cannot be empty."
- Perfect Prompt suggestion that is identical to an existing version (the LLM reproduced a historical prompt exactly): detect this via hash comparison before saving. Show a warning: "This suggestion is identical to Version 5. Save anyway?" Do not silently create a duplicate version.

**Internal Documentation Requirements (Full System)**
By Phase 5 deploy, the following documentation must exist and be up to date:

- `docs/data-model.md` — full schema with relationships
- `docs/breakdown-design.md` — attribution algorithm
- `docs/mass-prompting.md` — template syntax and batch execution
- `docs/perfect-prompt-design.md` — aggregation logic and synthesis
- `docs/runbook.md` — how to restart each service, rollback a deploy, connect to production DB, flush the Celery queue, and disable any feature via env flag
- `docs/query-analysis.md` — EXPLAIN ANALYZE output for the performance summary query
- `README.md` — local setup, test execution, staging URL, production URL, and environment variable reference

**This is the complete standards contract for all 5 phases of Prompt Lab.**