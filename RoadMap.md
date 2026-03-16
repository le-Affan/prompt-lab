# Playground Roadmap

# Project Brain — Playground Layer: Comprehensive Implementation Roadmap

---

## How to Read This Roadmap

The Playground ships in **5 phases**, each ending in a deployable product. Every phase builds on the last. Days are working days. Each day has a clear output — something tangible you can point to at the end of that day.

---

## Phase 1 — MVP: Prompt Versioning + A/B Testing

**Duration: Days 1–14 | Outcome: A live, usable prompt workbench with versioning and A/B testing**

---

### Week 1 — Foundation & Versioning

**Day 1 — Project Setup**

Stand up the monorepo. Initialize a Next.js 14 frontend and a FastAPI backend in the same repo. Configure Postgres with Docker Compose locally. Set up your environment variable structure for API keys. Configure ESLint, Prettier, and basic CI. At the end of this day you should be able to run `docker compose up` and get a working frontend and API server talking to each other.

**Day 2 — Database Schema: Core Versioning Tables**

Design and migrate your foundational schema. You need a `projects` table, a `prompts` table that stores the current active prompt text per project, and a `prompt_versions` table that is the heart of the system. The versions table stores: `id`, `prompt_id`, `content` (the full prompt text), `parent_version_id` (nullable FK to itself), `branch_name`, `commit_message`, `hash` (SHA256 of the content), `created_at`, and a `metadata` JSONB column for anything extra. Write Alembic migrations. The schema should be immutable on the versions side — no updates, only inserts.

**Day 3 — Versioning Backend: Core API**

Build the versioning endpoints. `POST /prompts/{id}/versions` — saves a new version, computes the hash, sets the parent pointer. `GET /prompts/{id}/versions` — returns the version tree with parent-child relationships. `GET /prompts/{id}/versions/{version_id}` — returns a single version. `POST /prompts/{id}/versions/{version_id}/rollback` — creates a new version pointing back to the target version's content (it doesn't mutate history, it just creates a new entry that is a copy of the target). `POST /prompts/{id}/versions/{version_id}/branch` — creates a new branch from any version.

**Day 4 — Diff Engine**

Implement the diff computation. Use the `diff-match-patch` library (Python port). When two version IDs are given, compute a character-level diff and return it as a structured JSON payload: an array of operations, each being `[type, content]` where type is `+`, `-`, or `=`. Also compute a higher-level "line diff" for the visual view. Write unit tests for your diff logic today — this is core infrastructure and needs to be reliable.

**Day 5 — Frontend: Prompt Editor**

Build the main prompt editor page. Use CodeMirror 6 for the editor — it handles large prompts gracefully and gives you syntax highlighting hooks you'll need later for the breakdown feature. The page layout: editor on the left two-thirds, version history panel on the right third. The editor should auto-save a draft locally on every keystroke (debounced 500ms). When the user explicitly saves, it hits the backend and creates a new version. The version panel shows a chronological list of versions with their commit messages and timestamps.

**Day 6 — Frontend: Version Tree + Diff Viewer**

Build the version history panel fully. Clicking a version shows a diff viewer below the version list — green for additions, red for deletions, identical to how GitHub shows it. Clicking "Restore this version" triggers the rollback endpoint and reloads the editor with that content. Clicking "Branch from here" prompts for a branch name and calls the branch endpoint. At end of day: a fully working prompt version control UI.

**Day 7 — Buffer + Integration Test**

End-to-end test the full versioning flow. Create a project, write a prompt, save multiple versions, branch, make changes on the branch, diff two arbitrary versions, restore an old version. Fix any rough edges. Write a brief internal doc describing the data model so future team members understand the parent-chain structure.

---

### Week 2 — A/B Testing

**Day 8 — LLM Execution Engine**

Build the core execution service. This is the layer that takes a prompt and model config and returns a response. Abstract it cleanly: an `execute(prompt: str, model: str, temperature: float, max_tokens: int) -> ExecutionResult` interface. Implement adapters for OpenAI and Anthropic to start. Store the API keys in environment variables. The execution result should include the output text, token counts, latency in ms, and the full model config used. This service will be called by every future feature.

**Day 9 — A/B Testing Schema + Backend**

Add the experiments schema. An `experiments` table with `id`, `name`, `input_text`, `created_at`. An `experiment_variants` table where each variant has a `prompt_version_id` (nullable), `model`, `temperature`, `max_tokens`, and a `label` (e.g. "Variant A"). An `experiment_runs` table that stores the actual outputs: `variant_id`, `output_text`, `latency_ms`, `token_count`, `run_at`. Build the endpoints: `POST /experiments` (create), `POST /experiments/{id}/run` (executes all variants in parallel using `asyncio.gather`), `GET /experiments/{id}/results`.

**Day 10 — Parallel Execution + Result Storage**

Implement the parallel execution properly. When a user runs an A/B test, all variants fire simultaneously, not sequentially. Use Python's `asyncio` with the async clients from OpenAI and Anthropic. Handle failures gracefully — if one variant fails, store the error and still return the others. The run should complete in the time of the slowest variant, not the sum of all of them.

**Day 11 — A/B Testing UI: Setup**

Build the experiment creation page. The user picks an experiment name, writes the shared input text, then configures each variant. For each variant: select a prompt version from a dropdown (populated from the versioning system), select a model, set temperature and max tokens. They can add up to 6 variants. The layout should feel like a configuration form, clean and structured.

**Day 12 — A/B Testing UI: Results**

Build the results view. This is the most important UI in the MVP. When a run completes, show a grid layout where each variant gets a column. Each column shows: the label, the model used, the prompt version used, the output text in a scrollable box, latency, and token count. A horizontal bar or subtle color coding shows relative latency. The user can click any cell to expand it to full width. At end of day this should feel genuinely useful — someone should be able to look at it and immediately understand which variant performed differently.

**Day 13 — Model Selector + Run History**

Add a proper model selector component (searchable dropdown with the full list of current OpenAI and Anthropic models). Build a run history view per experiment — every time the user runs an experiment, the results are stored and browsable. This is important for the Perfect Prompt feature later. Runs are timestamped and labeled with their run number.

**Day 14 — MVP Polish + Deployment**

Polish the MVP. Add loading states everywhere. Add error toasts. Write a basic `README`. Deploy to a staging environment (Railway or Render for backend, Vercel for frontend). At end of day you have a live URL. **This is your first shipped product.** Someone can go to the URL, create a project, write a prompt, version it, branch it, and run A/B tests across different models and prompt versions.

---

## Phase 2 — Prompt Scoring

**Duration: Days 15–21 | Outcome: Every prompt execution gets scored against user-defined criteria**

---

**Day 15 — Scoring Schema**

Add scoring tables. A `scoring_rubrics` table: `id`, `name`, `criteria` (JSONB array of criterion objects, each with `name`, `description`, `weight`), `project_id`. A `scores` table: `id`, `run_id` (FK to experiment_runs or a future generic execution table), `rubric_id`, `total_score` (0–100), `criterion_scores` (JSONB), `reasoning` (the LLM judge's explanation), `scored_at`. Design the scoring as a separate async step — it doesn't block execution, it runs after.

**Day 16 — Rubric Builder UI**

Build the rubric builder. The user creates a named rubric with a list of criteria. Each criterion has a name, a plain-English description, and a weight (0–100, weights must sum to 100). Give them a few default templates: "Conciseness + Accuracy", "Creative Quality", "Technical Correctness". The rubric is reusable across experiments.

**Day 17 — LLM Judge Implementation**

Build the scoring engine. The judge is a separate LLM call (use a capable model — GPT-4o or Claude Sonnet). The judge receives: the original prompt, the output to evaluate, and the rubric. The system prompt instructs the judge to return a structured JSON object with a score per criterion (0–10) and a one-sentence reasoning per criterion. Multiply by weights and normalize to 0–100. Store everything. Run unit tests with mock rubrics to verify the math is correct.

**Day 18 — Attaching Scores to A/B Runs**

Wire scoring into the experiment flow. When a user attaches a rubric to an experiment before running, the scoring fires automatically after execution completes. The results UI gets a score column per variant. The variant with the highest score gets a subtle highlight. The criterion breakdown is expandable — click the score to see how each criterion was rated.

**Day 19 — Prompt Version Leaderboard**

Build the version performance view. For any prompt, show a table of all its versions with their average scores across all runs. Sort by score descending. This gives the user a clear picture of which iteration of their prompt is performing best. Include run count, average latency, and average score. This is the first place the data starts to compound in value.

**Day 20 — Score History + Trend**

Add a simple line chart (use Recharts — already available in your React setup) showing score over time per prompt version. X-axis is run date, Y-axis is score. Multiple versions are overlaid as separate lines. The user can now see at a glance whether their prompt iteration is trending upward.

**Day 21 — Integration + Deploy**

Full end-to-end test of the scoring flow. Create a rubric, attach it to an A/B experiment, run it, verify scores appear correctly, check the leaderboard, check the trend chart. Fix anything broken. Deploy to staging. **Scoring is now live.**

---

## Phase 3 — Mass Prompting

**Duration: Days 22–30 | Outcome: Grid search over the prompt space with automatic scoring**

---

**Day 22 — Template Parser**

Build the prompt template engine. Adopt the `{{variable_name}}` syntax. The parser scans a prompt string and extracts all variable names. It validates that every variable referenced in the template has a defined value set. Build this as a pure utility function with thorough unit tests — edge cases include nested braces, variables appearing multiple times, and empty variable names.

**Day 23 — Variable Configuration UI**

Build the mass prompting setup page. At the top: the prompt template editor (same CodeMirror component, but highlighting `{{variable}}` tokens in a distinct color). Below: a variable table where each detected variable gets a row. For each variable, the user adds multiple values — one per line, or as a comma-separated list. A preview counter shows how many combinations will be generated (the Cartesian product). Warn the user if they're about to generate more than 50 combinations.

**Day 24 — Combination Generation + Schema**

Add the mass prompting schema. A `mass_runs` table with `id`, `template`, `constants` (JSONB), `variable_definitions` (JSONB), `rubric_id`, `created_at`. A `mass_run_combinations` table with `id`, `mass_run_id`, `variable_values` (JSONB — the specific variable values for this combination), `rendered_prompt` (the full prompt text after substitution), `status` (pending/running/complete/failed). Generate all combinations server-side and insert them all as rows before execution begins. This makes progress tracking trivial.

**Day 25 — Batch Execution Engine**

Build the batch executor. Takes a `mass_run_id`, queries all pending combinations, and executes them. Use a concurrency limiter — don't fire all 50 at once or you'll hit rate limits. Process in batches of 5 (configurable). After each execution, update the combination row with the output and fire the scoring job. Stream progress back to the frontend via Server-Sent Events so the user sees a live progress bar.

**Day 26 — Results Grid UI**

Build the mass run results grid. This is a table where rows are combinations and columns are: the variable values (one column per variable), the rendered prompt (truncated, expandable), the output (truncated, expandable), the score, and latency. The table is sortable and filterable. Sort by score descending by default so the best-performing combinations surface immediately. This is the core value of mass prompting — you want the winner to jump out.

**Day 27 — Combination Detail View**

Clicking a combination row opens a detail panel. Shows: the full rendered prompt, the full output, the score with criterion breakdown, the variable values that produced this result. A "Save as version" button creates a new prompt version from this combination's rendered prompt, linking it back to the mass run as its provenance. This closes the loop between mass prompting and versioning.

**Day 28 — Winner Analysis**

Build a simple analysis panel for mass runs. After completion: which variable had the most impact on score variance? Show a bar chart with average score per variable value for each variable independently. If `tone=formal` consistently scores higher than `tone=casual` across all other variables, that's visible immediately. This is a simple marginal effect analysis — compute it by grouping combinations by each variable's value and averaging scores.

**Day 29 — Constants UI + Model Variable**

Add support for constants (values that are substituted but not varied — useful for context that's fixed). Also add the model as a possible variable — users can include `{{model}}` and define multiple model names as values, turning model comparison into part of the grid search. Wire this through the execution engine.

**Day 30 — Integration + Deploy**

Full end-to-end mass prompting test. Template with 3 variables, 2-3 values each, rubric attached, run it, verify progress, check results grid, save a winner as a version. Deploy. **Mass prompting is live.**

---

## Phase 4 — Prompt Breakdown

**Duration: Days 31–44 | Outcome: Visual prompt-to-output attribution map**

This is the most complex feature in the Playground. Read the design carefully before starting Day 31.

**The Core Idea (reiterated clearly):** The user writes a large prompt. The system segments it into labeled sections — goals, context, constraints, tone instructions, examples, decision points, etc. It executes the prompt and gets the output. It then maps each section of the prompt to the portion(s) of the output it most influenced, and renders this as a two-column visual where you can draw a line from any prompt segment to its corresponding output section.

---

**Day 31 — Auto-Segmentation Design**

Design the segmentation strategy. There are two modes you'll support: manual (user draws their own segment boundaries using a delimiter syntax you define, e.g. `### [GOAL]`) and automatic (an LLM analyzes the prompt and segments it). Define your segment type taxonomy: `goal`, `context`, `constraint`, `tone`, `example`, `decision_point`, `input`, `output_format`, `other`. Write the system prompt for the auto-segmenter — it should return a JSON array of segments, each with `type`, `label`, `content`, and `start_char`/`end_char` offsets into the original prompt string.

**Day 32 — Segmentation Backend**

Build the segmentation service. `POST /breakdown/segment` accepts a prompt string and a `mode` (auto or manual). Auto mode calls the LLM segmenter and returns the segment array. Manual mode parses the delimiter syntax. Normalize both paths to the same output format. Store segmentation results in a `prompt_segments` table linked to a `breakdown_runs` table.

**Day 33 — Attribution Engine: Design**

This is the intellectually hard part. Design the attribution approach. The method: for each segment, run an ablation — execute the full prompt, then execute the prompt with that segment removed. The delta in output is the attribution signal. For a more nuanced approach, also run the prompt with only that segment (stripped of all others). From the three outputs (full, with segment, without segment), use an LLM to determine what output content that segment was responsible for. Build this as a formal interface before writing any code.

**Day 34 — Attribution Engine: Implementation (Part 1)**

Build the ablation runner. For a prompt with N segments, this generates N+1 executions (the full prompt + one run per segment removal). Run them in parallel. Store each ablation output linked to the segment it ablated. This will be expensive in API calls for large prompts — add a cost estimate warning in the UI before the user runs it.

**Day 35 — Attribution Engine: Implementation (Part 2)**

Build the attribution analyzer. Takes the full output and all ablation outputs. For each segment, asks an LLM judge: "Given that removing this segment changed the output in the following way [diff], which parts of the final output does this segment most contribute to? Return character offsets into the final output." Store attribution records as `(segment_id, output_start_char, output_end_char, confidence_score)` rows.

**Day 36 — Output Segmentation**

The output also needs to be segmented for the visualization to work cleanly. Run the LLM output segmenter on the final output — same taxonomy roughly, but adapted for outputs (conclusions, explanations, examples, caveats, recommendations, etc.). The output segments are what get highlighted in the right column of the visualization.

**Day 37 — Data Model + API**

Design the final breakdown data model. A `breakdown_runs` table with the original prompt, the execution output, and status. A `prompt_segments` table (linked to the run). An `output_segments` table. An `attributions` table storing `(prompt_segment_id, output_segment_id, confidence)` many-to-many relationships. Build `GET /breakdown/{run_id}` which returns all of this in a single response object the frontend can render.

**Day 38 — Breakdown UI: Left Column (Prompt Segments)**

Build the left column of the breakdown visualization. The prompt is displayed as a series of labeled, color-coded blocks stacked vertically. Each block corresponds to a segment. The color indicates the segment type (goals = blue, constraints = orange, examples = purple, etc.). Each block shows the segment label and truncated content. Hovering a block highlights it. Use a consistent color palette — define it once and use it everywhere.

**Day 39 — Breakdown UI: Right Column (Output Segments)**

Build the right column. The output is displayed similarly — segmented blocks, each labeled by what kind of output content it is. The colors are lighter versions of the same palette (since output segments are derived from prompt segments). Hovering an output segment shows which prompt segment drove it.

**Day 40 — Breakdown UI: Connection Lines (The Core Visual)**

This is the signature visual. When the user hovers or clicks a prompt segment, SVG lines are drawn from that segment's right edge to every output segment it influenced, weighted by confidence (thicker line = higher confidence). When they hover an output segment, lines go the other direction — back to the prompt segments that drove it. Use an SVG overlay positioned absolutely on top of the two columns. Calculate the Y midpoints of each block and draw bezier curves between them. Color the lines to match the segment type color.

**Day 41 — Segment Editor**

Build the manual segment editor mode. The user can edit the auto-detected segments — rename them, change their type, split a segment, merge two adjacent segments, delete a segment. After any edit, there's a "Re-run attribution" button that re-runs only the affected ablations rather than the whole thing. Changes are tracked so the user can see how redefining segments changes the attribution picture.

**Day 42 — Breakdown Mode Selection + Execution Flow**

Build the full user flow for starting a breakdown. From any prompt version, the user clicks "Run Breakdown." They choose: auto-segment or manual. If auto, they see a preview of the detected segments and can adjust. They see the cost estimate (number of API calls). They confirm. Progress bar shows ablation runs completing. When done, they land on the visualization.

**Day 43 — Breakdown History + Comparison**

Allow the user to run breakdown on multiple versions of the same prompt and compare. A side-by-side mode shows two breakdown visualizations next to each other. This lets the user see how refining the prompt changed the attribution picture — whether a segment that previously had weak attribution now drives more of the output clearly.

**Day 44 — Integration + Deploy**

Full end-to-end breakdown test on a real complex prompt (use a 500+ word prompt with multiple sections). Verify segmentation accuracy, attribution makes intuitive sense, visualizations render correctly. Fix visual bugs. Deploy. **Prompt Breakdown is live.**

---

## Phase 5 — The Perfect Prompt

**Duration: Days 45–52 | Outcome: AI-synthesized optimal prompt from all historical data**

---

**Day 45 — Data Aggregation Layer**

Build the aggregation query layer. For a given project and goal, assemble: the top 10 highest-scoring prompt versions, the top 5 A/B test results by score, the top 3 mass run combinations, the winning breakdown attributions (which segments had highest output coverage). This data becomes the context for the synthesis. Write this as a single `GET /projects/{id}/performance-summary` endpoint that returns a structured JSON object.

**Day 46 — Goal Definition UI**

Add a goal definition step to the Perfect Prompt flow. The user writes a natural language statement of what they want the optimal prompt to achieve: "I want a prompt that extracts action items from meeting transcripts with maximum completeness and structured formatting." They can also select a rubric to define the scoring standard. This goal statement becomes part of the synthesis prompt to the LLM.

**Day 47 — Synthesis Engine**

Build the synthesis LLM call. The system prompt instructs the model to act as a prompt optimization expert. It receives: the goal statement, the top-performing historical prompts verbatim, the scores and criterion breakdowns of each, the winning variable combinations from mass runs, and the breakdown attribution insights (which structural elements of past prompts drove output coverage). The model is instructed to synthesize a new prompt that adopts the strongest structural elements from the historical best performers, optimized for the stated goal. Output is the suggested prompt text plus an explanation of what it borrowed from each source.

**Day 48 — Suggestion Review UI**

Build the suggestion presentation page. Show the suggested prompt in the CodeMirror editor with annotations — use a sidebar panel that highlights different sections of the suggested prompt and explains the reasoning behind each (e.g. "This constraint section was derived from Version 7 which scored 91 on conciseness"). The source attribution is important — it makes the suggestion trustworthy and auditable rather than a black box.

**Day 49 — Accept, Edit, and Save Flow**

The user can: accept the suggestion as-is (saves as a new prompt version with a special "perfect_prompt_generated" tag), edit it inline and then save (common case — the suggestion is a strong starting point but needs tweaking), or reject it and try with a different goal statement. When saved, the new version is linked in the database back to the performance summary that generated it — full provenance chain.

**Day 50 — Iteration Loop**

Add the ability to run Perfect Prompt iteratively. After saving the suggested prompt, the user can run it through A/B testing or scoring immediately from the same screen. If the score improves, that new data feeds back into the historical pool. Re-running Perfect Prompt will now include the newest results. This creates a virtuous loop: score → mass prompt → breakdown → perfect prompt → score again.

**Day 51 — Perfect Prompt History**

Show a history of all Perfect Prompt suggestions for a project — what was suggested, when, what goal it was targeting, and how the resulting prompt actually scored after testing. This lets the user see whether the synthesis is getting better over iterations as more data accumulates.

**Day 52 — Final Integration, Polish + Full Deploy**

Full system end-to-end run. Start with a blank project. Write a prompt. Version it. Run A/B tests. Score. Mass prompt. Run Breakdown. Generate a Perfect Prompt. Score the generated prompt. Verify the entire data chain is intact and traceable. Polish any rough UI edges. Write a proper onboarding flow for new users (a brief guided tour highlighting the key features in order). Deploy the full system to production. **The Playground is complete.**

---

## Technology Reference

**Frontend:** Next.js 14 (App Router), TypeScript, Tailwind CSS, CodeMirror 6, Recharts, React Flow (for breakdown SVG), Zustand (state management), React Query (server state).

**Backend:** FastAPI (Python), SQLAlchemy (ORM), Alembic (migrations), Celery + Redis (background jobs for scoring and ablation runs), Pydantic (request/response models).

**Database:** PostgreSQL (primary store), Redis (job queue + caching).

**LLM Clients:** OpenAI Python SDK, Anthropic Python SDK, both using async clients.

**Infrastructure:** Docker Compose (local), Railway or Render (backend), Vercel (frontend), Supabase or Neon (managed Postgres in prod).

---

## Phase Summary

| Phase | Days | Features Shipped | Key Milestone |
| --- | --- | --- | --- |
| 1 | 1–14 | Prompt Versioning + A/B Testing | First live deployable product |
| 2 | 15–21 | Prompt Scoring | Data starts compounding across runs |
| 3 | 22–30 | Mass Prompting | Grid search over prompt space |
| 4 | 31–44 | Prompt Breakdown | Visual prompt→output attribution |
| 5 | 45–52 | The Perfect Prompt | Full synthesis loop complete |

Total: **52 working days** from scratch to a fully deployed Playground.

# Playground Standards

[Playground — Engineering & Deployment Standards](https://www.notion.so/Playground-Engineering-Deployment-Standards-318d366d1af480d49015ed88c5aaedc1?pvs=21)

# Risks

[Risks](https://www.notion.so/Risks-318d366d1af480d08b3df34969bc9695?pvs=21)

[File structure](https://www.notion.so/File-structure-318d366d1af480ce8e36f36f84d4e30c?pvs=21)