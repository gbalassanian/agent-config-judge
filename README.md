# agent-config-judge

A two-tier detector for misconfigured ElevenLabs conversational voice agents,
built to run against a portfolio of accounts rather than one agent at a time.

## The problem this exists to solve

A misconfigured voice agent usually looks fine. It answers calls, it doesn't
crash, and every config checkbox is ticked. The only way to find out it's
actually broken is to read what happened when a real user hit the specific
edge the config silently fails on — and nobody reads every transcript across
hundreds of accounts.

This repo is a working answer to that, built as a case study against my own
ElevenLabs workspace, not synthetic examples pretending to be one. Two tiers:

1. **Cheap pass** — proxy signals score every agent in the portfolio. No
   transcripts read, no LLM called. Its only job is deciding who gets read in
   depth. It over-flags on purpose: a false flag costs one judge call; a
   missed broken agent costs a client.
2. **LLM judge** — reads the config *and* the transcripts of whatever got
   flagged, scores nine criteria with cited evidence, names a cause for every
   failure, and maps that cause onto a recipe catalog. A cause with a known
   fix is **standard** (automatable); a cause with no known fix is
   **systemic** (needs a human) — by construction, not by judgment call.

A router then turns (classification × account ARR) into exactly one action.
Detection is 100% automated; anything that would touch a customer's live
agent comes back `requires_human_approval: true`.

## The finding that shaped this design

While pulling real data to build this repo's fixtures, I found this live in
my own workspace, on an agent called **Onboarding Assistant**
(`agent_3701kz9j7j4ffmcbxzdeq0a9h1t0`), running on the `react_sdk` channel.
A user asked to be transferred to a sales rep about pricing. The agent said:

> "Please wait while I connect you with a sales representative who can help
> you with pricing information."

It called `transfer_to_number`. The tool came back:

> `"error":"Transfer to number tool is only available for phone calls
> powered by Twilio, Exotel, or SIP trunking"`

The agent recovered gracefully in speech — "I am sorry, but I am unable to
transfer calls at this moment..." — but the user's actual request (a human,
for a pricing question) was never met. **Every config check on this agent
passes.** `transfer_to_number` is a real, valid, fully-configured tool. The
only place this failure is visible is the transcript, on the exact channel
this agent runs on. This is not a hypothetical from the case study brief —
it's this repo's own portfolio, and it's why the judge tier exists at all:
*a tool being configured is not the same as a tool that works.*

(Full parsed evidence, redacted only for phone numbers, is in
`fixtures/real_portfolio_snapshot.json` / `fixtures/recorded_judgements.json`;
provenance and the raw shapes pulled are in `scripts/build_real_fixture.py`.)

## What else the real portfolio turned up

Five real agents got scanned (see "Running it" below for how to reproduce
this yourself against the bundled fixtures, zero API keys required):

| Agent | Cheap-pass score | Classification | What's actually wrong |
|---|---|---|---|
| Onboarding Assistant | 40.0 (forced) | standard | The case above, plus a system prompt that's just `"Eres un asistente útil"` — long enough to dodge a naive length check, but no bounded role at all |
| No Borders - Intake | 57.8 | standard | System prompt literally says *"ofrecé pasarlo con una persona"* (offer to hand off to a person) if the customer is upset or the case is complex — but **zero transfer tools are configured anywhere**. This one config check alone catches it; no transcript needed |
| No Borders - Operations Copilot | 51.1 (forced) | standard | A webhook tool (`crm_lookup`) returned HTTP 401 twice in the same conversation, and the agent retried the *identical* query verbatim before giving up — the live instance of a cause (`multi_turn_repeats_failed_tool_call`) added to the catalog after finding it here |
| Customer Support Agent | 31.1 | standard | Completely empty: blank system prompt, no tools, no KB, zero conversations ever. A real, unconfigured stub, not a synthetic example |
| Recruiter Agent | 42.2 | **healthy** | Cheap pass flags missing knowledge_base and human_handoff — but this is a one-user internal recruiting roleplay tool with the full job description and candidate CV embedded directly in the prompt. Neither criterion actually applies, and the judge correctly walks the flag back to healthy |

The Recruiter Agent row is the one genuinely interesting non-failure: it's a
live demonstration of the cheap pass's designed-in over-flagging, and of the
judge tier correctly recognizing when a rubric criterion doesn't fit an
agent's real job (a distinction that has nothing to do with "trusting"
config — it comes from actually reading the prompt).

**ARR values used anywhere in this repo are a synthetic annotation for the
router demo**, not real revenue figures — the ElevenLabs Agents API has no
concept of account revenue, so there was nothing real to pull.

## Architecture

```
agent_config_judge/
  rubric.py             nine criteria (data) + the recipe catalog (the standard/systemic boundary)
  models.py             AgentConfigSnapshot / AggregateMetrics / ConversationRecord — see below
  cheap_pass.py          tier 1: proxy scorer over config + aggregate metrics only
  judge.py               tier 2: prompt, strict JSON contract, evidence-enforcing validator
  router.py              classification x ARR -> exactly one action
  elevenlabs_client.py   real Agents API client + mechanical metrics derivation
  pipeline.py            wires the three tiers together
  cli.py                 `agentjudge scan|fetch-portfolio|demo|show-recipe`
eval/
  golden_set.py          19 labeled cases: 5 real + 14 synthetic (2 required false-positive traps + more)
  run_eval.py             the harness — three separate scorecards, see "Eval results"
fixtures/
  real_portfolio_snapshot.json   the 5 real agents above, phone numbers redacted, everything else as it came
  recorded_judgements.json        genuine judge output for all 19 golden-set cases (see below)
scripts/
  build_real_fixture.py            provenance: how the real fixture was built from raw API shapes
  produce_recorded_judgements.py    provenance: how the recorded judgements were produced
tests/                            26 tests on the load-bearing contract rules (see "Running the tests")
```

### The nine criteria

Three are checkable from config alone; six only show up in transcripts —
config can pass all three and the agent can still be broken (see the finding
above).

| Criterion | Source | What "healthy" looks like |
|---|---|---|
| `system_prompt` | config | Clear, bounded role with explicit limits. A prompt that exists but contradicts itself is FAIL, not PASS |
| `knowledge_base` | config | At least one source connected, if the job needs one |
| `human_handoff` | config | A transfer path that **works on the channels this agent actually runs on** |
| `fallback` | behavior | Says it doesn't know, *then* escalates — in that order |
| `grounding` | behavior | Specific claims (numbers, limits, prices, policies) are attributable; conversational filler needs no source |
| `multi_turn` | behavior | Holds context; doesn't re-ask what it already has, doesn't loop on edge cases |
| `escalation_health` | behavior | Neither zero nor runaway |
| `sentiment` | behavior | No frustration *caused by the agent* — a user angry about something unrelated isn't a failure |
| `latency` | behavior | Within band for the channel; slow turns get a shared-cause note when one exists |

### The recipe catalog is the standard/systemic frontier

`rubric.RECIPE_CATALOG` maps a `cause_code` to a fix and a tier
(`self_serve`, batchable via a docs link; `nudge`, needs an account-specific
message). **A cause with no entry is systemic by definition.** The judge
names a cause; `judge.validate_judge_output()` looks it up and recomputes
classification from that mapping every time — it never trusts a
classification the model might claim on its own, and a cause_code that
isn't in the catalog gets deleted, dropping that agent to systemic even if
every other failure it named was mapped. Three failures that all map to
known recipes are still "standard"; one unmapped failure makes the whole
agent "systemic," regardless of how many other criteria passed.

Adding a recipe is how the system gets cheaper over time. Two got added
while building this repo, from causes the real data actually produced:
`multi_turn_repeats_failed_tool_call` (identical tool retry after a real
401) and `system_prompt_too_generic` (a prompt that exists, isn't
contradictory, just defines no role — the real `"Eres un asistente útil"`
case). `agentjudge show-recipe <cause_code>` looks one up from the CLI.

### Evidence is enforced, not requested

Every judge verdict needs a short verbatim transcript quote or a named
config field. `judge.py`'s validator does two things a prompt instruction
alone can't guarantee:

- A pass/fail with no evidence is downgraded to `unknown` — not trusted.
- **The quote is checked against the actual transcript sample**, not just
  checked for presence. A plausible-sounding but fabricated citation is
  discarded and the verdict downgraded, exactly like a missing one.

### Two judge backends, one validator

`LiveJudgeBackend` calls the real Anthropic API (needs `ANTHROPIC_API_KEY`).
`RecordedJudgeBackend` replays saved raw outputs from a JSON fixture, keyed
by agent_id. Both return an *unvalidated* dict and pass through the exact
same `validate_judge_output()` — recorded is not a convenience mock, it's how
the eval stays reproducible when the rubric or the recipe catalog change:
re-run against the same saved evidence and get a genuinely re-derived
classification.

### How the recorded judgements were produced

This matters enough to say plainly rather than bury in a code comment: this
build session had no `ANTHROPIC_API_KEY` available to hand to a script — the
sandbox this repo was built in has authorized tool access to Claude and to
this ElevenLabs workspace, but not a raw API key it could pass to a
subprocess making its own HTTP calls. Rather than fabricate plausible-looking
judge output by hand to fill `fixtures/recorded_judgements.json`, the model
building this repo worked through `judge.build_judge_prompt()`'s exact
contract for each of the 19 golden-set cases — reading each config and
transcript cold and answering the same nine questions a scripted API call
would have asked, under the same evidence rules the validator enforces
(`scripts/produce_recorded_judgements.py` re-validates every entry, quote
check included, before writing the fixture — a typo'd citation fails loudly
at build time, not silently at eval time).

That is a real judging pass, not a mock — but it is **not independent
validation**. The same reasoning process wrote the golden set's ground
truth labels and then, separately, worked through the judge prompt for the
same cases. Some convergence is inevitable and not meaningful; see "Eval
results" below for exactly how much and what that does and doesn't prove.
Anyone with a real `ANTHROPIC_API_KEY` can run `--backend live` and get a
genuinely independent number — that's the whole reason the two backends
share one validator.

## Running it

### Zero API keys

```bash
pip install -r requirements.txt
PYTHONPATH=. python3 -m agent_config_judge.cli demo
```

Runs the full pipeline (cheap pass → judge → router) against the real,
redacted 5-agent portfolio snapshot and the recorded judgements shipped in
`fixtures/`. This is exactly the output quoted in the table above.

```bash
PYTHONPATH=. python3 eval/run_eval.py
```

Runs the eval harness against all 19 golden-set cases (see "Eval results").

### Running the tests

```bash
pip install -r requirements.txt
pytest
```

26 tests covering the load-bearing contract rules — evidence enforcement
(including the fabricated-quote check), the recipe-mapping-owns-
classification rule, all four router branches, the forced-flag rule, and a
regression guard on the golden set (cheap-pass recall must stay perfect;
the two required false-positive traps must keep resolving to healthy).

### With your own ElevenLabs workspace

```bash
cp .env.example .env   # fill in ELEVENLABS_API_KEY
PYTHONPATH=. python3 -m agent_config_judge.cli fetch-portfolio --out my_snapshot.json
PYTHONPATH=. python3 -m agent_config_judge.cli scan --snapshot my_snapshot.json --backend recorded --fixture fixtures/recorded_judgements.json
```

The last command will error on any agent not already in the recorded
fixture (a different workspace has different agent_ids) — that's
intentional, not a bug: recorded replay only works for agents it has
judgements for. Add `ANTHROPIC_API_KEY` and pass `--backend live` to judge a
real new portfolio instead.

### Full pipeline with both real keys

```bash
PYTHONPATH=. python3 -m agent_config_judge.cli scan --snapshot my_snapshot.json --backend live
```

## Eval results

Run: `PYTHONPATH=. python3 eval/run_eval.py`, against the 19-case golden set
(5 real portfolio agents + 14 synthetic, see `eval/golden_set.py` — every
synthetic case is labeled as such in the file and carries a note on exactly
what failure shape it stands in for, including two required false-positive
traps: a grounding claim traced to a user-supplied number and a tool call
rather than a KB doc, and a zero-escalation-rate agent that actually
escalates through a ticket-creation tool the proxy metric doesn't count).

```
--- cheap pass (recall first, precision second) ---
n=19  TP=18 FP=1 TN=0 FN=0
recall:    100%
precision: 95%

--- judge: classification accuracy + false-positive rate on healthy agents ---
n=19  accuracy=100%
of 5 agents that are actually healthy, judge wrongly flagged 0 (0%) as standard/systemic
within the 30% recalibration threshold

--- judge: precision/recall on the specific failing criteria it names ---
criterion-level TP=18 FP=0 FN=0
recall:    100%
precision: 100%
of 16 correctly-identified mapped failures, cause_code matched ground truth in 100%
```

**Read this number honestly, not as a finished accuracy claim:**

- **n=19 is small.** These are directional numbers about whether the
  *mechanism* works on the cases it was built to cover, not a claim about
  real-world judge accuracy at portfolio scale. A meaningful accuracy claim
  needs a golden set at least in the low hundreds, ideally labeled by
  someone other than whoever wrote the detector.
- **The near-perfect scores mostly reflect shared authorship, not
  independent validation** — see "How the recorded judgements were
  produced" above. Ground truth and judge output came from the same
  reasoning process in one sitting. That process was genuinely careful (it
  found and fixed two real modeling gaps along the way — see below — rather
  than being steered to agree with itself), but it was never blind, and a
  live API call against labels it never saw would be a materially harder
  test. I did not try to manufacture disagreement to make this section look
  more rigorous; that would be a different kind of dishonesty. The honest
  version is: treat these numbers as "the mechanism is internally
  consistent," not "this judge is 100% accurate."
- **The one real miss in this run is genuine, not staged**: the cheap pass
  flags `synthetic_healthy_agent` — a case built to be fully healthy — as
  needing review. Root cause: with a single sampled conversation, one
  justified escalation reads as a 100% escalation rate, which trips the
  "runaway" ceiling check in `cheap_pass.py`. That's a real placeholder-
  threshold artifact at small sample sizes, exactly the kind of thing
  `cheap_pass.py`'s own comments say isn't calibrated yet.
- Building the golden set surfaced two real gaps in the rubric/catalog,
  fixed in place rather than worked around: a genuine third `system_prompt`
  failure shape needed its own recipe (`system_prompt_too_generic` — the
  real `"Eres un asistente útil"` case), and an early draft of the
  grounding false-positive trap accidentally smuggled in a real,
  un-grounded claim, caught only by actually re-deriving the aggregate
  metrics and re-reading the transcript rather than trusting the label I'd
  already written.

## Limitations, weakest first

1. **The eval numbers above are not independent validation** (see "Eval
   results"). This is the single biggest asterisk on everything in this
   repo. Getting a real accuracy number requires a live `ANTHROPIC_API_KEY`
   run against a golden set labeled by someone who didn't write the judge
   prompt.
2. **The evidence contract has no field for tool-call-level evidence.** A
   verdict can cite a transcript quote or a config field, but not "this
   tool was called twice with an identical error" — which is the actual
   mechanism behind two real recipes (`multi_turn_repeats_failed_tool_call`
   found on Operations Copilot, `latency_shared_slow_tool`). The judge has
   to cite an adjacent spoken-text quote instead of the real evidence,
   which is weaker than it should be. Closing this needs a third evidence
   type, not a prompt tweak.
3. **Every threshold in `cheap_pass.py` and `router.py` is an unvalidated
   placeholder** — explicitly marked as such in comments, and the eval run
   above already found one small-n failure mode in the escalation-health
   ceiling check. None of these have been tuned against a real labeled
   portfolio at scale; `ARR_HIGH_THRESHOLD_USD` in particular is a guess
   with no input from anyone who owns real account economics.
4. **The cheap pass's channel/tool cross-check for `human_handoff` is a
   partial heuristic, not a guarantee** — it only catches a
   `transfer_to_number`/channel mismatch when the sample happens to include
   observed channels. The *actual* backstop for "a configured tool doesn't
   work at runtime" is the forced-flag rule (any tool error in the sample
   forces a judge read regardless of score), which is a strictly narrower
   promise: it catches every tool that has already failed at least once in
   the sample, not every tool that theoretically could.
5. **Small conversation samples make several proxies noisy by
   construction** — `escalation_health`'s rate check is meaningless at n=1
   (see the eval finding above), and the same is true of `multi_turn`'s
   repeat-rate and `sentiment`'s frustration-rate proxies. This repo has no
   minimum-sample-size guard before trusting these proxies' verdicts, only
   before trusting nothing (the "no signal → unknown" rule already in
   `cheap_pass.py`).
6. **The rubric doesn't distinguish internal tools from customer-facing
   agents.** `human_handoff` and `knowledge_base` assume there's a customer
   on the other end who might need a human or a documented policy. Applied
   literally to an internal ops tool (Operations Copilot) or a one-user
   roleplay tool (Recruiter Agent), both get flagged for gaps that may not
   actually matter — the judge can reason its way past this per-agent (as
   it does for the Recruiter Agent), but nothing upstream of the judge
   knows to skip the check for an agent type where it doesn't apply.
7. **Sampling policy is undefined.** How many conversations per agent, how
   recent, how to handle an agent with thousands of conversations — none of
   this is decided here. The fixture's Recruiter Agent has zero sampled
   conversations simply because none were pulled while building this repo,
   which is itself a small demonstration of the gap: an agent can look
   clean purely because nobody happened to sample it.
8. **The `grounding` criterion's wording ties attribution specifically to
   "the KB,"** but this repo's own real data (Operations Copilot) shows a
   healthy pattern where facts come from a live tool call instead. The
   judge reasons past this correctly per-case, but the rubric text itself
   hasn't been updated to say "KB or tool call," which is what it actually
   means in practice.
9. **No live-write path exists, and none should be added lightly.** Every
   router action today produces an artifact for a human or customer to act
   on; `RouteAction.touches_live_agent` is `False` everywhere. That keeps
   `requires_human_approval` vacuously true-when-it-matters rather than
   load-bearing — the day an automated self-serve fix actually writes to a
   customer's agent, that table entry (and the approval flow around it)
   needs to be taken much more seriously than a boolean flip.
