# Message Notification Router — Deep-Dive Comparison Analysis

> **Scores:** Your repo → **11.8 / 30** · Friend's repo (trickymind1324) → **21.2 / 30**
> **Score gap: +9.4 points** — a ~80% relative advantage for the friend.

---

## 1. High-Level Summary

| Dimension | Your Repo | Friend's Repo |
|---|---|---|
| **Score** | 11.8 / 30 | 21.2 / 30 |
| **LLM used** | Google Gemini 2.0 Flash | Anthropic Claude (tool-use / structured outputs) |
| **Architecture shape** | Safety → Evidence → LLM → Utility Matrix → Postprocess | Perception (cached) → Triggers → Evidence → Reasoning (cached) → Merge Layer |
| **Reproducibility** | Depends on live API; no output cache committed | **Bit-identical reproduction with zero API calls** via committed caches |
| **Deterministic authority** | Safety runs first; LLM result feeds a utility score matrix | Triggers run **twice** (before LLM, and again in merge); merge layer has **final authority** |
| **Model output validation** | Basic JSON parsing with fallback | **Schema-enforced tool-use**, ≤2 retries with validation error injected, then deterministic fallback |
| **Self-check / reflection** | ❌ Not implemented | ✅ One bounded self-check pass if confidence low / evidence invalid / verdict conflicts trigger |
| **Prompt injection defense** | Regex patterns in safety module | Stripped **before model sees text** AND re-evaluated independently by trigger layer |
| **Media handling** | Passes image/audio path to Gemini; no ASR pre-processing | Separate **perception stage**: Claude vision for images + local faster-whisper ASR for audio, all **cached to disk** |
| **Evidence retrieval** | Cosine similarity on sentence-transformers embeddings | Deterministic weighted scoring (same-user +3, same-sender/business +2, content-pattern +2, event-polarity +1) — top-8 to reasoner |
| **message_type ownership** | LLM decides message_type; postprocess adjusts | **Deterministic layer owns message_type** — model cannot override |
| **Evaluation harness** | No separate evaluator | Full `evaluation/main.py` — sample scores, confusion matrix, trigger counts, frozen rule suite + merge fixtures, **determinism double-run** |
| **Codebase depth** | ~8 Python files, ~50-60KB total | ~9 lib modules (triggers.py alone is 19.5 KB), eval harness, docs/, audit trail per message |

---

## 2. Friend's Architecture — Detailed Breakdown

### Stage 0 — Context Build
All CSVs are joined **per message** into a rich enriched context object (`data.py`). This includes:
- User stats + quiet-hours flag
- Group + both memberships (user's role, muted state)
- Business + user-business relation (opted out, days since interaction)
- Cross-user sender reputation
- Per-user history index

### Stage 1 — Perception (cached in `code/cache/perception/`)

This is one of the **biggest differentiators**.

- **Images**: One Claude vision call per unique image that extracts OCR text, description, and **safety flags** (phishing QR, suspicious URL, credential ask, prize bait). Results schema-enforced and committed to disk.
- **Voice notes**: Local `faster-whisper small` (int8, greedy decoding) transcribes audio into text. ASR transcripts are **committed to the repo** — so the evaluator never needs a GPU or ASR runtime. An optional Claude text call then classifies urgency/request type from the transcript.
- **Key design insight**: The perception stage **never sees routing context**. It only describes what's in the media. This prevents prompt injection through images/audio from influencing perception output itself.

### Stage 2 — Trigger Scan (`lib/triggers.py`, 19.5 KB)

This is a single, shared `scan()` function that computes **12 scam/spam/chain/injection triggers (T1–T12)** and **3 guards (G1–G3)**:

| Trigger class | What it catches |
|---|---|
| T1 – credential ask | OTP/PIN/password requests from unknown/unverified source |
| T2 – QR/fee payment lure | Scan-this-QR, pay to release, clearance fee |
| T3 – prize/refund bait | Lottery won, claim your reward, government refund |
| T4 – lookalike domain | Domain with hyphens, mismatched from official, impersonates known brand |
| T5 – admin impersonation | Non-admin pretending to be admin in message text |
| T6 – chain blast | Forwarded ≥5 + share-in-N-groups language |
| T7 – prompt injection | Router-directed override text (stripped before LLM sees it) |
| T8-T12 – further scam patterns | Account block, KYC forced, suspicious deeplinks, etc. |
| G1 – opted-out promo protection | Verified business promo to user who opted out → always mute |
| G2 – promotion-never-notify | Promotional messages cannot escalate to notify regardless of confidence |
| G3 – quiet-hours cap | Even urgent messages capped to digest during quiet hours unless critical threshold |

**Critical feature**: The trigger scan runs **twice** — once before the LLM (to strip injection text and get a baseline verdict), and **again inside the merge layer** using the same shared function. This prevents any possibility of the two passes drifting out of sync.

### Stage 3 — Evidence Retrieval (`lib/evidence.py`)

Deterministic **weighted scoring** over `message_history.csv`:

```
+3  same user
+2  same sender OR same business
+2  content-pattern match
+1  event polarity (dismissed/reported/opened/replied)
```

Top-8 candidates pass to the reasoner. This is more **precise and auditable** than a cosine similarity approach because the weights are interpretable and testable.

### Stage 4 — Reasoning Call (`lib/reason.py`, `lib/llm.py`)

- Uses **Anthropic Claude** with **forced tool-use** (constrained decoding) → outputs are schema-enforced from the model's side, not repaired after the fact.
- Validates the output; if invalid → retry with validation error injected ≤2 times → deterministic fallback.
- **One bounded self-check pass** (re-reads media or fetches +8 more history rows) is allowed if:
  - Confidence is low (< threshold)
  - Evidence IDs are invalid/hallucinated
  - Verdict conflicts with a fired trigger
- Hard cap: maximum 1 extra self-check pass.

### Stage 5 — Merge Layer (`lib/merge.py`, 15.2 KB) — The Real Authority

This is perhaps the **most architecturally significant module**. It implements "the model proposes, deterministic code disposes":

- Trigger verdicts always override the model.
- Personalization clamps:
  - Quiet hours → cap notify to digest for non-critical
  - Muted-group membership → cap to digest unless direct @mention
  - Opted-out promotions → always mute regardless of model
  - Promotions-never-notify → hard rule
- Evidence IDs are re-validated (no hallucinated IDs pass through).
- **Confidence is clamped into calibrated bands** with grey-zone rows pinned to the band's low end (this explains better calibration scores).
- **The deterministic layer owns `message_type`** — the model can suggest but cannot set it unilaterally.

### Stage 6 — Output + Audit

- `dataset/output.csv` (filled template, same row order)
- `code/audit.jsonl` — per-message audit trail showing fired triggers, overrides, signals (this is gold for debugging and demonstrating the system understands what it's doing)

---

## 3. Your Repo's Architecture — Detailed Breakdown

### Stage 0 — Context Build (`context.py`)
Solid. Loads all CSVs and builds lookup indexes for users, groups, businesses, message history, group admins. Includes sentence-transformer embedding of history.

### Stage 1 — Safety Engine (`safety.py`)
Good design concept. Has:
- SCAM_PATTERNS (9 regex patterns)
- INJECTION_PATTERNS (10 patterns)
- FORWARD_GREETING_PATTERNS
- Business risk score computation

**Issue**: Runs only **once**, before the LLM. If the LLM produces a result that conflicts with safety logic, there's no second pass to enforce correctness. The merge is through a separate `utility_score()` function that doesn't re-run safety checks.

### Stage 2 — Evidence Retrieval (`evidence.py`)
Uses sentence-transformer cosine similarity to find relevant history messages. The approach is correct but:
- **Fewer candidates sent to LLM** (top-3 vs friend's top-8)
- No weighted scoring for same-sender / same-business relevance boost
- Prune step runs post-output, not integrated into reasoning

### Stage 3 — LLM Routing (`llm_router.py`, 10.7 KB) using Gemini
- No forced tool-use / schema enforcement — relies on `response_mime_type: application/json` which can still produce non-conforming output
- No retry-with-validation-error; falls back to rule router on failure
- **No self-check** pass for low-confidence or conflicting verdicts
- Passes media to Gemini inline (no pre-perception caching) — means the same image gets re-described on every run, losing determinism and wasting API quota

### Stage 4 — Rule-based Router (`rule_router.py`, 12.7 KB)
Actually well-written with good coverage of notify/digest/mute signals. But it's designed as a **fallback** (used only when LLM fails), not as the authoritative safety layer.

### Stage 5 — Utility Score Engine (`decision.py`, 11 KB)
The most unique part of your architecture. Computes four continuous signals:
- **S_trust** (sender/relationship strength)
- **S_urgency** (temporal + action pressure)
- **S_risk** (threat probability)
- **S_fatigue** (dismissal rate / notification overload)

Then fuses them with configurable weights (W1-W7 from config). This is a mathematically principled approach, but it has a critical weakness: the final action is decided by thresholds on a fused score, which means **the LLM's action label is not respected directly** — it's treated as one input among many signals. This produces inconsistency when the LLM is right but the numeric signal disagrees.

### Stage 6 — Postprocess (`postprocess.py`, 3.4 KB)
Basic confidence clamping and reason truncation.

---

## 4. Root Cause Analysis — Why The Score Gap Exists

### 🔴 Critical Differences (highest impact)

#### 4.1 — No Committed Cache = Non-Deterministic Evaluation
**Friend**: Results are identical whether run with or without an API key, because all model outputs are cached and committed to the repo. The evaluator can verify this by running twice and diffing.

**You**: Each run calls the live API. Rate limits, temperature, model version changes, or API unavailability can produce different outputs. This makes the system fragile under evaluation conditions.

> **Impact**: The evaluation framework likely penalizes non-determinism or runs the pipeline multiple times. Friend's approach scores consistently; yours may have been flagged.

#### 4.2 — No Pre-Perception Stage for Media
**Friend**: Images are perceived once via Claude vision (OCR + safety flags in a schema-enforced call), result cached. Voice notes are locally transcribed via faster-whisper (committed ASR transcripts). The reasoning stage receives **clean, structured media analysis**.

**You**: Raw image/audio file paths are sent to Gemini during the main routing call. Gemini does OCR "on the fly" but you get no safety flags from the media, no committed transcripts, and no separation of concerns.

> **Impact**: Friend's system extracts scam signals from images (phishing QR codes, suspicious URLs in posters) that your system misses because there's no dedicated image-safety-flag pass.

#### 4.3 — The Merge Layer vs. Utility Matrix
**Friend**: Deterministic trigger verdicts have final authority. The model proposes; deterministic code disposes. message_type is always set by the deterministic layer.

**You**: The utility score matrix fuses LLM signals and numeric scores into a combined decision. The weights (W1-W7 in `WEIGHTS.md`) are tunable, but because they were likely not tuned against the sample ground truth, the thresholds that determine notify/digest/mute may be miscalibrated.

> **Impact**: An incorrectly calibrated utility weight causes systematic errors across whole categories of messages (e.g., every business message digested when some should notify, or vice versa).

#### 4.4 — LLM Schema Enforcement and Reliability
**Friend**: Claude with forced tool-use produces schema-conforming JSON by construction. Retry with injected validation error allows the model to self-correct. Deterministic fallback only activates after 2 failed retries.

**You**: Gemini JSON mode can still produce non-JSON or misformatted output. On failure, the system falls back to `rule_based_route()`, which is a coarser signal that loses the nuance the LLM would have provided.

> **Impact**: A higher fallback rate to the rule router means more messages routed by coarse heuristics rather than nuanced LLM reasoning.

#### 4.5 — Trigger Dual-Pass vs. Single Safety Pass
**Friend**: The `triggers.scan()` function runs before the LLM (to strip injection text) and again in the merge layer. These two runs use the **same shared function** so they cannot drift.

**You**: Safety runs before LLM only. If the LLM's output conflicts with a safety signal (e.g., LLM says "notify" on a scam message), the utility matrix would need to catch this via S_risk — but the numeric threshold may not always override.

> **Impact**: Some scam/injection messages may have leaked through to notify/digest that should have been muted.

### 🟡 Moderate Differences (medium impact)

#### 4.6 — Evidence Retrieval Quality
**Friend**: Top-8 candidates with weighted deterministic scoring (same-user, same-sender, content pattern, event polarity). Evidence IDs re-validated in merge layer — hallucinated IDs are rejected.

**You**: Top-3 with cosine similarity. Post-output prune pass for irrelevance. No hallucination guard.

> **Impact**: Evidence score dimension of evaluation penalizes irrelevant or hallucinated IDs. Friend's re-validation prevents this completely.

#### 4.7 — Confidence Calibration
**Friend**: Confidence clamped into calibrated bands per action+type combination, with grey-zone rows pinned to the band's low end. The sample data calibration bands are derived from `sample_messages.csv` pattern analysis.

**You**: Basic clip between 0.7 and 0.95 in postprocess. No per-category calibration.

> **Impact**: Evaluators likely score confidence calibration. Miscalibrated confidence (e.g., 0.90 on a wrong prediction) may incur a double penalty.

#### 4.8 — message_type Authority
**Friend**: Deterministic layer sets message_type; model suggestion is overridden.

**You**: LLM sets message_type, postprocess may adjust. The rule_router also sets message_type when it's the fallback.

> **Impact**: If the LLM misclassifies message_type (e.g., "spam" vs "scam"), that contributes to message_type score loss. Having a deterministic override based on the trigger that fired would be more accurate.

### 🟢 Minor Differences (lower impact)

#### 4.9 — Self-Check Reflection
Friend's one bounded self-check pass allows the model to re-examine its verdict when there's a conflict or low confidence. This improves accuracy on edge cases without unbounded cost.

#### 4.10 — Evaluation Harness
Friend built a full `evaluation/main.py` that runs sample scoring, confusion matrix, trigger counts, frozen rule suite, and determinism double-run. This allowed iterative calibration **before final submission**. Your team likely relied on manual spot-checking.

#### 4.11 — Documentation and Transparency
Friend's system produces `code/audit.jsonl` — a per-message audit trail of fired triggers, overrides, and signals. This not only helps debugging but may factor into evaluator scoring of "reason usefulness" since the reason strings can be drawn directly from the trigger that fired.

---

## 5. Scoring Dimension Impact Map

| Scoring Dimension | Friend's Advantage | Why |
|---|---|---|
| **Action correctness** | 🔴 High | Dual-pass triggers + merge authority → fewer wrong action decisions |
| **message_type correctness** | 🔴 High | Deterministic layer owns type; derived directly from fired trigger |
| **Reason usefulness** | 🟡 Medium | Reason strings tied to specific trigger labels; more precise and consistent |
| **Evidence relevance** | 🟡 Medium | Re-validation of evidence IDs prevents hallucinated IDs; top-8 better coverage |
| **Confidence calibration** | 🟡 Medium | Per-band calibration vs. simple clip; reproducibility helps consistency |

---

## 6. What You Should Change (Priority Order)

> **For a potential future iteration or if you re-run:**

### Priority 1 — Build and commit output caches
Even if you re-run just the LLM calls once and commit the results, you gain reproducibility. This is the single highest-leverage change.

### Priority 2 — Make triggers the final authority (not input to utility matrix)
Restructure so that if a trigger fires, it **sets the output directly** and skips the utility matrix. The matrix is only for non-trigger cases. Re-run triggers after LLM to ensure no conflict passes through.

### Priority 3 — Add a perception stage
Before the main loop, pre-process all images and audio into structured perception JSON files (text, safety flags, transcripts). Commit them. The main loop reads from cache.

### Priority 4 — Fix message_type ownership
Make the trigger that fires also set the message_type authoritatively. Build a mapping: T1 → scam, T2 → scam, T3 → scam, T4 → scam, T6 → forward, injection → scam, etc.

### Priority 5 — Switch to schema-enforced LLM calls with retry
Either use Claude tool-use or Gemini's function-calling API with a strict JSON schema. Inject the validation error on retry. Keep the rule_router as last resort only.

### Priority 6 — Calibrate confidence per band
From `sample_messages.csv`, compute the actual confidence distribution per (action, message_type) combination. Clamp to those ranges rather than a flat 0.7–0.95 clip.

### Priority 7 — Add evidence ID re-validation
After retrieving evidence IDs, check that each ID exists in `message_history.csv` for the same user. Reject any that don't match.

---

## 7. Summary Table

```
                          YOUR REPO          FRIEND'S REPO
Score                     11.8 / 30          21.2 / 30
───────────────────────────────────────────────────────
Reproducibility           ❌ Live API         ✅ Committed caches, zero API key needed
Media Perception          ❌ Inline Gemini    ✅ Separate stage; cached; no routing context
Trigger Authority         ❌ One-pass input   ✅ Dual-pass, final authority
Model Reliability         ❌ JSON mode only   ✅ Tool-use + retry-with-error + self-check
Evidence Quality          ⚠️  Top-3 cosine    ✅ Top-8 weighted + hallucination guard
message_type Control      ❌ LLM sets it      ✅ Deterministic layer owns it
Confidence Calibration    ❌ Flat clip        ✅ Per-band calibration
Evaluation Harness        ❌ None             ✅ Full eval/main.py with sample scoring
Audit Trail               ❌ None             ✅ code/audit.jsonl per message
```

---

*Analysis based on reading the friend's full repo (README, code/README.md, code/main.py, lib module listing, cache structure) and your full codebase (main.py, rule_router.py, safety.py, decision.py, context.py, llm_router.py, postprocess.py, evidence.py, HR_ORCHESTRATE_APPROACH.md).*
