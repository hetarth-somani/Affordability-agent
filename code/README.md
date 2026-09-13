# Buy or Wait? — AI Financial Affordability & Decision Agent

Starter-to-production solution for the **HackerRank Orchestrate** Hackathon (September 2026).

---

## 1. Executive Summary

**Buy or Wait?** is an autonomous, deterministic, and AI-assisted financial decision engine. Given a user's expense or purchase request, the agent decides whether the user should:
- **ull_payment**: Pay the entire amount immediately (ffordable_now).
- **partial_payment**: Pay a safe partial amount on equest_date and the remainder on or before desired_completion_date (ffordable_with_plan).
- **installments**: Commit to a provider-offered installment schedule (ffordable_with_plan).
- **wait**: Defer the purchase until a specific conservative future date when full payment is safe (ffordable_later).
- **
ot_recommended**: Reject the transaction as unsafe within the 90-day horizon (
ot_affordable).

Every recommendation guarantees that the user's available balance **never breaches their required minimum balance** at any point across the 90-day forecast, after accounting for all recurring commitments, pending obligations, and essential spending.

---

## 2. System Architecture & Pipeline Flow

The engine is built with strict modularity, separation of concerns, and defensive evaluation:

`
┌─────────────────┐
│   dataset/      │ (Profiles, Events, Requests, Options, Rates, Messages, Images)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ data_loader.py  │ Indexed, typed O(1) DataBundle ingestion
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│evidence_agent.py│ Local Tesseract OCR (primary) + Groq Vision (fallback)
└────────┬────────┘ Multilingual message signal extraction (English & Bahasa Indonesia)
         │
         ▼
┌─────────────────┐
│ state_builder.py│ Conflict resolution, non-regular income exclusion, message-based
└────────┬────────┘ salary/rent updates, and calendar-month recurrence projection
         │
         ▼
┌─────────────────┐
│forecast_engine.py BalanceTimeline with O(log n) bisect queries and
└────────┬────────┘ intraday credit-before-debit chronological staging
         │
         ▼
┌─────────────────┐
│ plan_ranker.py  │ Combinatorial spending changes (1..3 items) + official 6-rule
└────────┬────────┘ tie-break ranking across full, partial, and installment options
         │
         ▼
┌─────────────────┐
│  validator.py   │ Independent field-by-field safety and contract validator
└────────┬────────┘ (Schema, chronology, bounds, schedule match)
         │
         ▼
┌─────────────────┐
│  explainer.py   │ Grounded natural-language explanation with fact-consistency
└────────┬────────┘ checking and deterministic fallback
         │
         ▼
┌─────────────────┐
│     main.py     │ Atomic write (.tmp -> output.csv) + evaluation/usage_report.md
└─────────────────┘
`

---

## 3. Module Breakdown

| Module | Role | Key Design Highlights |
|---|---|---|
| main.py | Pipeline Orchestrator | Coordinates the batch process for all 250 evaluation requests; performs atomic output writing via temporary file swap. |
| data_loader.py | Ingestion Engine | Reads and indexes all 7 CSV files into typed, immutable dataclass representations with (1)$ lookups by user and request. |
| models.py | Data Contracts | Strong typing for UserProfile, FinancialEvent, FinancialRequest, PaymentOption, and DataBundle. |
| evidence_agent.py | Evidence Resolution | **Local OCR first** (Tesseract) with direction-aware parsing (
et pay, alance due, mount in words); falls back to Groq Vision; extracts Indonesian and English message signals. |
| state_builder.py | Financial State Builder | Identifies recurrence via longest consistent chain; removes non-regular credits (bonuses, commissions, arrears); incorporates message signals (employment end, rent inflation, salary date revisions). |
| orecast_engine.py | Timeline Engine | Event-driven BalanceTimeline using isect for fast sublinear horizon checks; guarantees intraday credits are recognized before same-day debits. |
| x.py | FX Converter | Exact fixed exchange rate lookups pegged to settlement date and currency pair. |
| plan_ranker.py | Decision Engine | Generates plan candidates; performs exhaustive combination search across flexible spending levers (stop/reduce); ranks using the official 6-rule specification. |
| alidator.py | Contract Validator | Verifies all 8 fields against specification bounds, chronology, and payment option schedules; provides safe fallback on invalid states. |
| explainer.py | Explanation Agent | Generates concise, third-person explanations via Groq; validates every number against computed facts to guarantee zero hallucinated figures. |
| cache.py | Content-Addressed Cache | Disk cache keyed by hash of input data; prevents redundant VLM, OCR, and LLM calls. |
| llm_client.py | Inference Client | Resilient Groq client with exponential backoff and fine-grained token accounting. |
| usage_report.py | Cost Reporting | Summarizes full-dataset token usage, model calls, and cost metrics into evaluation/usage_report.md. |

---

## 4. Role of cache.py in the Main Pipeline

cache.py implements a persistent, content-addressed disk cache (.cache/). In the main pipeline:
1. **Image Evidence Caching**: Document OCR and vision extraction results are cached by (image_id, event_id). Re-running the pipeline incurs **0 additional API cost or OCR latency**.
2. **Message Evidence Caching**: Structured message interpretations are cached by message_id.
3. **Explanation Caching**: Explanations are cached by (request_id, fact_lines). Unchanged computed facts immediately retrieve verified explanations.

This ensures **deterministic execution, idempotency, fast execution, and zero unnecessary API spending**.

---

## 5. Setup & Running Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Optional: Tesseract OCR (if installed on PATH, used for local document reading)

### 1. Installation
Clone the repository and install the dependencies:
`ash
git clone https://github.com/interviewstreet/hackerrank-orchestrate-september26.git
cd hackerrank-orchestrate-september26

python -m venv env
# On Windows:
.\env\Scripts\activate
# On macOS/Linux:
source env/bin/activate

pip install -r code/requirements.txt
`

### 2. Environment Configuration
Copy the environment template and provide your Groq API key:
`ash
cp code/.env.example code/.env
# Edit code/.env and insert your GROQ_API_KEY
`

### 3. Run the Evaluation Pipeline
Execute the main entry point to process all requests in dataset/requests.csv:
`ash
python code/main.py
`
This produces:
- output.csv: Complete, validated predictions for all 250 requests.
- code/evaluation/usage_report.md: Detailed token usage and cost accounting report.

---

## 6. Financial Decision Principles & Safety Guarantees

1. **Balance Protection**: Available balance must remain $\ge \text{minimum\_balance\_to\_keep}$ on every day of the 90-day horizon after every projected payment and essential expense.
2. **Conservative Accounting**:
   - Pending debits are reserved immediately.
   - Pending credits, bonuses, commissions, lottery proceeds, and investment gains are **never counted** until settled.
   - Confirmed salary is recognized strictly on its confirmed settlement date.
3. **Evidence Isolation**: Supporting messages and image documents are treated as untrusted data. Instructions embedded within them never override core challenge rules.
4. **Multilingual Support**: Explicitly detects employment termination (hubungan kerja.*berakhir), rent increases (menaikkan sewa), confirmed invoices, and salary revisions across both English and Bahasa Indonesia.
5. **Globally Optimal Spending Adjustments**: Uses combinatorial search across up to 3 flexible recurring expenses, testing both stopping and reducing to minimums, prioritizing the plan with the lowest financial impact.

---

## 7. Submission Artifacts

Per competition contract:
- **code.zip**: Contains the 16 core pipeline files, requirements, and evaluation/usage_report.md (no secrets, no caches).
- **output.csv**: 250 rows matching all schema constraints.
- **chat_transcript**: log.txt recording all agent actions and turns.
