# Token Usage and Cost Analysis

Figures below cover the final full-dataset run that produced `output.csv`. All model calls go through Groq.

## Per-model totals

| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) |
|---|---|---|---|---|---|---|
| Groq | openai/gpt-oss-120b | 90 | 39,259 | 43,094 | 82,353 | $0.0317 |

## Overall totals

- Requests processed: **250**
- Model calls: **90**
- Input tokens: **39,259**
- Output tokens: **43,094**
- Total tokens: **82,353**
- Average tokens per request: **329.4**
- Estimated total cost: **$0.0317**
- Estimated cost per request: **$0.000127**

## Notes on efficiency

- Image understanding runs once per image and is cached on disk, so re-runs and graders do not re-pay for it.
- Message fact extraction runs once per unique message across the whole dataset rather than once per request.
- Explanations are cached by request id plus the exact fact set, so an unchanged decision never regenerates its explanation.
- All planning, forecasting, and validation are deterministic Python with no model calls.
