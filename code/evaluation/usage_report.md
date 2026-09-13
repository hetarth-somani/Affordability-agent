# Token Usage and Cost Analysis

Figures below cover the final full-dataset run that produced `output.csv`. All model calls go through Groq.

## Per-model totals

| Provider | Model | Calls | Input tokens | Output tokens | Total tokens | Est. cost (USD) |
|---|---|---|---|---|---|---|
| Groq | openai/gpt-oss-120b | 101 | 43,680 | 45,888 | 89,568 | $0.0341 |

## Overall totals

- Requests processed: **250**
- Model calls: **101**
- Input tokens: **43,680**
- Output tokens: **45,888**
- Total tokens: **89,568**
- Average tokens per request: **358.3**
- Estimated total cost: **$0.0341**
- Estimated cost per request: **$0.000136**

## Notes on efficiency

- Image understanding runs once per image and is cached on disk, so re-runs and graders do not re-pay for it.
- Message fact extraction runs once per unique message across the whole dataset rather than once per request.
- Explanations are cached by request id plus the exact fact set, so an unchanged decision never regenerates its explanation.
- All planning, forecasting, and validation are deterministic Python with no model calls.
