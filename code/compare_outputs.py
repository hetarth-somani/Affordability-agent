"""Compare our output.csv to a reference output.csv field by field.

Usage (run from repo root):
    python code/compare_outputs.py

Reads:
  output.csv                                    ← our predictions
  buy-or-wait-affordability-agent/output.csv    ← reference (friend's)

Prints:
  - Per-field agreement rate
  - Summary of where we agree vs differ on affordability_status
  - Full diff for every row where any field differs (sorted by request_id)

Note: agreement != correctness. Rows where we match the reference are
likely right; rows where we differ need inspection to decide which is
better.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUR_OUTPUT = REPO_ROOT / "output.csv"
REF_OUTPUT = REPO_ROOT / "buy-or-wait-affordability-agent" / "output.csv"

SCORED_FIELDS = [
    "amount_safe_to_pay",
    "affordability_status",
    "recommended_payment_method",
    "payment_plan",
    "earliest_date_for_full_payment",
    "spending_changes_needed",
]

AMOUNT_TOLERANCE = 1.0

STATUS_VALUES = [
    "affordable_now",
    "affordable_with_plan",
    "affordable_later",
    "not_affordable",
]


def _normalize(field: str, value) -> str:
    text = "" if pd.isna(value) else str(value).strip()
    if field == "earliest_date_for_full_payment" and text.lower() in ("nan", "none"):
        return ""
    return text


def _fields_agree(field: str, ours: str, theirs: str) -> bool:
    if field == "amount_safe_to_pay":
        try:
            return abs(float(ours) - float(theirs)) <= AMOUNT_TOLERANCE
        except (ValueError, TypeError):
            return ours == theirs
    return ours == theirs


def main() -> None:
    if not OUR_OUTPUT.exists():
        print(f"[!] {OUR_OUTPUT} not found. Run  python code/main.py  first.")
        sys.exit(1)
    if not REF_OUTPUT.exists():
        print(f"[!] Reference output not found at {REF_OUTPUT}")
        sys.exit(1)

    ours_df = pd.read_csv(OUR_OUTPUT, dtype=str).set_index("request_id")
    ref_df = pd.read_csv(REF_OUTPUT, dtype=str).set_index("request_id")

    common_ids = sorted(set(ours_df.index) & set(ref_df.index))
    only_ours = set(ours_df.index) - set(ref_df.index)
    only_ref = set(ref_df.index) - set(ours_df.index)

    total = len(common_ids)
    print(f"Rows in our output:        {len(ours_df)}")
    print(f"Rows in reference output:  {len(ref_df)}")
    print(f"Rows in common:            {total}")
    if only_ours:
        print(f"Only in ours:              {sorted(only_ours)}")
    if only_ref:
        print(f"Only in reference:         {sorted(only_ref)}")

    per_field_agree: Counter = Counter()
    status_our_dist: Counter = Counter()
    status_ref_dist: Counter = Counter()
    # confusion[our_status][ref_status] = count
    confusion: dict[str, Counter] = defaultdict(Counter)
    differing_rows: list[dict] = []
    fully_agree_count = 0

    for rid in common_ids:
        our_row = ours_df.loc[rid]
        ref_row = ref_df.loc[rid]

        field_agrees = {}
        for f in SCORED_FIELDS:
            our_val = _normalize(f, our_row.get(f, ""))
            ref_val = _normalize(f, ref_row.get(f, ""))
            agree = _fields_agree(f, our_val, ref_val)
            field_agrees[f] = agree
            if agree:
                per_field_agree[f] += 1

        our_status = _normalize("affordability_status", our_row.get("affordability_status", ""))
        ref_status = _normalize("affordability_status", ref_row.get("affordability_status", ""))
        status_our_dist[our_status] += 1
        status_ref_dist[ref_status] += 1
        confusion[our_status][ref_status] += 1

        if all(field_agrees.values()):
            fully_agree_count += 1
        else:
            differing_rows.append(
                {
                    "request_id": rid,
                    "field_agrees": field_agrees,
                    "our": {f: _normalize(f, our_row.get(f, "")) for f in SCORED_FIELDS},
                    "ref": {f: _normalize(f, ref_row.get(f, "")) for f in SCORED_FIELDS},
                }
            )

    # ── Report ─────────────────────────────────────────────────────────
    bar = "=" * 62
    print(f"\n{bar}")
    print(f"  AGREEMENT REPORT  (ours vs reference, {total} rows)")
    print(bar)

    print("\nPer-field agreement rate:")
    max_len = max(len(f) for f in SCORED_FIELDS)
    for f in SCORED_FIELDS:
        agree = per_field_agree[f]
        pct = 100 * agree / total
        blocks = int(pct / 5)
        bar_str = "#" * blocks + "-" * (20 - blocks)
        print(f"  {f:<{max_len}}  [{bar_str}]  {agree:3d}/{total}  ({pct:5.1f}%)")

    print(f"\nFully agree on all fields: {fully_agree_count}/{total}  "
          f"({100*fully_agree_count/total:.1f}%)")

    print("\naffordability_status distribution:")
    print(f"  {'Status':<28}  {'Ours':>6}  {'Ref':>6}")
    for s in STATUS_VALUES:
        print(f"  {s:<28}  {status_our_dist[s]:>6}  {status_ref_dist[s]:>6}")

    print("\naffordability_status cross-tab (row=ours, col=reference):")
    col_w = 20
    header_line = " " * 28 + "".join(f"{s:<{col_w}}" for s in STATUS_VALUES)
    print(f"  {header_line}")
    for our_s in STATUS_VALUES:
        row_str = f"  {our_s:<28}"
        for ref_s in STATUS_VALUES:
            count = confusion[our_s][ref_s]
            cell = str(count) if count else "·"
            row_str += f"{cell:<{col_w}}"
        print(row_str)

    if not differing_rows:
        print("\nPerfect agreement on all rows!")
        return

    print(f"\n{'-'*62}")
    print(f"DIFFERING ROWS  ({len(differing_rows)} out of {total})")
    print(f"{'-'*62}")
    for diff in differing_rows:
        rid = diff["request_id"]
        print(f"\n  {rid}")
        for f in SCORED_FIELDS:
            if not diff["field_agrees"][f]:
                print(f"    [x] {f}")
                print(f"        ours: {diff['our'][f]!r}")
                print(f"        ref:  {diff['ref'][f]!r}")


if __name__ == "__main__":
    main()
