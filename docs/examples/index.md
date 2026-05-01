# Worked examples

Three end-to-end walk-throughs covering the workflows Wanamaker is
designed for. Each runs on the bundled `public_benchmark` dataset, so
no private data is required to follow along.

The Quickstart shows you what the tool does. These examples show you
why each piece exists and when you'd use it.

## [DTC reallocates linear TV to CTV](tv_to_ctv.md)

A retailer wants to shift $5K/week from linear TV to CTV. Walks through
`compare-scenarios` (which plan looks better?) and `recommend-ramp`
(how much of the move can we defend right now?). Best read after
finishing the [Quickstart](../quickstart.md).

**Commands covered:** `wanamaker run`, `compare-scenarios`,
`recommend-ramp`.

## [B2B refresh after a noisy quarter](refresh_after_noisy_quarter.md)

A B2B SaaS company runs MMM quarterly. After a noisy Q4, the historical
paid-search ROI estimate moved 15% — is this a real shift or sampling
noise? Walks through the refresh diff and its movement classification
("expected", "user-induced", "unexplained", etc.) and how to triage
each class.

**Commands covered:** `wanamaker fit`, `refresh`, `report`.

## [A lift test changes the verdict](lift_test_changes_verdict.md)

A retailer ran a 6-week geo-holdout on paid social and found the lift
was 12% lower than the MMM was estimating. Walks through feeding the
experiment back as a calibration prior and seeing how the
recommendations change. Includes a side-by-side ramp comparison between
the uncalibrated and calibrated runs.

**Commands covered:** `wanamaker fit` (twice), `showcase`,
`recommend-ramp`.

## Suggested order

If you're new to MMM workflows: **TV → CTV → Refresh → Lift test**.
The TV → CTV example is the simplest end-to-end loop; the refresh and
lift-test examples each add one piece of operational machinery on top.

If you already run MMM elsewhere: jump straight to the
[refresh example](refresh_after_noisy_quarter.md) — that's the workflow
that distinguishes Wanamaker from Robyn / PyMC-Marketing.
