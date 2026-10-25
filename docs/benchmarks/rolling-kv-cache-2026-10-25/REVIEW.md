# Final local review

- Implementation, tests and docs reviewed. Defaults remain rebuild/FP32; no training or native
  arithmetic changes. Sessions are request-scoped and invalidate on weight/backend changes.
- Positional continuation is noGrad-only; overflow is checked. Retained KV storage and rotary
  tables do not grow with generated length. Origin explicitly rejects rolling semantics.
- Quality/complexity/length/duplicate/coverage gates were not relaxed. Shared desktop fixtures
  were extracted to keep the existing class-size gate, not to exclude tests from checks.
- Working Java/frontend/native source equals the frozen qualified candidate. Original source
  and binary SHA-256 manifests both pass after all runs.
- All 90 successful case receipts verified. The original failed harness invocation and the
  original analyzer's four swap review findings remain visible. No failed case was relabeled.
- Swap findings investigated with old/new repeats and per-shape native accounting. The existing
  unaligned prefill fallback is memory-heavy in both revisions. No new native-memory regression,
  retained-buffer growth or default numerical mismatch was found. The exact OS paging trigger
  cannot be attributed from global counters and is not claimed known.
- Eight final long samples manually reviewed; quality is mixed, with repetition in both modes.
  Rolling is an optional speed/semantics trade-off, not a cure for repetition or accuracy.
- Local acceptance is complete. Exact final-SHA CI and final PR review must still pass before
  the conditionally authorized rolling merge. ND migration has not begun on this branch.
