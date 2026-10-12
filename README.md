# Piotroski × Liquidity — research code

<!-- how-to-use-with-ai -->
> **New here?** See [How to use this repo with AI (Claude, Gemini, ChatGPT)](HOW_TO_USE_WITH_AI.md) — clone it, paste it into an AI tool, and ask in plain English.


[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/herrrickshaw/piotroski-liquidity-research/blob/main/notebooks/colab_test.ipynb)

Point-in-time backtests asking one question: **is the Piotroski F-score premium
conditioned by firm size, or by stock liquidity?**

> **Educational and research use only. Not investment advice.**
> Findings are historical averages on short samples. Past performance does not
> guarantee future returns. Consult a SEBI-registered (or your jurisdiction's) adviser.

---

## 🔴 Read this before the results

**Numbers in `reports/` are being regenerated. Treat every figure as provisional.**

Five agents were sent to *refute* (not confirm) each derived field. They found **eight
defects**. Three fixed, one documented-and-unfixed, a re-collection running.

| | |
|---|---|
| **Fixed — invalidates the tables** | US `revenue` stored the ASC-606 *contract-revenue subset*, not total. ADM FY2023 held **$25.69B vs a 10-K ~$93.9B — 3.7× wrong**. `cost_of_revenue > revenue` (impossible) fired on 262/5,418 rows. Revenue feeds Piotroski tests 8 and 9, so it **flips booleans** — and it skews to multi-segment filers, i.e. large caps, i.e. the cell carrying the headline. |
| **NOT fixed — documented** | `EBIT = PBT + interest` is **wrong for lenders**; for a bank/NBFC interest is cost of goods sold. EBIT inflates ~4×, capital employed ~9×, and **the errors cancel into the 5–25% plausible band** so no range check fires. It survived validation because the checks used (RELIANCE 8.9%, TCS 56.8%) are both non-financials. **ROCE for Indian lenders is wrong and there is no guard.** |
| **Fixed** | Per-market floors were **fabricated** — US $475k reproduced under no universe definition ($300k with junk, $896k without). The "India 47th percentile" anchor existed only because 1,079 of 3,476 panel names are delisted. Deleted, not recomputed. |
| **Fixed** | `(x or 0)` silently keeps NaN — `float(str(nan))` succeeds so the `None` fallback never fires, and **NaN is truthy**. Voided 15% of gate-passing rows. |
| **Fixed earlier** | The US price panel was an **interrupted alphabetical collection** — CME, Cummins and all of D–L absent. The 597-ticker sample held only A,B,C,M,N,P,Q,R,S,T. |

**Coverage rose 13% → 43%** of available tickers once the panel and an inconsistent field
requirement were fixed. Sample sizes below are understated.

Run `data_completeness_audit.py` before trusting any sample size here.

## The finding

**The premium tracks illiquidity, not size.** Double-sorting liquidity within size
(market cap point-in-time: last close × shares as filed):

| Size | Liquidity | n | F≥70 | F<40 | **Premium** | @818 | @1,673 |
|---|---|---|---|---|---|---|---|
| Small | **Illiquid** | 2,703 | +3.4% | −10.0% | **+13.4pp** | +13.8 | +12.2 |
| Small | Liquid | 2,703 | +4.6% | −2.9% | +7.5pp | +7.7 | **+13.9** |
| **Large** | **Illiquid** | 2,703 | +6.8% | **−24.6%** | **+31.4pp** | +33.7 | +27.8 |
| Large | Liquid | 2,702 | +8.2% | +0.9% | **+7.4pp** | **−1.7** | +3.6 |

Illiquid beats liquid **within both size buckets**; the largest premium is in **large,
illiquid** firms — 4× the next cell.

**The last two columns matter.** The same cells at three sample sizes, as data defects
were fixed:
- **Small+Liquid** reversed at n=1,673 (+13.9, appearing to contradict the thesis — which
  was retracted on it) then reverted to +7.5. The middle run was the outlier.
- **Large+Liquid** climbed −1.7 → +3.6 → **+7.4**. An earlier draft called it *"the only
  negative cell"*. **That claim is withdrawn — it is positive**, merely smallest.
- **Large+Illiquid and Small+Illiquid held** across all three.

The thesis rests on the two cells that survived a 3.3× sample increase. The two that moved
are shown moving.

**It is downside avoidance, not upside capture** — the +31.4pp comes from weak-F names
crashing (**−24.6%**), not strong-F names winning (+6.8%). That mechanism is the most
stable thing here: it held at all three sample sizes.

**Capacity ≈ $250,000**, set by execution rather than fees: above it a position exceeds
~15–20% of the median illiquid name's $0.16M daily volume.

| Capital | Cost/yr | % of ADV | Net premium | Executable |
|---|---|---|---|---|
| $10,000 | 1.18% | 0.6% | +12.1% | yes |
| $100,000 | 1.99% | 6.1% | +11.3% | yes |
| $250,000 | 3.34% | 15.2% | +10.0% | at the limit |
| $1,000,000 | 10.1% | **61%** | +3.2% | **no** |
| $10,000,000 | 91.2% | **608%** | −77.9% | **no** |

Separately, ROCE rises monotonically with liquidity (11.0% → 21.6% ex-cash), confirming
Fang–Noe–Tice while the table above confirms Amihud. **Liquid firms are better
businesses; illiquid stocks are better investments.** Both hold.

## Papers

- **[reports/PIOTROSKI_LIQUIDITY_PAPER.md](reports/PIOTROSKI_LIQUIDITY_PAPER.md)** — full note with method, results, and limitations
- **[reports/CAVEMAN_SUMMARY.md](reports/CAVEMAN_SUMMARY.md)** — same findings in plain words, with the assumptions and data sources stated explicitly

## Code

| file | does |
|---|---|
| `piotroski_plus.py` | canonical 9-point F-score + separate 3-point ROCE block; tunable weight vectors |
| `size_vs_liquidity_us.py` | **the headline** — double-sorts liquidity within size |
| `cost_vs_edge.py` | Corwin–Schultz spread + Amihud impact → capacity ceiling |
| `daily_breakout_combo_us.py` | darvas × piotroski7, scanned daily (12,773 signals) |
| `sweep_piotroski_plus_us.py` | weight sweep, US, EDGAR real filing dates |
| `sweep_piotroski_plus.py` | weight sweep, India, screener.in |
| `roace_by_liquidity.py` | ROCE by tier — level, cash-adjusted, 5y stability |
| `backtest_liquidity_forward.py` | liquidity → forward returns, 10.5y point-in-time |
| `verify_f_band_us.py` | reconciles against a prior "US Piotroski is inverted" claim |
| `data_completeness_audit.py` | **run first** — how much of the data an analysis actually uses |
| `validate_current_ratio_proxy.py` | validates (and rejects) a proxy metric |
| `golden_cross.py` | 50/200-DMA, one implementation for all markets |

## Method notes

- **No lookahead.** Entry gated on EDGAR's real `filed` date (1,449 distinct
  fiscal-end→filing lags, incl. a 2009 year filed in 2012). India lacks filing dates, so
  it uses a conservative 120-day proxy.
- **Survivorship.** Delisted names retained; delisting exits at the last traded close.
  Dropping them is how backtests invent alpha.
- **Corporate actions.** The panels store raw close. Unfiltered, splits manufactured a
  fake +12.20% mean with 248% sd (`GOLDBEES` 3359.60 → 33.55 is a 1:100 unit split).
  Filtering 1-day moves outside [−50%, +100%] collapses that sd to 29%.
- **Median is the headline.** Mean and median disagreed on *every* result; the median was
  correct each time. `LARGE F<40` posts a 117% mean against a 15.1% median.

## Metrics that failed validation

**Eleven** derived metrics looked populated and plausible and were wrong. **Every one was
caught by an independent check; none by reasoning.** Documented in
[the paper §5](reports/PIOTROSKI_LIQUIDITY_PAPER.md):

| metric | failure | caught by |
|---|---|---|
| ROCE via ratio inversion | 95–137% (real 5–25%) | plausibility range |
| Quarterly operating profit | would splice quarterly into an annual series | reading source structure |
| Illiquidity premium | +12.20% mean, 248% sd — unadjusted splits | inspecting the actual prices |
| Current-ratio proxy | 62.1% vs a 57.6% coin-flip baseline | independent ground truth |
| Gross margin | 98% for a services firm (true ~42%) | TCS: raw materials 0.1% of sales |
| Darvas gradient | reversed under proper power | 117 → 12,773 signals |
| US price panel | missing CME, Cummins, all of D–L | counting first letters |
| US revenue tag | ADM $25.7B vs a $93.9B 10-K | `cost > revenue` is impossible |
| Lender EBIT | wrong for banks — **cancels into the plausible band** | ROE cross-check on a financial |
| Per-market floors | fabricated; reproduce under no definition | recomputing them |
| `(x or 0)` | NaN is truthy, so `nan or 0` is `nan` | executing it |

**Three were corrections to this analysis's own conclusions. Three more were fixes
proposed here that the data refuted *before they shipped*:**

- an `interest/revenue ≥ 25%` lender gate — caught **Adani Power** (a utility), missed
  **360ONE** and **5PAISA** (both NBFCs). Interest/revenue measures leverage, not lending.
- *"bonds pass the value gate on face value"* — **0 of 446 clear ₹1 crore**; median bond
  turnover is ₹0.9 lakh/day. Reasoned from a precedent instead of measuring.
- the per-market floors, asserted as "the 47th percentile" without recomputation.

Each was a plausible mechanism asserted without a check. **The pattern is the finding:
a mechanism that sounds right is not evidence.**

## Limitations

1. **Power.** 8–10 rebalances; stocks co-move within one, so effective n approaches the
   rebalance count. No standard errors are reported because none would be credible.
2. **Broken panel.** See the warning above — sample sizes are provisional.
3. **No factor controls.** Size is controlled (imperfectly — `corr(turnover, mcap) = +0.797`);
   beta and book-to-market are not.
4. **Range restriction.** The sample sits at the 77th percentile of US liquidity;
   genuinely illiquid names are untested.
5. **Regime, not skill.** −28.1% in 2020, +14.4%/+7.1% in 2021–22, negative through 2024–25.

## References

- Amihud (2002), *Journal of Financial Markets* 5(1) — illiquidity and stock returns
- Corwin & Schultz (2012), *Journal of Finance* 67(2) — high/low spread estimator
- Fang, Noe & Tice (2009), *Journal of Financial Economics* 94(1) — liquidity and firm value
- Piotroski (2000), *Journal of Accounting Research* 38 — value investing / F-score
- Walkshäusl et al., EconStor — F-score revisited from an investor's perspective

## Data

Not included (large, and not ours to redistribute). Scripts expect:
- `global-stock-screener/cache_seed/fundamentals_history/US.parquet` — SEC EDGAR
- `global-stock-screener/cache_seed/ltm/US.parquet` — prices (**the complete panel**)
- `global-stock-screener/cache_seed/fundamentals_history/IN.parquet` — screener.in
