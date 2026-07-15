# Piotroski × Liquidity — research code

Point-in-time backtests asking one question: **is the Piotroski F-score premium
conditioned by firm size, or by stock liquidity?**

> **Educational and research use only. Not investment advice.**
> Findings are historical averages on short samples. Past performance does not
> guarantee future returns. Consult a SEBI-registered (or your jurisdiction's) adviser.

---

## 🔴 Read this before the results

**Every US result in `reports/` was computed on a broken price panel and must be re-run.**

The panel used (`global-market-data/cache_seed/ltm/US.parquet`) turned out to be an
**interrupted alphabetical collection**: 5,358 symbols, but CME Group and Cummins absent
and its most-covered names all starting with B. The 597 tickers behind every US number
contain letters **A,B,C,M,N,P,Q,R,S,T and no D–L, O, U–Z**.

The fix is a path change (`global-stock-screener/cache_seed/ltm/US.parquet`, 9,278
symbols — EDGAR overlap rises 50% → 97%), already applied to the scripts. **The scripts
are correct; the committed CSVs are not.** Re-running them is the first job.

The *direction* of the findings is unlikely to reverse — they are structural, not
letter-specific — but **every sample size and cell count is provisional.**

Run `data_completeness_audit.py` before trusting any sample size here.

---

## The finding

**The premium tracks illiquidity, not size.** Double-sorting liquidity within size
(market cap point-in-time: last close × shares as filed):

| Size | Liquidity | n | F≥70 | F<40 | **Premium** |
|---|---|---|---|---|---|
| Small | Illiquid | 819 | +7.2% | −6.6% | **+13.8pp** |
| Small | Liquid | 818 | −1.8% | −9.6% | +7.7pp |
| **Large** | **Illiquid** | 818 | +8.2% | **−25.6%** | **+33.7pp** |
| Large | Liquid | 818 | +3.6% | +5.3% | **−1.7pp** |

Illiquid beats liquid in **both** size buckets. The largest premium is in **large but
illiquid** firms. The only negative cell is large-and-liquid.

**It is downside avoidance, not upside capture** — the +33.7pp comes from weak-F names
crashing (−25.6%), not strong-F names winning (+8.2%).

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

Six derived metrics looked populated and plausible and were wrong. **Every one was caught
by an independent check; none by reasoning.** Documented in
[the paper §5](reports/PIOTROSKI_LIQUIDITY_PAPER.md):

| metric | failure | caught by |
|---|---|---|
| ROCE via ratio inversion | 95–137% (real 5–25%) | plausibility range |
| Quarterly operating profit | would splice quarterly into an annual series | reading source structure |
| Illiquidity premium | +12.20% mean, 248% sd — unadjusted splits | inspecting the actual prices |
| Current-ratio proxy | 62.1% vs a 57.6% coin-flip baseline | independent ground truth |
| Gross margin | 98% for a services firm (true ~42%) | TCS: raw materials 0.1% of sales |
| Darvas gradient | reversed under proper power | 117 → 12,773 signals |

Two were corrections to this analysis's own earlier conclusions.

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
