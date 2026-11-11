
```markdown
# DATA_SOURCES.md - piotroski-liquidity-research

Status as of 2026-11-08. Research and education only; not investment advice.
This repo ships code, written reports and three derived CSVs. It does not ship raw prices or filings.
All figures in `reports/` are provisional (see README status banner).

## 1. Inputs (not bundled; located in sibling repos)

### 1.1 US fundamentals - SEC EDGAR XBRL
- Publisher: US Securities and Exchange Commission. APIs: https://data.sec.gov/api/xbrl/companyfacts/ and submissions.
- Terms: US government public information; fair access (declared User-Agent, <= 10 req/s).
- Held at: `global-stock-screener/cache_seed/fundamentals_history/US.parquet` (111,949 rows, 4,597 tickers, fy_end 1987..2029; 12 forward-dated rows to be dropped).
- Fields used: equity, cfo, ebit, total_assets, current_assets, current_liabilities, revenue, cost_of_revenue, net_income, shares, long-term debt, `filed` (real filing date).
- Point-in-time: entry is gated on `filed`. Caveat fixed in 2026-07: the revenue tag previously captured the ASC 606 contract-revenue subset (ADM FY2023 $25.7B vs $93.9B in the 10-K).
- Refresh: collector `sec_history_collector.py` (in global-stock-screener), on demand.

### 1.2 US prices - 10-year daily panel
- Declared location: `global-stock-screener/cache_seed/ltm/US.parquet` (file is absent as of 2026-11-08; US prices now live in `compact/US_2025plus.parquet`, 2025-01-02..2026-07-22). Original publisher: Yahoo Finance (restrictive terms: https://legal.yahoo.com/us/en/yahoo/terms/product-atos/apiforydn/index.html).
- Fields: Date, Open, High, Low, Close (raw), Volume.
- Caveats: raw close, unadjusted splits; delisted names retained; an earlier interrupted collection missed tickers D to L (fixed).

### 1.3 India fundamentals - screener.in (login)
- Publisher: Screener. Terms: https://www.screener.in/guides/terms/ (personal, non-commercial viewing). Private use only; company-level values are not to be redistributed.
- Coverage: about 10 years (2017-2026) annual; no filing dates, so a 120-day lag proxy is used.
- Caveats: quarterly vs annual splice risk; IP throttling; EBIT = PBT + interest is wrong for lenders (documented, unfixed); units are Rs crore in screener rows but absolute INR in yfinance rows of the shared parquet.

### 1.4 India prices - NSE/BSE bhavcopy
- Publisher: NSE (https://www.nseindia.com) / BSE (https://www.bseindia.com). NSE permits research-oriented download (up to 2 GB aggregate) and does not grant redistribution.
- Held at: `cache_seed/ltm/IN.parquet` (4,423,382 rows, 3,476 symbols, 2016-01-01..2026-07-13 per script header; 964 delisted names retained). Raw close; unit splits (e.g. GOLDBEES 1:100) require filtering of 1-day moves outside [-50%, +100%].

### 1.5 yfinance (check data only)
- 5 years, no EBIT; used only to cross-check other sources. Same Yahoo terms as above.

## 2. Published outputs (this repo)

| File | Rows | Content | Inputs | Terms |
|---|---|---|---|---|
| `reports/daily_breakout_combo_us.csv` | 12,773 | Darvas x Piotroski-7 breakout signals 2016-06-02..2025-10-03, forward return, turnover, F9, tier | 1.1, 1.2 | derived statistics; OK with attribution |
| `reports/sweep_piotroski_plus_us.csv` | 3,294 | annual June rebalance, forward return, canonical/quality/turnaround/safety scores, 597 symbols | 1.1, 1.2 | derived |
| `reports/roace_by_liquidity_india.csv` | 291 | per-company ebit, capital employed, cash, ROACE, F-score, turnover, tier | 1.3, 1.4 | screener.in-derived company figures: private only; publish tier aggregates instead |
| `reports/PIOTROSKI_LIQUIDITY_PAPER.md`, `CAVEMAN_SUMMARY.md` | - | method, results, limitations | own | own work |

Known value issues in outputs: ROACE up to 21.5 (7 tickers above 60%), fwd_ret up to +6,696% (sweep) and +4,808% (daily combo; `wret` is winsorised).

## 3. Methods and literature (citations only)
Piotroski (2000), J. Accounting Research 38; Amihud (2002), J. Financial Markets 5(1); Corwin and Schultz (2012), J. Finance 67(2); Fang, Noe and Tice (2009), J. Financial Economics 94(1); Krauss, Krueger and Beerstecher (2015), IWQW Discussion Paper 13/2015 (replace the stale "Walkshausl et al." line in the README).

## 4. Planned / candidate sources (reachable 2026-11-08)
SEC Financial Statement Data Sets (filing dates, bulk); Open Source Asset Pricing and AQR/q-factor data for independent benchmarks; Ken French factors for the missing factor controls; BSE/NSE result-announcement timestamps to replace the 120-day proxy; RBI DBIE for bank-specific profitability; SEBI delisting data for survivorship checks; GLEIF (CC0) for issuer linking.

## 5. Open items
1. Re-run all three scripts against corrected panels and replace the 2026-07-15 CSVs.
2. Add winsorisation or caps and publish the rule.
3. Remove hard-coded absolute paths; record panel versions (hash, row count, date range) in each report header.
4. Move company-level India figures to a private location before any publication.
5. Add LICENSE.
```
