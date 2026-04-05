# Keyword-Based Self-Serve Advertiser Dashboard
### E-Commerce Ads Platform — End-to-End Data Science Project

A self-serve advertiser dashboard for FMCG brands to create keyword-based
sponsored product campaigns on a quick commerce platform. Brands can discover
keywords, forecast campaign performance, and manage bids — all powered by a
real-time pCTR prediction engine.

---

## Problem Statement

Quick commerce platforms (like Blinkit, Zepto, Swiggy Instamart) run
sponsored product ads where FMCG brands pay to appear in high-visibility
slots in the product feed. The challenge:

- Brands don't know which keywords to bid on
- Brands can't predict how many impressions/clicks their budget will get
- Without pacing logic, budgets exhaust in the first few hours of the day
- Auction winners are determined by a weighted score — not just highest bid

This project builds the data science engine and self-serve dashboard that
solves all four problems.

---

## Live Demo


streamlit run app.py



---

## Architecture
```

┌─────────────────────────────────────────────────────────────┐
│                    Self-Serve Dashboard                      │
│                    (Streamlit app.py)                        │
│                                                              │
│  Tab 1: Keyword Planner  │  Tab 2: Forecast  │  Tab 3: Bids │
└──────────────────────────┼───────────────────┼──────────────┘
│
▼
┌─────────────────────────────────────────────────────────────┐
│                   Prediction Engine                          │
│                 src/prediction_engine.py                     │
│                                                              │
│  get_impression_range()    forecast_keyword_v2()             │
│  predict_pctr()            get_bid_recommendations()         │
│  get_keyword_recommendations()  predict_budget_exhaustion()  │
└──────────────────────────┼───────────────────┼──────────────┘
│                   │
┌────────────┘                   └────────────┐
▼                                             ▼
┌─────────────────────────┐              ┌─────────────────────────┐
│   LightGBM pCTR Model   │              │      Data Tables         │
│   models/               │              │      data/processed/     │
│                         │              │                          │
│   calibrated_pctr_      │              │  keyword_master.csv      │
│   model.pkl             │              │  keyword_impression_     │
│   AUC: 0.727            │              │  log.csv                 │
│   Log Loss: 0.085       │              │  campaign_keyword_       │
│   Mean pCTR: 0.0171     │              │  config.csv              │
│                         │              │  campaign_hourly_        │
└─────────────────────────┘              │  spend.csv               │
│  product_catalog.csv     │
└─────────────────────────┘
```
---

## Auction Mechanism

This project implements a **relevance-weighted CPM auction** — the same
mechanism used by Amazon Sponsored Products and Google Shopping Ads.
auction_score = cpm_bid
× relevance_score    (query ↔ product match)
× a2c_rate           (historical add-to-cart rate)
× purchase_affinity  (user affinity to category)
Winner = highest auction_score
Spend  = (impressions_won / 1000) × cpm_bid

**Key insight:** A relevant ad with a lower bid can beat an irrelevant
high-bidding ad. This makes the auction fair for brands and improves
user experience.

---

## Feed Structure
Position 1  → organic
Position 2  → SPONSORED ◆  (slot 1 — highest visibility)
Position 3  → organic
Position 4  → SPONSORED ◆  (slot 2)
...
Position 24 → SPONSORED ◆  (slot 12 — lowest visibility)
12 sponsored slots per page
Visibility decays by 0.06 per slot
Slot 2 visibility = 1.00, Slot 24 visibility = 0.34

---

## pCTR Model

| Decision | Choice | Reason |
|----------|--------|--------|
| Algorithm | LightGBM | Fast, handles imbalance, tabular data |
| Calibration | Platt scaling (sigmoid) | Raw probs ~0.47, needed ~0.02 |
| Split | Time-based 70/15/15 | Prevents future data leakage |
| Class weights | balanced | 97% non-clicks vs 3% clicks |
| Features | 19 across 5 groups | Keyword + slot + temporal + auction |

### Model Performance
| Metric | Value | Benchmark |
|--------|-------|-----------|
| Val AUC | 0.717 | Industry: 0.70–0.76 |
| Test AUC | 0.727 | Industry: 0.70–0.76 |
| Log Loss | 0.085 | Lower is better |
| Mean pCTR | 0.0171 | Actual CTR: 0.0182 ✓ |

### Features Used
Temporal  : hour_of_day, day_of_week
Slot      : slot_position, slot_visibility, position
Keyword   : intent_level_encoded, category_encoded,
competition_score, avg_competitor_bid,
historical_ctr, avg_daily_searches, keyword_score
User      : a2c_rate, purchase_affinity, relevance_score
Auction   : cpm_bid, auction_score, p_win_slot,
bid_to_competition_ratio

---

## Dashboard Features

### Tab 1 — Keyword Planner
- Top 20 keywords recommended per product
- Ranked by weighted score: searches × CTR × impressions × match
- Impression range (P10/P50/P90) per keyword
- Season indicator (🔥 peak / 📈 above average / 📊 normal)
- CPM bid guidance (min / suggested / top of page)
- One-click add to campaign

### Tab 2 — Campaign Forecast
- Aggregate impressions, clicks, orders, spend, ROAS
- All metrics shown as ranges (low/mid/high)
- Budget burn curve (cumulative spend vs daily budget)
- Hourly spend distribution
- Budget utilisation warnings and suggestions
- Per-keyword breakdown table

### Tab 3 — Bid Manager
- Per-keyword CPM bid adjustment
- Bids can only increase (not decrease) once set
- Impression share updates on bid change
- Season indicator per keyword

---

## Data Sources

| Table | Source | Rows |
|-------|--------|------|
| keyword_impression_log | Criteo Display Advertising Dataset (Kaggle) | 100,000 |
| keyword_master | Synthesized — real Indian market CPC/volume data | 54 |
| product_catalog | Open Food Facts (Kaggle) + manual curation | 38 |
| campaign_keyword_config | Synthesized | 430 |
| campaign_hourly_spend | Derived from impression log | 21,846 |

---

## Project Structure
```
ecommerce-ads-dashboard/
│
├── app.py                          ← Streamlit dashboard
│
├── src/
│   └── prediction_engine.py        ← All prediction functions
│
├── models/
│   ├── calibrated_pctr_model.pkl   ← Calibrated LightGBM
│   ├── pctr_model.pkl              ← Raw LightGBM
│   └── feature_cols.pkl            ← Feature column names
│
├── data/
│   ├── raw/                        ← Original downloaded files
│   └── processed/                  ← Cleaned, feature-engineered tables
│
├── notebooks/
│   ├── 01_data_preparation.ipynb   ← Data collection + synthesis
│   ├── 02_feature_engineering.ipynb← Feature joins + engineering
│   ├── 03_model_training.ipynb     ← LightGBM training + calibration
│   └── 04_prediction_engine.ipynb  ← Prediction function testing
│
├── requirements.txt
└── README.md
```
---

## How to Run Locally
```bash
# 1. Clone the repository
git clone https://github.com/yourusername/ecommerce-ads-dashboard
cd ecommerce-ads-dashboard

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate          # Windows
source venv/bin/activate       # Mac/Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the dashboard
streamlit run app.py
```

---

## Key Design Decisions

**Why CPM instead of CPC?**
Quick commerce sponsored products are impression-based — brands
pay for visibility, not just clicks. CPM aligns incentives better
for brand awareness campaigns common in FMCG.

**Why time-based train/val/test split?**
Random splits leak future CTR patterns into training data.
Time-based splitting ensures the model is evaluated on truly
unseen future data — matching real production conditions.

**Why Platt scaling calibration?**
LightGBM with balanced class weights outputs ~0.47 mean probability
for a dataset with 2% actual CTR. Uncalibrated probabilities make
auction scores meaningless and spend forecasts 20× too high.
Platt scaling corrects mean pCTR from 0.47 → 0.0171 ≈ actual 0.0182.

**Why P10/P50/P90 impression range instead of point estimate?**
Single point estimates give false precision. A range communicates
uncertainty honestly — brands can plan for conservative (P10)
and optimistic (P90) scenarios when setting budgets.

**Why impression share via sigmoid function?**
The sigmoid naturally models competitive auction dynamics —
bidding below average gives diminishing returns,
bidding above average gives diminishing marginal gains.
Matches real-world auction behavior observed in Google/Amazon Ads.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | LightGBM + scikit-learn calibration |
| Dashboard | Streamlit |
| Data processing | pandas, numpy |
| Experiment tracking | MLflow |
| Model monitoring | Evidently AI |
| Language | Python 3.13 |

---

## Limitations and Future Work
Current limitations:
→ Training data based on Criteo dataset (display ads, not search)
→ 90-day synthetic data — limited seasonal signal
→ No real-time data pipeline (forecasts based on historical patterns)
→ Trend data sparse due to sample size
Future improvements:
→ Real-time spend tracking with Redis + WebSocket
→ A/B testing framework for auction mechanism
→ Multi-touch attribution model
→ Keyword expansion via embedding similarity
→ Automated bid optimization (target ROAS bidding)

---

## Author

Built as a portfolio project to demonstrate end-to-end
data science skills in the ads/e-commerce domain.

Inspired by real keyword auction systems at
Amazon Sponsored Products, Google Shopping Ads,
and quick commerce platforms in India.

