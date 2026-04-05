# src/prediction_engine.py

import pandas as pd
import numpy as np
import pickle
import os

# ── Load models and data ──────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR  = os.path.join(BASE_DIR, 'models')
DATA_DIR    = os.path.join(BASE_DIR, 'data', 'processed')

# Load calibrated model
with open(os.path.join(MODELS_DIR, 'calibrated_pctr_model.pkl'), 'rb') as f:
    calibrated_model = pickle.load(f)

# Load feature cols
with open(os.path.join(MODELS_DIR, 'feature_cols.pkl'), 'rb') as f:
    feature_cols = pickle.load(f)

# Load data tables
df_keywords = pd.read_csv(os.path.join(DATA_DIR, 'keyword_master.csv'))
df_keywords = df_keywords.drop(columns=['Unnamed: 0'], errors='ignore')

df_products = pd.read_csv(os.path.join(DATA_DIR, 'product_catalog.csv'))
df_products = df_products.drop(columns=['Unnamed: 0'], errors='ignore')

df_config   = pd.read_csv(os.path.join(DATA_DIR, 'campaign_keyword_config.csv'))
df_hourly   = pd.read_csv(os.path.join(DATA_DIR, 'campaign_hourly_spend.csv'))




# Compute avg order value per category from impression log
# Add this near the top of prediction_engine.py after loading data

df_impressions = pd.read_csv(os.path.join(DATA_DIR, 'keyword_impression_log.csv'))
df_impressions = df_impressions.drop(columns=['Unnamed: 0'], errors='ignore')

avg_order_value_by_category = (
    df_impressions[df_impressions['order_value'].notna()]
    .merge(df_keywords[['keyword_id', 'category']], on='keyword_id', how='left')
    .groupby('category')['order_value']
    .mean()
    .round(2)
    .to_dict()
)

# Default fallback
DEFAULT_AVG_ORDER_VALUE = df_impressions['order_value'].mean().round(2)

# ── Functions ─────────────────────────────────────────────────────────────────

def get_impression_share(cpm_bid, avg_competitor_bid):
    x = (cpm_bid-avg_competitor_bid)/avg_competitor_bid * 3
    share = 1/(1+np.exp(-x))
    clip = np.clip(share, 0.02, 0.97).round(3)
    return clip

def predict_pctr(keyword_id, slot_position=2, hour_of_day=9,
                 day_of_week=0, cpm_bid=100):
    kw       = df_keywords[df_keywords['keyword_id'] == keyword_id].iloc[0]
    cpm_norm = (cpm_bid - 20) / (200 - 20)

    intent_map   = {'branded': 0, 'generic': 1, 'long_tail': 2}
    category_map = {
        'Beverages': 0, 'Confectionery': 1, 'Dairy': 2,
        'Personal Care': 3, 'Snacks': 4
    }

    features = {
        'hour_of_day'             : hour_of_day,
        'day_of_week'             : day_of_week,
        'slot_position'           : slot_position,
        'slot_visibility'         : round(1.0 - (slot_position/2 - 1) * 0.06, 2),
        'position'                : slot_position,
        'intent_level_encoded'    : intent_map.get(kw['intent_level'], 1),
        'category_encoded'        : category_map.get(kw['category'], 0),
        'competition_score'       : float(kw['competition_score']),
        'avg_competitor_bid'      : float(kw['avg_competitor_bid']),
        'historical_ctr'          : float(kw['historical_ctr']),
        'avg_daily_searches'      : int(kw['avg_daily_searches']),
        'keyword_score'           : float(kw['keyword_score']),
        'a2c_rate'                : 0.25,
        'purchase_affinity'       : 0.40,
        'relevance_score'         : 0.75,
        'cpm_bid'                 : cpm_bid,
        'auction_score'           : round(cpm_norm * 0.75 * 0.25 * 0.40, 4),
        'p_win_slot'              : 0.50,
        'bid_to_competition_ratio': round(cpm_bid / (float(kw['avg_competitor_bid']) + 1e-8), 3)
    }

    X_pred = pd.DataFrame([features])
    pctr   = calibrated_model.predict_proba(X_pred)[:, 1][0]

    return round(float(pctr), 4)



def get_bid_recommendations(keyword_id,
                             avg_order_value=200,
                             target_roas=4.0):
    kw = df_keywords[
        df_keywords['keyword_id'] == keyword_id
    ].iloc[0]

    avg_comp_bid = float(kw['avg_competitor_bid'])
    cvr          = 0.12

    # Floor bid — minimum to enter auction
    floor_bid = avg_comp_bid * 0.30  # avg_competitor_bid × 0.30

    # Suggested bid — gives ~50% impression share
    # invert sigmoid: find bid where share = 0.55
    # bid = avg_comp × (1 + log(0.55/0.45) / 3)
    suggested_bid = avg_comp_bid * (1+np.log(0.55/0.45) /3)

    # Max bid — based on economics
    # max_bid = avg_order_value × CVR / target_roas
    max_bid = avg_order_value * cvr/target_roas

    # Weighted bid — blends competition + economics
    weighted_bid =  0.6 * suggested_bid + 0.4 * min(max_bid, max_bid)

    return {
        'keyword_id'   : keyword_id,
        'keyword_text' : kw['keyword_text'],
        'floor_bid'    : round(floor_bid, 2),
        'suggested_bid': round(suggested_bid, 2),
        'max_bid'      : round(max_bid, 2),
        'weighted_bid' : round(weighted_bid, 2),
        'top_of_page'  : round(float(kw['top_of_page_bid']), 2)
    }
def get_keyword_recommendations(product_id, top_n=20):
    product = df_products[
        df_products['product_id'] == product_id
    ].iloc[0]

    category     = product['our_category']
    product_name = product['product_name']

    # Step 2 → filter keywords by same category
    kw_filtered = df_keywords[
        df_keywords['category'] == category
    ].copy()

    # Step 3 → compute product match score per keyword
    def match_score(keyword_text):
        product_tokens = set(product_name.lower().split())
        keyword_tokens = set(keyword_text.lower().split())

        # Token overlap
        overlap = product_tokens & keyword_tokens
        union   = product_tokens | keyword_tokens
        base    = len(overlap) / len(union) if union else 0

        # Brand/product name bonus
        name_bonus = 0.30 if any(
            t in keyword_text.lower()
            for t in product_tokens
            if len(t) > 3  # ignore short words like "1l", "of"
        ) else 0

        # Category relevance bonus
        # juice → fruit juice, cold drink gets bonus
        category_terms = {
            'Beverages'    : ['juice','drink','water','coffee',
                            'beverage','cola','energy'],
            'Snacks'       : ['chips','snack','biscuit','noodle',
                            'namkeen','cracker','wafer'],
            'Dairy'        : ['milk','butter','cheese','curd',
                            'paneer','dairy','ghee'],
            'Confectionery': ['chocolate','candy','sweet','toffee',
                            'wafer','cocoa'],
            'Personal Care': ['soap','shampoo','detergent','wash',
                            'cream','lotion','toothpaste']
        }

        cat_bonus = 0.15 if any(
            term in keyword_text.lower()
            for term in category_terms.get(category, [])
        ) else 0

        return min(1.0, base + name_bonus + cat_bonus)

    kw_filtered['product_match_score'] = kw_filtered[
        'keyword_text'
    ].apply(match_score)

    # Step 4 → compute weighted ranking score
    # normalize each component first
    def normalize(s):
        return (s - s.min()) / (s.max() - s.min() + 1e-8)

    # Penalize zero match score keywords
    kw_filtered['match_penalty'] = (
        kw_filtered['product_match_score'] == 0).astype(int) * 0.15

    kw_filtered['final_score'] = (
        0.25 * normalize(kw_filtered['avg_daily_searches'])
    + 0.25 * normalize(kw_filtered['historical_ctr'])
    + 0.20 * normalize(kw_filtered['impressions_won'])
    + 0.30 * kw_filtered['product_match_score']
    - kw_filtered['match_penalty']).clip(0, 1)



    # Step 5 → return top N
    kw_filtered['final_score'] = (
    0.25 * normalize(kw_filtered['avg_daily_searches'])
    + 0.25 * normalize(kw_filtered['historical_ctr'])
    + 0.20 * normalize(kw_filtered['impressions_won'])
    + 0.30 * kw_filtered['product_match_score'])  # increased from 0.20)

        # Only recommend keywords with some relevance
    kw_filtered = kw_filtered[
        kw_filtered['product_match_score'] > 0]

    top_keywords = (
        kw_filtered
        .sort_values('final_score', ascending=False)
        .head(top_n)
        [[
            'keyword_id', 'keyword_text', 'intent_level',
            'avg_daily_searches', 'historical_ctr',
            'competition_score', 'avg_competitor_bid',
            'product_match_score', 'final_score'
        ]]
        .reset_index(drop=True))

    top_keywords.index += 1  # rank starts at 1

    return top_keywords, product_name, category

def predict_budget_exhaustion(keyword_id, daily_budget,
                               current_hour=12):
    kw_hourly = df_hourly[
        df_hourly['keyword_id'] == keyword_id
    ].copy()

    # Step 2 → avg spend per hour for this keyword
    avg_hourly_spend = kw_hourly['spend_simulated'].mean()

    # Step 3 → hours remaining in day
    hours_remaining = 24 - current_hour

    # Step 4 → projected total spend for rest of day
    projected_spend = avg_hourly_spend * hours_remaining

    # Step 5 → hours until exhaustion
    if avg_hourly_spend > 0:
        hours_to_exhaustion = daily_budget / avg_hourly_spend
    else:
        hours_to_exhaustion = 24

    # Step 6 → predicted exhaustion hour
    exhaustion_hour = min(current_hour + hours_to_exhaustion, 24)

    # Step 7 → budget status
    if exhaustion_hour >= 23:
        status = 'Budget lasts full day'
    elif exhaustion_hour >= 18:
        status = 'Budget exhausts in evening'
    elif exhaustion_hour >= 12:
        status = 'Budget exhausts in afternoon'
    else:
        status = 'Budget exhausts in morning'

    return {
        'keyword_id'          : keyword_id,
        'daily_budget'        : daily_budget,
        'avg_hourly_spend'    : round(avg_hourly_spend, 2),
        'hours_to_exhaustion' : round(hours_to_exhaustion, 1),
        'exhaustion_hour'     : round(exhaustion_hour, 1),
        'exhaustion_time'     :  f"{int(min(exhaustion_hour, 23)):02d}:{int((min(exhaustion_hour, 23.99) % 1) * 60):02d}",
        'status'              : status
    }


# ── Seasonal multipliers ──────────────────────────────────────────────────────
SEASONAL_MULTIPLIERS = {
    'Confectionery': {1:1.0, 2:0.9, 3:1.0, 4:0.8, 5:0.7, 6:0.7,
                      7:0.8, 8:0.9, 9:1.0, 10:1.8, 11:2.1, 12:1.4},
    'Beverages':     {1:0.8, 2:0.8, 3:1.0, 4:1.5, 5:1.9, 6:1.7,
                      7:1.4, 8:1.2, 9:1.0, 10:0.9, 11:0.8, 12:0.8},
    'Snacks':        {1:1.0, 2:1.0, 3:1.3, 4:1.1, 5:1.0, 6:1.0,
                      7:1.0, 8:1.0, 9:1.0, 10:1.4, 11:1.6, 12:1.2},
    'Personal Care': {1:1.0, 2:1.0, 3:1.0, 4:1.0, 5:1.1, 6:1.2,
                      7:1.3, 8:1.1, 9:1.0, 10:1.0, 11:1.0, 12:1.0},
    'Dairy': {            1:1.2, 2:1.1, 3:1.1, 4:1.0, 
                        5:0.9, 6:0.9, 7:0.9, 8:1.0, 
                        9:1.0, 10:1.1, 11:1.2, 12:1.3}  # ← Oct/Nov/Dec higher for Dairy},
}

def get_seasonal_multiplier(category, target_date=None):
    """Get seasonal multiplier for a category on a given date."""
    if target_date is None:
        month = pd.Timestamp.today().month
    else:
        month = pd.to_datetime(target_date).month

    category_multipliers = SEASONAL_MULTIPLIERS.get(category, {})
    return category_multipliers.get(month, 1.0)


def get_impression_range(keyword_id, cpm_bid, target_date=None):
    """
    Returns p10/p50/p90 impression estimates for a keyword.
    Accounts for seasonal effects based on target date.
    """
    # Get keyword metadata
    kw = df_keywords[df_keywords['keyword_id'] == keyword_id]
    if kw.empty:
        return None
    kw = kw.iloc[0]

    # Base impression share
    base_share = get_impression_share(cpm_bid, kw['avg_competitor_bid'])

    # Seasonal multiplier
    category        = kw.get('category', 'Snacks')
    seasonal_mult   = get_seasonal_multiplier(category, target_date)

    # Base daily impressions (mid estimate)
    base_impressions = kw['avg_daily_searches'] * base_share * seasonal_mult

    # Impression range using historical variance
    # Pull actual daily impressions from hourly spend table
    kw_hourly = df_hourly[df_hourly['keyword_id'] == keyword_id]

    if len(kw_hourly) > 0:
        daily_impressions = (
            kw_hourly.groupby('date')['impressions'].sum()
        )
        # Scale historical variance to current bid level
        hist_std = daily_impressions.std()
        hist_cv  = hist_std / (daily_impressions.mean() + 1e-8)
    else:
        hist_cv = 0.25  # default 25% coefficient of variation

    # Apply variance to get range
    low  = round(base_impressions * (1 - hist_cv * 1.5))
    mid  = round(base_impressions)
    high = round(base_impressions * (1 + hist_cv * 1.5))

    # Season label for display
    month = pd.to_datetime(target_date).month if target_date else pd.Timestamp.today().month
    if seasonal_mult >= 1.5:
        season_label = 'Peak season'
    elif seasonal_mult >= 1.2:
        season_label = 'Above average season'
    elif seasonal_mult <= 0.8:
        season_label = 'Low season'
    else:
        season_label = 'Normal season'

    return {
        'keyword_id':        keyword_id,
        'keyword_text':      kw['keyword_text'],
        'category':          category,
        'target_date':       str(target_date) if target_date else 'today',
        'seasonal_multiplier': round(seasonal_mult, 2),
        'season_label':      season_label,
        'impression_share':  round(base_share, 3),
        'impressions_low':   max(0, low),
        'impressions_mid':   max(0, mid),
        'impressions_high':  max(0, high),
    }


def forecast_keyword_v2(keyword_id, product_id, cpm_bid,
                         target_date=None, daily_budget=5000,
                         avg_order_value=150):
    """
    Upgraded forecast with impression ranges and budget exhaustion.
    Returns p10/p50/p90 for impressions, clicks, orders.
    """
    # Get impression range
    imp_range = get_impression_range(keyword_id, cpm_bid, target_date)
    if imp_range is None:
        return None

    kw = df_keywords[df_keywords['keyword_id'] == keyword_id].iloc[0]

    # Predict pCTR for mid scenario (slot 2, peak hour 9AM)
    pctr_peak = predict_pctr(
        keyword_id=keyword_id,
        slot_position=2,
        hour_of_day=9,
        day_of_week=1
    )

    # pCTR for off-peak (slot 12, 2PM)
    pctr_offpeak = predict_pctr(
        keyword_id=keyword_id,
        slot_position=12,
        hour_of_day=14,
        day_of_week=1
    )

    # Blended CTR — weighted average across slots
    blended_ctr = (pctr_peak * 0.4 + pctr_offpeak * 0.6)

    # Clicks range
    clicks_low  = round(imp_range['impressions_low']  * blended_ctr)
    clicks_mid  = round(imp_range['impressions_mid']  * blended_ctr)
    clicks_high = round(imp_range['impressions_high'] * blended_ctr)

    # CVR from historical data
    kw_data    = df_hourly[df_hourly['keyword_id'] == keyword_id]
    total_clicks  = kw_data['clicks'].sum()
    total_orders  = kw_data['orders'].sum()
    cvr = total_orders / (total_clicks + 1e-8)
    cvr = max(0.05, min(cvr, 0.30))  # clip to realistic range

    # Orders range
    orders_low  = round(clicks_low  * cvr, 1)
    orders_mid  = round(clicks_mid  * cvr, 1)
    orders_high = round(clicks_high * cvr, 1)

    # Spend and ROAS (mid scenario)
    daily_spend = round(
    imp_range['impressions_mid'] * (cpm_bid / 1000), 2
    )
    revenue_mid = orders_mid * avg_order_value
    roas        = round(revenue_mid / (daily_spend + 1e-8), 2)

    # Budget exhaustion
    exhaustion  = predict_budget_exhaustion(keyword_id, daily_budget)

    return {
        'keyword_id':          keyword_id,
        'keyword_text':        kw['keyword_text'],
        'category':            kw['category'],
        'target_date':         str(target_date) if target_date else 'today',
        'season_label':        imp_range['season_label'],
        'seasonal_multiplier': imp_range['seasonal_multiplier'],
        'impression_share':    imp_range['impression_share'],

        # Impression range
        'impressions_low':     imp_range['impressions_low'],
        'impressions_mid':     imp_range['impressions_mid'],
        'impressions_high':    imp_range['impressions_high'],

        # Click range
        'clicks_low':          clicks_low,
        'clicks_mid':          clicks_mid,
        'clicks_high':         clicks_high,

        # Order range
        'orders_low':          orders_low,
        'orders_mid':          orders_mid,
        'orders_high':         orders_high,

        # Financials
        'daily_spend':         daily_spend,
        'roas':                roas,
        'blended_ctr':         round(blended_ctr * 100, 2),

        # Budget
        'daily_budget':        daily_budget,
        'exhaustion_time':     exhaustion['exhaustion_time'],
        'exhaustion_status':   exhaustion['status'],
    }


def get_keyword_trend(keyword_id, reference_date=None):
    """
    Returns week-over-week trend for a keyword.
    Compares last 7 days vs previous 7 days.
    """
    if reference_date is None:
        ref = pd.Timestamp('2024-03-31')  # use end of our data range
    else:
        ref = pd.to_datetime(reference_date)

    # Filter keyword data
    kw_data = df_hourly[df_hourly['keyword_id'] == keyword_id].copy()
    kw_data['date'] = pd.to_datetime(kw_data['date'])

    # This week — last 7 days
    this_week_mask = (
        (kw_data['date'] > ref - pd.Timedelta(days=7)) &
        (kw_data['date'] <= ref)
    )
    # Last week — 7-14 days ago
    last_week_mask = (
        (kw_data['date'] > ref - pd.Timedelta(days=14)) &
        (kw_data['date'] <= ref - pd.Timedelta(days=7))
    )

    this_week = kw_data[this_week_mask]
    last_week = kw_data[last_week_mask]

    # Compute metrics
    def safe_pct_change(new, old):
        if old == 0 or new == 0:
            return 0.0  # not enough data
        change = (new - old) / old * 100
        return round(max(-99, min(99, change)), 1)  # cap at ±99%

    this_impr   = this_week['impressions'].sum()
    last_impr   = last_week['impressions'].sum()

    this_clicks = this_week['clicks'].sum()
    last_clicks = last_week['clicks'].sum()

    this_ctr    = this_clicks / (this_impr + 1e-8) * 100
    last_ctr    = last_clicks / (last_impr + 1e-8) * 100

    this_spend  = this_week['spend_simulated'].sum() if 'spend_simulated' in this_week.columns else this_week['spend'].sum()
    last_spend  = last_week['spend_simulated'].sum() if 'spend_simulated' in last_week.columns else last_week['spend'].sum()

    impr_trend   = safe_pct_change(this_impr,   last_impr)
    clicks_trend = safe_pct_change(this_clicks, last_clicks)
    ctr_trend    = safe_pct_change(this_ctr,    last_ctr)
    spend_trend  = safe_pct_change(this_spend,  last_spend)

    def trend_label(pct):
        if abs(pct) < 5:
            return "→ Stable"
        elif pct > 0:
            return f"↑ {abs(pct):.0f}%"
        else:
            return f"↓ {abs(pct):.0f}%"

        

    # At the end of get_keyword_trend, before return
# Scale up to realistic daily volumes using keyword master

    kw_meta = df_keywords[df_keywords['keyword_id'] == keyword_id]
    if not kw_meta.empty:
        avg_searches = kw_meta.iloc[0]['avg_daily_searches']
        # Scale factor = avg daily searches / avg daily impressions in data
        avg_daily_impr = max(this_impr, last_impr, 1) / 7
        scale = avg_searches / (avg_daily_impr + 1e-8)
        scale = min(scale, 1000)  # cap scaling

        this_impr   = int(this_impr * scale)
        last_impr   = int(last_impr * scale)
        this_clicks = int(this_clicks * scale)
        last_clicks = int(last_clicks * scale)
    # Only show trend if meaningful data exists
    MIN_IMPRESSIONS = 100  # scaled impressions
    return {
            'keyword_id':        keyword_id,
            'this_week_impr':    int(this_impr),
            'last_week_impr':    int(last_impr),
            'impr_trend_pct':    0.0,
            'impr_trend_label':  '→ Stable',
            'clicks_trend_pct':  0.0,
            'clicks_trend_label':'→ Stable',
            'ctr_trend_pct':     0.0,
            'ctr_trend_label':   '→ Stable',
            'spend_trend_pct':   0.0,
            'spend_trend_label': '→ Stable',
            'is_trending_up':    False,
            'is_trending_down':  False,
        }


   
        
    



