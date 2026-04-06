import streamlit as st
import pandas as pd
import numpy as np
import math
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.prediction_engine import (
    get_impression_range,
    forecast_keyword_v2,
    get_bid_recommendations,
    get_keyword_recommendations,
    predict_budget_exhaustion,
    df_keywords,
    df_products,
    df_config
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="E-Commerce Ads",
    page_icon="🛒",
    layout="wide"
)

# ── Session state init ────────────────────────────────────────────────────────
if 'selected_keywords' not in st.session_state:
    st.session_state['selected_keywords'] = []
if 'original_bids' not in st.session_state:
    st.session_state['original_bids'] = {}
if 'current_bids' not in st.session_state:
    st.session_state['current_bids'] = {}
if 'campaign_created' not in st.session_state:
    st.session_state['campaign_created'] = False
if 'last_product' not in st.session_state:
    st.session_state['last_product'] = None

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛒 E-Commerce Ads")
    st.markdown("### Campaign Manager")
    st.divider()

    # Brand selector
    brands = sorted(df_products['brands'].dropna().unique().tolist())
    selected_brand = st.selectbox("Select Brand", brands)

    # Category filter
    brand_categories = df_products[
        df_products['brands'] == selected_brand
    ]['our_category'].unique().tolist()

    if len(brand_categories) > 1:
        selected_category_filter = st.selectbox(
            "Product Category",
            ['All'] + sorted(brand_categories)
        )
    else:
        selected_category_filter = (
            brand_categories[0] if brand_categories else 'All'
        )

    # Filter products by brand + category
    if selected_category_filter == 'All':
        brand_products = df_products[
            df_products['brands'] == selected_brand
        ]['product_name'].tolist()
    else:
        brand_products = df_products[
            (df_products['brands'] == selected_brand) &
            (df_products['our_category'] == selected_category_filter)
        ]['product_name'].tolist()

    # Product selector
    selected_product_name = st.selectbox("Select Product", brand_products)

    # Get product details
    selected_product = df_products[
        df_products['product_name'] == selected_product_name
    ].iloc[0]
    product_id = selected_product['product_id']
    category   = selected_product['our_category']

    # Auto reset when product changes
    if st.session_state['last_product'] != product_id:
        st.session_state['selected_keywords'] = []
        st.session_state['original_bids']     = {}
        st.session_state['current_bids']      = {}
        st.session_state['last_product']      = product_id

    st.divider()

    # Campaign dates
    st.markdown("### Campaign Dates")
    start_date = st.date_input(
        "Start Date", value=pd.Timestamp("2024-10-01")
    )
    end_date = st.date_input(
        "End Date", value=pd.Timestamp("2024-10-31")
    )

    # Daily budget
    daily_budget = st.number_input(
        "Daily Budget (₹)",
        min_value=500,
        max_value=100000,
        value=5000,
        step=500
    )

    st.divider()
    st.markdown(f"**Category:** {category}")
    st.markdown(
        f"**Keywords added:** {len(st.session_state['selected_keywords'])}"
    )

    if st.button("🔄 Reset Campaign", use_container_width=True):
        st.session_state['selected_keywords'] = []
        st.session_state['original_bids']     = {}
        st.session_state['current_bids']      = {}
        st.session_state['campaign_created']  = False
        st.rerun()

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "🔍 Keyword Planner",
    "📊 Campaign Forecast",
    "💰 Bid Manager"
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — KEYWORD PLANNER
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("### Recommended Keywords")
    st.caption(
        f"Top 20 keywords for **{selected_product_name}** "
        f"— ranked by impressions × CTR × relevance"
    )

    # Keyword count badge
    added_count = len(st.session_state['selected_keywords'])
    if added_count > 0:
        st.info(
            f"✅ {added_count} keyword(s) added to campaign — "
            f"go to **Campaign Forecast** tab to see predictions"
        )

    recommendations, product_name_rec, category_rec = (
        get_keyword_recommendations(product_id, top_n=20)
    )

    if recommendations is None or recommendations.empty:
        st.warning("No keyword recommendations found for this product.")
    else:
        # Header row
        col1, col2, col3, col4, col5, col6 = st.columns(
            [2.5, 1.2, 1.2, 1.8, 1.8, 1.2]
        )
        with col1: st.markdown("**Keyword**")
        with col2: st.markdown("**Searches/day**")
        with col3: st.markdown("**Competition**")
        with col4: st.markdown("**Impressions/day**")
        with col5: st.markdown("**Suggested Bid**")
        with col6: st.markdown("**Action**")
        st.divider()

        for _, row in recommendations.iterrows():
            kw_id    = row['keyword_id']
            kw_text  = row['keyword_text']
            is_added = kw_id in st.session_state['selected_keywords']

            # CPM bid computation
            kw_meta     = df_keywords[
                df_keywords['keyword_id'] == kw_id
            ].iloc[0]
            floor_bid = math.ceil(float(kw_meta['floor_bid_cpm']) * 2) / 2
            suggested   = float(kw_meta['suggested_bid_cpm'])
            top_of_page = float(kw_meta['top_of_page_bid_cpm'])
            default_bid = suggested

            # Impression range
            imp_range = get_impression_range(
                keyword_id=kw_id,
                cpm_bid=default_bid,
                target_date=str(start_date)
            )

            col1, col2, col3, col4, col5, col6 = st.columns([2.0, 1.0, 1.0, 1.5, 1.5, 2.0])  # col6 wider now

            with col1:
                st.markdown(f"**{kw_text}**")
                if imp_range:
                    season_emoji = (
                        "🔥" if imp_range['seasonal_multiplier'] >= 1.5
                        else "📈" if imp_range['seasonal_multiplier'] >= 1.2
                        else "📊"
                    )
                    st.caption(
                        f"{season_emoji} {imp_range['season_label']} "
                        f"· {imp_range['seasonal_multiplier']}×"
                    )

            with col2:
                st.markdown(f"{row['avg_daily_searches']:,}")

            with col3:
                comp = row['competition_score']
                if comp >= 0.7:
                    st.markdown("🔴 High")
                elif comp >= 0.4:
                    st.markdown("🟡 Medium")
                else:
                    st.markdown("🟢 Low")

            with col4:
                if imp_range:
                    st.markdown(f"**{imp_range['impressions_mid']:,}**")
                    st.caption(
                        f"↕ {imp_range['impressions_low']:,} "
                        f"– {imp_range['impressions_high']:,}"
                    )
                else:
                    st.markdown("—")

            with col5:
                st.markdown(f"**₹{suggested:.0f}** CPM")
                st.caption(
                    f"Min: ₹{floor_bid:.1f}  |  Top: ₹{top_of_page:.1f}"
                )

            with col6:
                if is_added:
                    st.success("Added ✓")
                else:
                    st.number_input(
                        "Set bid (₹ CPM)",
                        min_value=0.0,              # allow any value
                        value=float(suggested),
                        step=5.0,
                        key=f"bid_input_tab1_{kw_id}",
                        help=f"Min: ₹{floor_bid:.0f} | Suggested: ₹{suggested:.0f} | Top: ₹{top_of_page:.0f}"
                    )
                    if st.button("+ Add", key=f"add_{kw_id}"):
                        entered_bid = st.session_state.get(
                            f"bid_input_tab1_{kw_id}", suggested
                        )
                        if float(entered_bid) < float(floor_bid) - 0.01:
                            st.error(
                                f"❌ Bid ₹{entered_bid:.0f} is below minimum "
                                f"₹{floor_bid:.0f}. Please increase your bid."
                            )
                        else:
                            st.session_state['selected_keywords'].append(kw_id)
                            st.session_state['original_bids'][kw_id] = float(entered_bid)
                            st.session_state['current_bids'][kw_id]  = float(entered_bid)
                            st.rerun()

    st.divider()

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CAMPAIGN FORECAST
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("### Campaign Forecast")

    if not st.session_state['selected_keywords']:
        st.info(
            "No keywords added yet. Go to "
            "**Keyword Planner** tab and add keywords."
        )
    else:
        selected_kw_ids = st.session_state['selected_keywords']
        st.caption(
            f"{len(selected_kw_ids)} keywords · "
            f"{str(start_date)} to {str(end_date)}"
        )

        # Aggregate forecast
        total_impr_low    = 0
        total_impr_mid    = 0
        total_impr_high   = 0
        total_clicks_low  = 0
        total_clicks_mid  = 0
        total_clicks_high = 0
        total_orders_low  = 0
        total_orders_mid  = 0
        total_orders_high = 0
        total_spend       = 0
        forecasts         = []

        for kw_id in selected_kw_ids:
            kw_meta_t2   = df_keywords[df_keywords['keyword_id'] == kw_id].iloc[0]
            suggested_t2 = float(kw_meta_t2['suggested_bid_cpm'])
            current_bid  = float(
                st.session_state['current_bids'].get(kw_id, suggested_t2)
            )
            st.write(f"{kw_id}: live_bid={st.session_state.get(f'live_bid_{kw_id}', 'NOT SET')}, current_bids={st.session_state['current_bids'].get(kw_id, 'NOT SET')}, using={current_bid}")
           
            fc = forecast_keyword_v2(
                keyword_id=kw_id,
                product_id=product_id,
                cpm_bid=current_bid,
                target_date=str(start_date),
                daily_budget=daily_budget
            )
            if fc:
                total_impr_low    += fc['impressions_low']
                total_impr_mid    += fc['impressions_mid']
                total_impr_high   += fc['impressions_high']
                total_clicks_low  += fc['clicks_low']
                total_clicks_mid  += fc['clicks_mid']
                total_clicks_high += fc['clicks_high']
                total_orders_low  += fc['orders_low']
                total_orders_mid  += fc['orders_mid']
                total_orders_high += fc['orders_high']
                total_spend       += fc['daily_spend']
                forecasts.append(fc)

        # ── Summary metrics ───────────────────────────────────────────────────
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric(
                "Daily Impressions",
                f"{total_impr_mid:,}",
                f"↕ {total_impr_low:,} – {total_impr_high:,}"
            )
        with col2:
            st.metric(
                "Daily Clicks",
                f"{total_clicks_mid:,}",
                f"↕ {total_clicks_low:,} – {total_clicks_high:,}"
            )
        with col3:
            st.metric(
                "Daily Orders",
                f"{round(total_orders_mid)}",
                f"↕ {round(total_orders_low)} – {round(total_orders_high)}"
            )
        with col4:
            st.metric("Daily Spend", f"₹{total_spend:,.0f}")
        with col5:
            roas = forecasts[0]['roas'] if forecasts else 0
            st.metric("Est. ROAS", f"{roas:.1f}x")

        st.divider()

        # ── Budget utilisation ────────────────────────────────────────────────
        utilisation      = total_spend / daily_budget * 100
        suggested_budget = round(total_spend * 1.1)

        if utilisation >= 90:
            st.error(
                f"⚠️ Budget will exhaust before end of day — "
                f"{utilisation:.0f}% utilised"
            )
        elif utilisation >= 70:
            st.success(
                f"✅ Good budget utilisation — "
                f"{utilisation:.0f}% of ₹{daily_budget:,} will be spent"
            )
        elif utilisation >= 40:
            st.info(
                f"ℹ️ Moderate utilisation — {utilisation:.0f}% of budget used. "
                f"Consider adding more keywords or increasing bids."
            )
        else:
            st.warning(
                f"⚠️ Low utilisation — only {utilisation:.0f}% of "
                f"₹{daily_budget:,} budget will be spent. "
                f"Add more keywords or increase bids to improve reach."
            )
        st.caption(
            f"💡 Suggested daily budget based on your keywords: "
            f"₹{suggested_budget:,}"
        )

        st.divider()

        # ── Budget burn chart ─────────────────────────────────────────────────
        st.markdown("#### Budget Burn Forecast")

        hour_weights_list = [
            0.10, 0.05, 0.03, 0.02, 0.02, 0.03,
            0.05, 0.08, 0.12, 0.13, 0.10, 0.09,
            0.08, 0.07, 0.07, 0.07, 0.08, 0.10,
            0.12, 0.14, 0.13, 0.10, 0.08, 0.06
        ]
        total_w           = sum(hour_weights_list)
        hour_weights_norm = [w / total_w for w in hour_weights_list]

        hours         = list(range(24))
        cumulative    = []
        hourly_spends = []
        running_total = 0

        for h in hours:
            hourly_spend   = total_spend * hour_weights_norm[h]
            running_total += hourly_spend
            cumulative.append(round(min(running_total, daily_budget), 2))
            hourly_spends.append(round(hourly_spend, 2))

        burn_df = pd.DataFrame({
            'Hour'            : [f"{h:02d}:00" for h in hours],
            'Cumulative Spend': cumulative,
            'Daily Budget'    : [daily_budget] * 24,
            'Hourly Spend'    : hourly_spends
        }).set_index('Hour')

        st.line_chart(burn_df[['Cumulative Spend', 'Daily Budget']])

        st.markdown("#### Hourly Spend Distribution")
        st.bar_chart(burn_df[['Hourly Spend']])

        st.divider()

        # ── Per keyword breakdown ─────────────────────────────────────────────
        st.markdown("#### Per Keyword Breakdown")

        fc_df = pd.DataFrame([{
            'Keyword'    : fc['keyword_text'],
            'Impressions': fc['impressions_mid'],
            'Clicks'     : fc['clicks_mid'],
            'Orders'     : round(fc['orders_mid']),
            'Spend (₹)'  : f"₹{fc['daily_spend']:,.0f}",
            'ROAS'       : f"{fc['roas']:.1f}x",
            'Season'     : fc['season_label'],
        } for fc in forecasts])

        st.dataframe(fc_df, use_container_width=True, hide_index=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — BID MANAGER
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("### Bid Manager")
    st.caption("You can only increase bids — not decrease once set.")

    if not st.session_state['selected_keywords']:
        st.info(
            "No keywords in campaign yet. "
            "Add keywords from the Keyword Planner tab."
        )
    else:
        for kw_id in st.session_state['selected_keywords']:
            kw_row = df_keywords[df_keywords['keyword_id'] == kw_id]
            if kw_row.empty:
                continue
            kw_row = kw_row.iloc[0]

            floor_bid       = float(kw_row['floor_bid_cpm'])
            suggested_bid_val = float(kw_row['suggested_bid_cpm'])
            top_of_page_bid = float(kw_row['top_of_page_bid_cpm'])
            original_bid    = float(
                st.session_state['original_bids'].get(kw_id, suggested_bid_val)
            )
            current_bid     = float(
                st.session_state['current_bids'].get(kw_id, original_bid)
            )

            st.markdown(f"#### {kw_row['keyword_text']}")
            st.caption(
                f"Original bid: ₹{original_bid:.0f}  |  "
                f"Min: ₹{floor_bid:.1f}  |  "
                f"Suggested: ₹{suggested_bid_val:.0f}  |  "
                f"Top of page: ₹{top_of_page_bid:.0f}"
            )

            col1, col2, col3 = st.columns([2, 1, 1])

            with col1:
                # Don't set value= when key exists in session state
                # to avoid overriding what brand typed
                if f"bid_input_{kw_id}" not in st.session_state:
                    st.session_state[f"bid_input_{kw_id}"] = current_bid

                st.number_input(
                    "CPM Bid (₹)",
                    min_value=0.0,
                    step=5.0,
                    key=f"bid_input_{kw_id}"
                )

            # Read live bid from widget
            live_bid = float(st.session_state[f"bid_input_{kw_id}"])

            with col2:
                st.write("")
                st.write("")
                if st.button("Apply Bid", key=f"apply_{kw_id}"):
                    if live_bid >= original_bid:
                        st.session_state['current_bids'][kw_id] = live_bid
                        st.success(f"✅ Updated to ₹{live_bid:.0f}")
                    else:
                        st.error(
                            f"Cannot go below ₹{original_bid:.0f}. "
                            f"Bids can only increase."
                        )

            with col3:
                st.write("")
                st.write("")
                if live_bid > original_bid:
                    st.success(f"₹{original_bid:.0f} → ₹{live_bid:.0f}")
                elif live_bid < original_bid:
                    st.warning(f"Below original bid")
                else:
                    st.info("At original bid")

            # Metrics based on live bid
            imp_range = get_impression_range(
                keyword_id=kw_id,
                cpm_bid=live_bid,
                target_date=str(start_date)
            )

            m1, m2, m3 = st.columns(3)

            with m1:
                share_pct = (
                    imp_range['impression_share'] * 100
                    if imp_range else 0
                )
                st.metric(
                    "Impression Share",
                    f"{share_pct:.0f}%",
                    help=f"At ₹{live_bid:.0f} CPM bid"
                )

            with m2:
                if imp_range:
                    st.metric(
                        "Est. Impressions/day",
                        f"{imp_range['impressions_mid']:,}",
                        f"↕ {imp_range['impressions_low']:,} "
                        f"– {imp_range['impressions_high']:,}"
                    )

            with m3:
                if imp_range:
                    season_emoji = (
                        "🔥" if imp_range['seasonal_multiplier'] >= 1.5
                        else "📈" if imp_range['seasonal_multiplier'] >= 1.2
                        else "📊"
                    )
                    st.metric(
                        "Season",
                        f"{season_emoji} {imp_range['season_label']}"
                    )

        st.divider()