"""
Latur District Socio-Economic Survey (Jilha Samajik Aarthik Sarvekshan) 2021 — Dashboard
Source: DSA_2021_Latur.pdf (District Statistical Office, Latur / Directorate of Economics
& Statistics, Government of Maharashtra)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ----------------------------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------------------------
st.set_page_config(
    page_title="Latur District Survey 2021 Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------------------------------------------------------
# GLOBAL STYLES
# ----------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem; font-weight: 800; color: #1a3c6e; margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.05rem; color: #5a6b7d; margin-top: 0;
    }
    div[data-testid="stMetric"] {
        background-color: #f4f7fb; border: 1px solid #e3e9f0;
        border-radius: 10px; padding: 12px 14px;
    }
    .section-title {
        font-size: 1.4rem; font-weight: 700; color: #1a3c6e;
        border-bottom: 3px solid #ff8a3d; padding-bottom: 6px; margin-top: 10px;
    }
    .note-box {
        background-color: #fff8ec; border-left: 4px solid #ff8a3d;
        padding: 10px 14px; border-radius: 6px; font-size: 0.9rem; color: #6a5a3d;
    }
</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------------
# DATA (extracted from the source PDF — see notes per section for table refs)
# ----------------------------------------------------------------------------

TALUKAS = ["Latur", "Renapur", "Ahmedpur", "Chakur", "Jalkot",
           "Udgir", "Nilanga", "Deoni", "Ausa", "Shirur Anantpal"]

villages_df = pd.DataFrame({
    "Taluka": TALUKAS,
    "Inhabited": [120, 79, 123, 47, 83, 48, 126, 152, 53, 97],
    "Uninhabited": [2, 0, 0, 0, 2, 0, 4, 10, 1, 1],
})
villages_df["Total"] = villages_df["Inhabited"] + villages_df["Uninhabited"]

income_df = pd.DataFrame({
    "Series": ["Current Prices (2019-20)", "Current Prices (2019-20)",
               "2011-12 Base Series (2019-20)", "2011-12 Base Series (2019-20)"],
    "Measure": ["District Income (₹ lakh)", "Per-Capita Income (₹)",
                "District Income (₹ lakh)", "Per-Capita Income (₹)"],
    "Latur District": [40123, 152473, 29959, 113849],
    "Maharashtra State": [2818555, 229488, 2134065, 173757],
})

banking_df = pd.DataFrame({
    "Indicator": ["Deposits (₹ lakh)", "Advances (₹ lakh)", "Per-Capita Deposit (₹)"],
    "Latur District": [855836, 607545, 26972],
    "Maharashtra State": [29968803, 20918212, 1006721],
})
banking_df["CD Ratio (District) %"] = round(607545 / 855836 * 100, 1)

landholding_df = pd.DataFrame({
    "Size Class (Hectares)": ["0.00 - 0.99", "1.00 - 1.99", "2.00 - 4.99",
                               "5.00 - 9.99", "10.00 - 19.99", "20.00 & above"],
    "Number of Holdings": [190447, 148729, 83406, 11115, 970, 72],
    "Area (Hectares)": [99772.86, 200475.01, 238286.61, 72178.07, 13192.39, 2506.23],
    "% of Holdings": [43.81, 34.21, 19.19, 2.56, 0.22, 0.02],
    "% of Area": [15.93, 32.00, 38.04, 11.52, 2.11, 0.40],
})

literacy_df = pd.DataFrame({
    "Census Year": [1991, 1991, 1991, 2001, 2001, 2001, 2011, 2011, 2011],
    "Category": ["Female", "Total", "Male"] * 3,
    "Literacy Rate (%)": [39.74, 55.57, 70.47, 59.40, 71.50, 82.90, 69.60, 77.30, 84.40],
})

population_growth_df = pd.DataFrame({
    "Value (thousands)": [87, 139, 213, 342, 490, 625, 732, 818, 910, 1049,
                           1080, 1293, 1335, 1590, 1677, 1829, 2080, 2454],
    "Index": list(range(1, 19)),
})

schemes_df = pd.DataFrame({
    "Scheme": ["MGNREGA", "NULM (NRLM-POP, Urban Livelihood)", "ICDS", "PMAY-G (beneficiaries)"],
    "Metric": ["Expenditure (₹ lakh)", "Expenditure (₹ lakh)", "Expenditure (₹ lakh)", "Houses sanctioned (No.)"],
    "2018-19": [187.21, 99.88, 1554.8, None],
    "2019-20": [None, 170.2, 550, None],
    "2020-21 Target": [4601.21, 202.66, 1127.16, 1774],
    "2020-21 Achievement": [None, 67.42, 977.64, 302],
})

elections_df = pd.DataFrame({
    "Constituency": ["Lok Sabha - Latur", "AC 234 Latur Rural", "AC 235 Latur City",
                      "AC 236 Ahmedpur", "AC 237 Udgir (SC)", "AC 238 Nilanga", "AC 239 Ausa"],
    "Type": ["Lok Sabha 2019", "Vidhan Sabha 2019", "Vidhan Sabha 2019", "Vidhan Sabha 2019",
             "Vidhan Sabha 2019", "Vidhan Sabha 2019", "Vidhan Sabha 2019"],
    "Total Electors": [1886657, 321972, 372369, 321329, 299139, 316292, 283461],
    "Male Votes": [625060, 107638, 112089, 113832, 94999, 103656, 98635],
    "Female Votes": [545401, 91398, 98363, 99225, 82984, 92586, 87821],
    "Other Votes": [6078, 563, 1335, 1924, 1947, 1076, 878],
    "Total Votes Polled": [1176539, 199599, 211787, 214981, 179930, 197318, 187334],
    "Turnout (%)": [62.36, 61.99, 56.88, 66.90, 60.15, 62.38, 66.09],
    "NOTA Votes": [6564, 27500, 727, 1762, 1097, 1272, 783],
})

budget_df = pd.DataFrame({
    "Plan Head": ["District Annual Plan (General)", "Gram Panchayat Fund Scheme",
                  "Rural Livelihood / Related Programmes"],
    "Target (₹ lakh)": [73050.85, 22649.04, 230078.92],
    "Achievement (₹ lakh)": [42016.70, 15062.07, 197012.64],
})
budget_df["Utilisation (%)"] = round(budget_df["Achievement (₹ lakh)"] / budget_df["Target (₹ lakh)"] * 100, 1)

apmc_df = pd.DataFrame({
    "Category": ["Market Fee / Cess", "Licence & Other Fees", "Weighment Charges",
                 "Other Charges", "Development Fund Contribution", "Miscellaneous",
                 "Rent / Lease Income", "Other Income", "Sundry Income"],
    "2017-18": [432.05, 153.44, 0.20, 26.57, 525.12, 3689.54, 0.12, 123.80, 0.00],
    "2018-19": [246.19, 149.25, 0.00, 0.00, 582.61, 435.16, 3.26, 240.81, 0.00],
    "2019-20": [57.74, 124.03, 0.00, 0.00, 905.78, 0.45, 0.00, 143.70, 0.00],
})

# ----------------------------------------------------------------------------
# SIDEBAR
# ----------------------------------------------------------------------------
st.sidebar.title("📊 Latur District Survey")
st.sidebar.caption("Jilha Samajik Aarthik Sarvekshan — 2021")
section = st.sidebar.radio(
    "Navigate to section",
    ["🏠 District Overview", "💰 District Income", "🏦 Banking & Finance",
     "🌾 Agriculture & Land", "👥 Population & Literacy", "🏛️ Government Schemes",
     "🗳️ Elections", "📋 Budget & APMC", "🤖 ML Insights", "ℹ️ About this Dashboard"],
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div class='note-box'>⚠️ The source PDF uses a legacy, non-Unicode Marathi font "
    "(DVOT-Surekh). Marathi text does not extract or render reliably from it, even visually. "
    "All figures shown here are numeric values read directly from the report's tables; "
    "labels are reconstructed from table structure, English/bilingual headers in the report, "
    "and the standard format used across Maharashtra's district socio-economic surveys.</div>",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------------
# HEADER
# ----------------------------------------------------------------------------
st.markdown("<p class='main-header'>Latur District — Socio-Economic Survey Dashboard</p>", unsafe_allow_html=True)
st.markdown("<p class='sub-header'>District Statistical Abstract 2021 · Directorate of Economics & Statistics, Government of Maharashtra</p>", unsafe_allow_html=True)
st.markdown("---")

# ----------------------------------------------------------------------------
# SECTION: OVERVIEW
# ----------------------------------------------------------------------------
if section == "🏠 District Overview":
    st.markdown("<p class='section-title'>District Profile</p>", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Geographical Area", "7,157 km²", "2.32% of Maharashtra")
    c2.metric("Talukas", "10")
    c3.metric("Total Villages", "948", "928 inhabited + 20 uninhabited")
    c4.metric("Share of Marathwada Area", "11.04%")

    st.markdown("#### Taluka-wise Villages")
    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = px.bar(
            villages_df.sort_values("Total", ascending=True),
            x="Total", y="Taluka", orientation="h",
            text="Total", color="Total", color_continuous_scale="Blues",
            title="Number of Villages by Taluka",
        )
        fig.update_layout(coloraxis_showscale=False, height=420)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(villages_df, use_container_width=True, hide_index=True)

    st.markdown(
        "<div class='note-box'>Note: village figures (Table on district area/villages) are "
        "assigned to talukas using the standard administrative ordering used in Maharashtra "
        "government publications for Latur district; the underlying row-level totals "
        "(928 inhabited, 20 uninhabited, 948 total, per Census-linked data) are taken directly "
        "from the report.</div>", unsafe_allow_html=True,
    )

    st.markdown("#### All Talukas")
    st.write(", ".join(TALUKAS))

# ----------------------------------------------------------------------------
# SECTION: DISTRICT INCOME
# ----------------------------------------------------------------------------
elif section == "💰 District Income":
    st.markdown("<p class='section-title'>District Domestic Product (Table 2.1 – 2.4)</p>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("District Income 2019-20 (Current Prices)", "₹40,123 lakh")
    c2.metric("Per-Capita Income 2019-20 (Current Prices)", "₹1,52,473")

    tab1, tab2 = st.tabs(["Current Prices", "2011-12 Base Series (Constant Prices)"])

    with tab1:
        sub = income_df[income_df["Series"].str.startswith("Current")]
        col1, col2 = st.columns(2)
        for i, row in sub.reset_index(drop=True).iterrows():
            fig = go.Figure(data=[
                go.Bar(name="Latur District", x=[row["Measure"]], y=[row["Latur District"]]),
                go.Bar(name="Maharashtra State", x=[row["Measure"]], y=[row["Maharashtra State"]]),
            ])
            fig.update_layout(barmode="group", title=row["Measure"], height=380,
                               yaxis_title="₹ (log scale)", yaxis_type="log")
            (col1 if i == 0 else col2).plotly_chart(fig, use_container_width=True)

    with tab2:
        sub = income_df[income_df["Series"].str.startswith("2011")]
        col1, col2 = st.columns(2)
        for i, row in sub.reset_index(drop=True).iterrows():
            fig = go.Figure(data=[
                go.Bar(name="Latur District", x=[row["Measure"]], y=[row["Latur District"]]),
                go.Bar(name="Maharashtra State", x=[row["Measure"]], y=[row["Maharashtra State"]]),
            ])
            fig.update_layout(barmode="group", title=row["Measure"], height=380,
                               yaxis_title="₹ (log scale)", yaxis_type="log")
            (col1 if i == 0 else col2).plotly_chart(fig, use_container_width=True)

    st.markdown("#### Full Table")
    st.dataframe(income_df, use_container_width=True, hide_index=True)
    st.caption("Source: District at a Glance table (rows 1.1–2.4), Latur District Survey 2021. Values in ₹ lakh / ₹ per capita.")

# ----------------------------------------------------------------------------
# SECTION: BANKING
# ----------------------------------------------------------------------------
elif section == "🏦 Banking & Finance":
    st.markdown("<p class='section-title'>Banking Indicators (as of 31 March 2020)</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Deposits", "₹8,558.36 crore")
    c2.metric("Advances", "₹6,075.45 crore")
    c3.metric("Credit-Deposit Ratio", f"{banking_df['CD Ratio (District) %'].iloc[0]}%")

    fig = px.bar(
        banking_df, x="Indicator", y=["Latur District", "Maharashtra State"],
        barmode="group", title="Latur District vs Maharashtra State",
        log_y=True,
    )
    fig.update_layout(height=460, yaxis_title="₹ (log scale)")
    st.plotly_chart(fig, use_container_width=True)

    st.dataframe(banking_df, use_container_width=True, hide_index=True)
    st.caption("Source: District at a Glance — Banking table, Latur District Survey 2021.")

# ----------------------------------------------------------------------------
# SECTION: AGRICULTURE
# ----------------------------------------------------------------------------
elif section == "🌾 Agriculture & Land":
    st.markdown("<p class='section-title'>Operational Landholdings — Agriculture Census 2015-16</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Total Holdings", f"{landholding_df['Number of Holdings'].sum():,}")
    c2.metric("Total Area", f"{landholding_df['Area (Hectares)'].sum():,.2f} ha")
    c3.metric("Marginal + Small Holdings (<2 ha)", f"{landholding_df['% of Holdings'][:2].sum():.1f}%")

    col1, col2 = st.columns(2)
    with col1:
        fig = px.pie(landholding_df, names="Size Class (Hectares)", values="% of Holdings",
                     title="Share of Holdings by Size Class", hole=0.4)
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.pie(landholding_df, names="Size Class (Hectares)", values="% of Area",
                      title="Share of Cultivated Area by Size Class", hole=0.4)
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.bar(landholding_df, x="Size Class (Hectares)",
                  y=["Number of Holdings"], title="Number of Holdings by Size Class",
                  text_auto=True)
    fig3.update_layout(height=420)
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        "<div class='note-box'>Reading: 43.8% of all landholdings in the district are below "
        "1 hectare (marginal), but these account for only ~16% of total cultivated area — "
        "consistent with a highly fragmented landholding pattern typical of the Marathwada "
        "region.</div>", unsafe_allow_html=True,
    )

    st.dataframe(landholding_df, use_container_width=True, hide_index=True)
    st.caption("Source: Table 1.x, Agriculture Census 2015-16 data, Latur District Survey 2021.")

# ----------------------------------------------------------------------------
# SECTION: POPULATION & LITERACY
# ----------------------------------------------------------------------------
elif section == "👥 Population & Literacy":
    st.markdown("<p class='section-title'>Literacy Rate Trend (Census 1991 / 2001 / 2011)</p>", unsafe_allow_html=True)

    fig = px.bar(literacy_df, x="Census Year", y="Literacy Rate (%)", color="Category",
                 barmode="group", text="Literacy Rate (%)",
                 color_discrete_map={"Male": "#2b6cb0", "Female": "#dd6b20", "Total": "#38a169"})
    fig.update_layout(height=440)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Male Literacy 2011", "84.40%", "+13.93 pts since 1991")
    c2.metric("Female Literacy 2011", "69.60%", "+29.86 pts since 1991")
    c3.metric("Total Literacy 2011", "77.30%", "+21.73 pts since 1991")

    st.markdown("<p class='section-title'>Population Growth Trend (Census 1961–2011)</p>", unsafe_allow_html=True)
    fig2 = px.bar(population_growth_df, x="Index", y="Value (thousands)",
                  title="Population Growth Chart (as plotted in the report, values in thousands)")
    fig2.update_layout(height=420, xaxis_title="Chart data points (chronological order)")
    st.plotly_chart(fig2, use_container_width=True)
    st.markdown(
        "<div class='note-box'>This chart reproduces the 18 data points plotted in the "
        "report's district population-growth figure (Census decades 1961–2011). Because the "
        "Marathi axis labels could not be reliably read from the source file (see the note in "
        "the sidebar), the series is shown in chronological order without individually "
        "labelling each bar as Rural / Urban / Total.</div>", unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# SECTION: GOVERNMENT SCHEMES
# ----------------------------------------------------------------------------
elif section == "🏛️ Government Schemes":
    st.markdown("<p class='section-title'>Rural Development & Welfare Scheme Performance</p>", unsafe_allow_html=True)
    st.caption("Selected schemes with clearly attributable figures from the district scheme-performance table (Target vs Achievement).")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MGNREGA Expenditure 2020-21 (Target)", "₹4,601.21 lakh")
    c2.metric("NULM Expenditure 2020-21 (Achieved)", "₹67.42 lakh")
    c3.metric("ICDS Expenditure 2020-21 (Achieved)", "₹977.64 lakh")
    c4.metric("PMAY-G Houses Sanctioned", "1,774", "302 completed")

    st.dataframe(schemes_df, use_container_width=True, hide_index=True)

    plot_df = schemes_df.melt(id_vars=["Scheme", "Metric"],
                               value_vars=["2018-19", "2019-20", "2020-21 Target", "2020-21 Achievement"],
                               var_name="Period", value_name="Value").dropna()
    fig = px.bar(plot_df, x="Scheme", y="Value", color="Period", barmode="group",
                 title="Scheme Financials Across Years (₹ lakh, except PMAY-G in units)")
    fig.update_layout(height=460)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown(
        "<div class='note-box'>The full scheme-performance table in the report spans ~25 "
        "national/state schemes (BRGF, PMGSY, NSAP, NRDWP, PMKSY, IWMP, NLRMP, DDUGJY, AMRUT, "
        "UDAY, PMFBY, PMUY and others). Most of these rows are marked 'न.उ.' (not applicable / "
        "not reported at district level in this table) in the source, so only the schemes with "
        "clean numeric entries are charted here to avoid misrepresenting the data.</div>",
        unsafe_allow_html=True,
    )

# ----------------------------------------------------------------------------
# SECTION: ELECTIONS
# ----------------------------------------------------------------------------
elif section == "🗳️ Elections":
    st.markdown("<p class='section-title'>2019 Lok Sabha & Vidhan Sabha Elections</p>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Lok Sabha Turnout 2019", "62.36%")
    c2.metric("Total Electors (Lok Sabha)", "18,86,657")
    c3.metric("Highest AC Turnout", "66.90% (Ahmedpur)")

    fig = px.bar(elections_df, x="Constituency", y="Turnout (%)", color="Type",
                 text="Turnout (%)", title="Voter Turnout by Constituency")
    fig.update_layout(height=440, xaxis_tickangle=-30)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        fig2 = px.bar(elections_df, x="Constituency", y=["Male Votes", "Female Votes", "Other Votes"],
                      title="Votes Polled by Gender", barmode="stack")
        fig2.update_layout(height=420, xaxis_tickangle=-30)
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        fig3 = px.bar(elections_df, x="Constituency", y="NOTA Votes",
                      title="NOTA Votes by Constituency", text="NOTA Votes")
        fig3.update_layout(height=420, xaxis_tickangle=-30)
        st.plotly_chart(fig3, use_container_width=True)

    st.dataframe(elections_df, use_container_width=True, hide_index=True)
    st.caption("Source: Table 10.5, Latur District Survey 2021 (Election data, 2019 Lok Sabha & Vidhan Sabha).")

# ----------------------------------------------------------------------------
# SECTION: BUDGET & APMC
# ----------------------------------------------------------------------------
elif section == "📋 Budget & APMC":
    st.markdown("<p class='section-title'>District Annual Plan — Target vs Achievement (2020-21)</p>", unsafe_allow_html=True)

    col1, col2 = st.columns([1.3, 1])
    with col1:
        fig = go.Figure()
        fig.add_trace(go.Bar(name="Target (₹ lakh)", x=budget_df["Plan Head"], y=budget_df["Target (₹ lakh)"]))
        fig.add_trace(go.Bar(name="Achievement (₹ lakh)", x=budget_df["Plan Head"], y=budget_df["Achievement (₹ lakh)"]))
        fig.update_layout(barmode="group", height=440, title="District Plan Outlay: Target vs Achievement")
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        st.dataframe(budget_df, use_container_width=True, hide_index=True)
        for _, row in budget_df.iterrows():
            st.metric(row["Plan Head"], f"{row['Utilisation (%)']}% utilised")

    st.markdown("<p class='section-title'>Agricultural Produce Market Committee (APMC) — Income Heads</p>", unsafe_allow_html=True)
    plot_apmc = apmc_df.melt(id_vars="Category", var_name="Year", value_name="₹ lakh")
    fig4 = px.bar(plot_apmc, x="Category", y="₹ lakh", color="Year", barmode="group",
                  title="APMC Income by Category, 2017-18 to 2019-20")
    fig4.update_layout(height=460, xaxis_tickangle=-30)
    st.plotly_chart(fig4, use_container_width=True)
    st.dataframe(apmc_df, use_container_width=True, hide_index=True)
    st.caption("Source: Table 10.6 (District Plan / Gram Panchayat / APMC financials), Latur District Survey 2021.")

# ----------------------------------------------------------------------------
# SECTION: ML INSIGHTS
# ----------------------------------------------------------------------------
elif section == "🤖 ML Insights":
    st.markdown("<p class='section-title'>Predictive Models & Interactive What-If Tools</p>", unsafe_allow_html=True)
    st.markdown(
        "<div class='note-box'>These models are trained live on the figures extracted from "
        "the report. The underlying datasets are small (a handful of data points per series), "
        "which is typical for an annual district statistical publication — so treat outputs as "
        "indicative trend estimates, not precision forecasts. Each tab shows the model's fit "
        "quality (R²) so you can judge how much to trust it.</div>", unsafe_allow_html=True,
    )

    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs(
        ["📈 Literacy Forecast", "🗳️ Turnout Predictor", "🏘️ Taluka Clustering", "🌾 Landholding Pareto Model"]
    )

    # ---------------- 1. LITERACY FORECAST (Linear Regression) ----------------
    with ml_tab1:
        st.markdown("#### Forecast literacy rate by category using linear regression on Census years")
        st.caption("Model: separate `LinearRegression` fit per category on 3 Census points (1991, 2001, 2011).")

        target_year = st.slider("Project literacy rate for year:", min_value=1991, max_value=2041,
                                 value=2021, step=1, key="lit_year")

        results = []
        fig = go.Figure()
        colors = {"Male": "#2b6cb0", "Female": "#dd6b20", "Total": "#38a169"}
        for cat in ["Male", "Female", "Total"]:
            sub = literacy_df[literacy_df["Category"] == cat].sort_values("Census Year")
            X = sub[["Census Year"]].values
            y = sub["Literacy Rate (%)"].values
            model = LinearRegression().fit(X, y)
            r2 = r2_score(y, model.predict(X))
            pred = float(model.predict([[target_year]])[0])
            pred = max(0, min(100, pred))
            results.append({"Category": cat, "Predicted Literacy Rate (%)": round(pred, 2),
                             "Model R²": round(r2, 4), "Slope (pts/year)": round(model.coef_[0], 3)})

            years_line = np.linspace(1991, max(2041, target_year), 60)
            preds_line = np.clip(model.predict(years_line.reshape(-1, 1)), 0, 100)
            fig.add_trace(go.Scatter(x=years_line, y=preds_line, mode="lines",
                                      name=f"{cat} (trend)", line=dict(color=colors[cat], dash="dash")))
            fig.add_trace(go.Scatter(x=sub["Census Year"], y=sub["Literacy Rate (%)"], mode="markers",
                                      name=f"{cat} (actual)", marker=dict(color=colors[cat], size=10)))
            fig.add_trace(go.Scatter(x=[target_year], y=[pred], mode="markers",
                                      name=f"{cat} (prediction)", marker=dict(color=colors[cat], size=14, symbol="star")))

        fig.update_layout(height=460, title=f"Literacy Rate Trend & Projection to {target_year}",
                           yaxis_title="Literacy Rate (%)", xaxis_title="Year")
        st.plotly_chart(fig, use_container_width=True)

        res_df = pd.DataFrame(results)
        c1, c2, c3 = st.columns(3)
        for col, row in zip([c1, c2, c3], res_df.to_dict("records")):
            col.metric(f"{row['Category']} literacy, {target_year}", f"{row['Predicted Literacy Rate (%)']}%",
                       f"R² = {row['Model R²']}")
        st.dataframe(res_df, use_container_width=True, hide_index=True)
        st.caption("Only 3 historical points per category are available, so treat projections far "
                   "from 1991–2011 (especially post-2011) as rough extrapolation, not a validated forecast.")

    # ---------------- 2. ELECTION TURNOUT PREDICTOR (Linear Regression) ----------------
    with ml_tab2:
        st.markdown("#### Predict voter turnout from constituency size & gender composition")
        st.caption("Model: `LinearRegression` trained on the 7 constituencies (Lok Sabha + 6 Assembly seats) in the report.")

        feat_df = elections_df.copy()
        feat_df["Female Share (%)"] = feat_df["Female Votes"] / (feat_df["Male Votes"] + feat_df["Female Votes"]) * 100
        X = feat_df[["Total Electors", "Female Share (%)"]].values
        y = feat_df["Turnout (%)"].values

        turnout_model = LinearRegression().fit(X, y)
        preds = turnout_model.predict(X)
        r2 = r2_score(y, preds)

        col1, col2 = st.columns(2)
        with col1:
            in_electors = st.number_input("Total Electors", min_value=50000, max_value=3000000,
                                           value=320000, step=10000)
        with col2:
            in_fshare = st.slider("Female Vote Share (%)", 30.0, 60.0, 46.0, 0.5)

        pred_turnout = float(turnout_model.predict([[in_electors, in_fshare]])[0])
        st.metric("Predicted Turnout", f"{pred_turnout:.2f}%")
        st.caption(f"Model fit on historical data: R² = {r2:.3f} · "
                   f"Coefficients — Electors: {turnout_model.coef_[0]:.6f}, "
                   f"Female Share: {turnout_model.coef_[1]:.3f}, Intercept: {turnout_model.intercept_:.2f}")

        comp_df = feat_df[["Constituency", "Total Electors", "Female Share (%)", "Turnout (%)"]].copy()
        comp_df["Predicted Turnout (%)"] = np.round(preds, 2)
        fig2 = px.scatter(comp_df, x="Turnout (%)", y="Predicted Turnout (%)", text="Constituency",
                          title="Actual vs Model-Predicted Turnout (in-sample)")
        fig2.add_shape(type="line", x0=50, y0=50, x1=70, y1=70, line=dict(dash="dash", color="gray"))
        fig2.update_traces(textposition="top center", marker=dict(size=10, color="#2b6cb0"))
        fig2.update_layout(height=440)
        st.plotly_chart(fig2, use_container_width=True)
        st.dataframe(comp_df, use_container_width=True, hide_index=True)
        st.caption("With only 7 training rows this model is illustrative — useful for exploring "
                   "directional what-if effects, not for precise electoral forecasting.")

    # ---------------- 3. TALUKA CLUSTERING (KMeans) ----------------
    with ml_tab3:
        st.markdown("#### Group talukas by village composition using K-Means clustering")
        st.caption("Model: `KMeans` on standardized Inhabited / Uninhabited village counts per taluka.")

        k = st.slider("Number of clusters (k)", min_value=2, max_value=4, value=3, key="kmeans_k")

        feat = villages_df[["Inhabited", "Uninhabited"]].values
        scaler = StandardScaler()
        feat_scaled = scaler.fit_transform(feat)

        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(feat_scaled)
        cluster_df = villages_df.copy()
        cluster_df["Cluster"] = kmeans.labels_.astype(str)

        fig3 = px.scatter(cluster_df, x="Inhabited", y="Uninhabited", color="Cluster", text="Taluka",
                          size="Total", title=f"Taluka Clusters by Village Composition (k={k})")
        fig3.update_traces(textposition="top center")
        fig3.update_layout(height=480)
        st.plotly_chart(fig3, use_container_width=True)

        st.dataframe(cluster_df.sort_values("Cluster"), use_container_width=True, hide_index=True)
        st.caption("Clustering is based only on village counts (the one clean feature set available "
                   "at taluka level in the source tables). Adding more taluka-level indicators "
                   "(population, literacy, income) would make this more meaningful, but those "
                   "breakdowns weren't reliably extractable from the source PDF's garbled Marathi labels.")

    # ---------------- 4. LANDHOLDING PARETO MODEL (Log-Log Regression) ----------------
    with ml_tab4:
        st.markdown("#### Model the landholding-size vs. holding-count relationship")
        st.caption("Model: `LinearRegression` on log(size midpoint) vs log(number of holdings) — "
                   "a classic power-law / Pareto-style fit for landholding distributions.")

        midpoints = np.array([0.5, 1.5, 3.5, 7.5, 15.0, 25.0])
        lh = landholding_df.copy()
        lh["Size Midpoint (ha)"] = midpoints
        X = np.log(lh[["Size Midpoint (ha)"]].values)
        y = np.log(lh["Number of Holdings"].values)

        pareto_model = LinearRegression().fit(X, y)
        r2 = r2_score(y, pareto_model.predict(X))

        in_size = st.slider("Hypothetical landholding size (hectares)", 0.2, 30.0, 4.0, 0.1)
        pred_log_count = pareto_model.predict([[np.log(in_size)]])[0]
        pred_count = float(np.exp(pred_log_count))
        st.metric(f"Predicted number of holdings near {in_size} ha", f"{pred_count:,.0f}")
        st.caption(f"Model fit: R² = {r2:.3f} (log-log) · Power-law exponent ≈ {pareto_model.coef_[0]:.2f}")

        size_range = np.linspace(0.3, 28, 100)
        fitted_counts = np.exp(pareto_model.predict(np.log(size_range).reshape(-1, 1)))
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(x=lh["Size Midpoint (ha)"], y=lh["Number of Holdings"], mode="markers",
                                   name="Actual (by size class)", marker=dict(size=12, color="#2b6cb0")))
        fig4.add_trace(go.Scatter(x=size_range, y=fitted_counts, mode="lines",
                                   name="Fitted power-law trend", line=dict(color="#dd6b20", dash="dash")))
        fig4.add_trace(go.Scatter(x=[in_size], y=[pred_count], mode="markers", name="Your prediction",
                                   marker=dict(size=16, color="#e53e3e", symbol="star")))
        fig4.update_layout(height=460, xaxis_type="log", yaxis_type="log",
                           xaxis_title="Landholding Size (hectares, log scale)",
                           yaxis_title="Number of Holdings (log scale)",
                           title="Landholding Size vs. Count (log-log)")
        st.plotly_chart(fig4, use_container_width=True)
        st.caption("Confirms the strongly fragmented pattern seen in the Agriculture & Land section: "
                   "holding counts fall off sharply as size increases.")

# ----------------------------------------------------------------------------
# SECTION: ABOUT
# ----------------------------------------------------------------------------
elif section == "ℹ️ About this Dashboard":
    st.markdown("<p class='section-title'>About This Report & Dashboard</p>", unsafe_allow_html=True)
    st.markdown("""
This dashboard is built from **DSA_2021_Latur.pdf** — the *Jilha Samajik Aarthik Sarvekshan*
(District Socio-Economic Survey / District Statistical Abstract), Latur, 2021 — a 202-page
annual statistical publication issued by the District Statistical Office, Latur (part of the
Directorate of Economics & Statistics, Government of Maharashtra).

**What the report covers (full table of contents, ~30 chapters):** district geography,
climate & rainfall, district income & GDDP, land use & agriculture, irrigation, animal
husbandry & fisheries, forests, industries, cooperation, banking & finance, prices, transport
& communication, power, employment & rural livelihood schemes (MGNREGA, NRLM, NULM, PMAY,
PMFBY, etc.), education, health & family welfare (including immunization coverage), housing,
social welfare, the District Annual Plan, and the 2019 general elections.

**Why this dashboard is a curated subset, not a page-by-page reproduction:**
""")
    st.markdown(
        "<div class='note-box'>"
        "<b>Font/encoding issue in the source PDF:</b> the report's Marathi text is set in a "
        "legacy 8-bit TrueType font (DVOT-Surekh / DVOT-SurekhMR), which does not map to "
        "standard Unicode. This means Marathi labels come out garbled both in machine text "
        "extraction <i>and</i> in the rasterized/visual rendering of the pages — this is a "
        "defect in the source file itself, not a limitation of reading it. Numbers, English "
        "text, and bilingual table headers (e.g. scheme names like MGNREGA, NULM, ICDS, AICTE) "
        "render correctly and were used to build every figure in this dashboard."
        "</div>", unsafe_allow_html=True,
    )
    st.markdown("""
**Approach taken:**
- Every number shown is read directly from the report's data tables (income, banking,
  landholding, literacy, elections, schemes, budget, APMC).
- Row/column labels for ambiguous or heavily garbled tables (e.g., the full land-use time
  series, detailed taluka-wise education and health tables) were **not** guessed at and are
  intentionally left out rather than risk misrepresenting the source.
- Taluka names and standard section names are supplied from the well-established structure
  used across all Maharashtra district socio-economic surveys, cross-checked against the
  real administrative geography of Latur district.

**Sections included:** District Overview · District Income · Banking & Finance ·
Agriculture & Land · Population & Literacy · Government Schemes · Elections ·
Budget & APMC · ML Insights.

**ML Insights tab:** four interactive, live-trained models built on the extracted data —
a literacy-rate forecaster (linear regression per gender category), an election turnout
predictor (linear regression, with adjustable elector count / gender-share inputs), a
K-Means clustering of talukas by village composition, and a log-log power-law model of the
landholding size-vs-count relationship. Each model reports its R² so you can judge fit
quality against the small underlying sample sizes.
""")