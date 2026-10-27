"""
Smart Segment: customer segmentation + RFM scoring
Self-contained: reads Online_Retail.csv, trains K-Means on startup (cached).
Needs only this file and Online_Retail.csv in the same folder.
 
Run:  streamlit run app.py
"""
import os
 
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
 
st.set_page_config(page_title="Smart Segment", page_icon="🎯", layout="wide")
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "Online_Retail.csv")
FEATURES = ["Frequency", "TotalQuantity", "Monetary", "Recency"]
 
DESCRIPTIONS = {
    "Regular Customers": "Typical repeat buyers: a few orders, moderate spend, purchased recently.",
    "High-Spending Customers": "Top-tier accounts: very frequent, very large-volume orders and the highest revenue.",
    "Loyal Customers": "Frequent bulk buyers who order often and recently, with high spend.",
    "At-Risk Customers": "Only one or two orders, low spend, and no purchase for many months.",
}
STRATEGY = {
    "High-Spending Customers": "Monitor preferred products and plan capacity for large orders.",
    "At-Risk Customers": "Review recent demand before replenishing; consider customer re-engagement.",
    "Loyal Customers": "Monitor availability of frequently purchased products and plan reliable fulfillment.",
    "Regular Customers": "Use standard replenishment and efficient order processing.",
}
ICONS = {
    "Regular Customers": "🟦",
    "High-Spending Customers": "💎",
    "Loyal Customers": "💚",
    "At-Risk Customers": "⚠️",
}
 
 
# ---------------------------------------------------------------
# Data + model (runs once, then cached)
# ---------------------------------------------------------------
def read_csv_any(source):
    try:
        return pd.read_csv(source, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(source, "seek"):
            source.seek(0)
        return pd.read_csv(source, encoding="latin-1")
 
 
@st.cache_data(show_spinner="Training the model on your dataset (first run only)...")
def build(raw: pd.DataFrame):
    df = raw.dropna(subset=["CustomerID", "Description"]).copy()
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df = df[~df["InvoiceNo"].astype(str).str.startswith("C")]
    df = df[(df["Quantity"] > 0) & (df["UnitPrice"] > 0)]
    df["TotalAmount"] = df["Quantity"] * df["UnitPrice"]
 
    cust = df.groupby("CustomerID").agg(
        Frequency=("InvoiceNo", "nunique"),
        TotalQuantity=("Quantity", "sum"),
        Monetary=("TotalAmount", "sum"),
        LastPurchase=("InvoiceDate", "max"),
    ).reset_index()
    cust["Recency"] = (df["InvoiceDate"].max() - cust["LastPurchase"]).dt.days
 
    scaler = StandardScaler()
    scaled = scaler.fit_transform(cust[FEATURES])
    km = KMeans(n_clusters=4, random_state=42, n_init=10).fit(scaled)
    cust["Cluster"] = km.labels_
 
    # Name clusters from their real profile (not from fixed ids)
    prof = cust.groupby("Cluster")[FEATURES].mean()
    names = {}
    names[prof["Monetary"].idxmax()] = "High-Spending Customers"
    rest = prof.drop(index=list(names))
    names[rest["Recency"].idxmax()] = "At-Risk Customers"
    rest = prof.drop(index=list(names))
    names[rest["Monetary"].idxmax()] = "Loyal Customers"
    names[prof.drop(index=list(names)).index[0]] = "Regular Customers"
    cust["Segment"] = cust["Cluster"].map(names)
 
    summary = cust.groupby("Segment").agg(
        Customers=("CustomerID", "count"),
        Avg_Orders=("Frequency", "mean"),
        Avg_Quantity=("TotalQuantity", "mean"),
        Avg_Spend=("Monetary", "mean"),
        Avg_Recency_Days=("Recency", "mean"),
    ).round(2)
    summary["Strategy"] = summary.index.map(STRATEGY)
 
    tx = df.merge(cust[["CustomerID", "Segment"]], on="CustomerID")
    top = (tx.groupby(["Segment", "Description"])["Quantity"].sum()
             .reset_index(name="Total_Quantity_Sold")
             .sort_values(["Segment", "Total_Quantity_Sold"], ascending=[True, False])
             .groupby("Segment").head(10))
 
    prod = df.groupby("Description").agg(
        Total_Quantity_Sold=("Quantity", "sum"),
        Number_of_Orders=("InvoiceNo", "nunique"),
        Number_of_Customers=("CustomerID", "nunique"),
        Total_Revenue=("TotalAmount", "sum"),
    ).reset_index()
    hi = prod["Total_Quantity_Sold"].quantile(0.75)
    lo = prod["Total_Quantity_Sold"].quantile(0.25)
    prod["Inventory_Recommendation"] = prod["Total_Quantity_Sold"].apply(
        lambda q: "Prioritize stock monitoring" if q >= hi
        else "Review before replenishing" if q <= lo
        else "Maintain regular replenishment")
    prod = prod.sort_values("Total_Quantity_Sold", ascending=False)
 
    return {
        "kmeans": km, "scaler": scaler, "names": names,
        "cust": cust.drop(columns=["LastPurchase"]),
        "summary": summary, "top": top, "prod": prod,
    }
 
 
def get_raw():
    if os.path.exists(CSV_PATH):
        return read_csv_any(CSV_PATH)
    st.warning(f"`Online_Retail.csv` was not found in `{BASE_DIR}`. "
               "Place it next to app.py, or upload it below.")
    up = st.file_uploader("Upload Online_Retail.csv", type="csv")
    if up is None:
        st.stop()
    return read_csv_any(up)
 
 
# ---------------------------------------------------------------
# Prediction + RFM helpers
# ---------------------------------------------------------------
def clustering(M, frequency, total_quantity, monetary, recency):
    row = pd.DataFrame([{"Frequency": frequency, "TotalQuantity": total_quantity,
                         "Monetary": monetary, "Recency": recency}])[FEATURES]
    scaled = M["scaler"].transform(row)
    cid = int(M["kmeans"].predict(scaled)[0])
    dists = np.linalg.norm(M["kmeans"].cluster_centers_ - scaled, axis=1)
    return M["names"][cid], {M["names"][i]: d for i, d in enumerate(dists)}
 
 
def _pct(series, value):
    return ((series < value).mean() + (series <= value).mean()) / 2
 
 
def _score(p):
    return int(min(5, max(1, np.ceil(p * 5))))
 
 
def rfm_scores(base, recency, frequency, monetary):
    r = 1 - _pct(base["Recency"], recency)     # more recent = better
    f = _pct(base["Frequency"], frequency)
    m = _pct(base["Monetary"], monetary)
    return (_score(r), _score(f), _score(m)), (r, f, m)
 
 
def rfm_band(total):
    return "Excellent" if total >= 13 else "Strong" if total >= 10 else "Average" if total >= 7 else "Weak"
 
 
# ---------------------------------------------------------------
# UI
# ---------------------------------------------------------------
st.title("🎯 Smart Segment")
st.caption("Enter a customer's details to find their segment (Regular, Loyal, High-Spending or At-Risk) "
           "and their RFM score.")
 
try:
    M = build(get_raw())
except Exception as e:  # show the real problem on the page instead of a blank screen
    st.error("Something went wrong while loading the data or training the model.")
    st.exception(e)
    st.stop()
 
base = M["cust"]
tab_predict, tab_segments, tab_products = st.tabs(
    ["🔮 Predict Segment", "📊 Segment Overview", "🛒 Product Demand & Inventory"])
 
# ---------- Tab 1 ----------
with tab_predict:
    st.subheader("Enter Customer Details")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        recency = st.number_input("Recency (days since last order)", 0, 730, 40, 1)
    with c2:
        frequency = st.number_input("Frequency (number of orders)", 1, 1000, 4, 1)
    with c3:
        monetary = st.number_input("Monetary (total spend £)", 0.0, 1_000_000.0, 1700.0, 50.0)
    with c4:
        total_quantity = st.number_input("Total quantity bought", 1, 1_000_000, 1000, 10)
 
    if st.button("Predict Segment", type="primary"):
        label, dists = clustering(M, frequency, total_quantity, monetary, recency)
        st.success(f"The customer belongs to the '{label}' cluster")
        st.markdown(f"**{ICONS[label]} {label}**: {DESCRIPTIONS[label]}")
        st.info(f"**Recommended supply-chain strategy:** {STRATEGY[label]}")
 
        (r, f, m), (rp, fp, mp) = rfm_scores(base, recency, frequency, monetary)
        total = r + f + m
        st.subheader("RFM Analysis")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Recency score", f"{r} / 5")
        k2.metric("Frequency score", f"{f} / 5")
        k3.metric("Monetary score", f"{m} / 5")
        k4.metric("Total RFM score", f"{total} / 15", rfm_band(total), delta_color="off")
        st.write(f"**RFM code:** {r}{f}{m}")
        st.caption(f"Scores compare this customer with all {len(base):,} customers in the dataset (5 = best). "
                   f"Recency is better than {rp:.0%} of customers, frequency higher than {fp:.0%}, "
                   f"spend higher than {mp:.0%}.")
 
        with st.expander("How close is this customer to each segment?"):
            d = pd.DataFrame({"Segment": list(dists), "Distance (scaled)": np.round(list(dists.values()), 3)})
            st.dataframe(d.sort_values("Distance (scaled)"), hide_index=True, width="stretch")
 
# ---------- Tab 2 ----------
with tab_segments:
    st.subheader("Segment Summary")
    order = [s for s in ["Regular Customers", "At-Risk Customers", "Loyal Customers",
                         "High-Spending Customers"] if s in M["summary"].index]
    summ = M["summary"].loc[order]
    for col, (name, row) in zip(st.columns(len(summ)), summ.iterrows()):
        col.metric(f"{ICONS[name]} {name}", f"{int(row['Customers']):,} customers")
    st.dataframe(summ, width="stretch")
    a, b = st.columns(2)
    a.markdown("**Customers per segment**"); a.bar_chart(summ["Customers"])
    b.markdown("**Average spend per segment (£)**"); b.bar_chart(summ["Avg_Spend"])
    st.warning("The High-Spending group has only a handful of customers (very large wholesale-style "
               "accounts), so treat its averages with care.")
 
# ---------- Tab 3 ----------
with tab_products:
    st.subheader("Top Products by Segment")
    seg = st.selectbox("Segment", list(M["summary"].index))
    t = M["top"][M["top"]["Segment"] == seg].head(5).set_index("Description")["Total_Quantity_Sold"]
    st.bar_chart(t, horizontal=True)
 
    st.subheader("Inventory Planning Recommendations")
    rec = st.multiselect("Filter by recommendation", sorted(M["prod"]["Inventory_Recommendation"].unique()))
    q = st.text_input("Search product name")
    tbl = M["prod"]
    if rec:
        tbl = tbl[tbl["Inventory_Recommendation"].isin(rec)]
    if q:
        tbl = tbl[tbl["Description"].str.contains(q, case=False, na=False)]
    st.dataframe(tbl.head(200), hide_index=True, width="stretch")
    st.caption(f"Showing up to 200 of {len(tbl):,} products.")