import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# --- PAGE CONFIG ---
st.set_page_config(page_title="Geospatial AI for Wind Planning", layout="wide")

st.title("🌍 AI-Based Geospatial Segmentation for Wind Turbine Planning")

# --- SIMULATE POINT CLOUD DATA ---
@st.cache_data
def generate_data():
    def cluster(center, scale, n): return center + np.random.randn(n, 3) * scale
    np.random.seed(42)
    clusters = {
        "building": cluster(np.array([50, 50, 10]), 2, 750),
        "water": cluster(np.array([20, 80, 1]), 1.5, 750),
        "vegetation": cluster(np.array([80, 20, 5]), 3, 750),
        "road": cluster(np.array([50, 80, 0.5]), 1, 750),
    }
    points = np.vstack(list(clusters.values()))
    labels = ["building"] * 750 + ["water"] * 750 + ["vegetation"] * 750 + ["road"] * 750
    df = pd.DataFrame(points, columns=["X", "Y", "Z"])
    df["Label"] = labels
    color_map = {
        "building": [200, 50, 50],
        "water": [50, 50, 200],
        "vegetation": [50, 200, 50],
        "road": [150, 150, 150]
    }
    df[["R", "G", "B"]] = df["Label"].apply(lambda x: color_map[x]).apply(pd.Series)
    return df

df = generate_data()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Segmentation Settings")
n_clusters = st.sidebar.slider("Number of clusters", 2, 6, 4)

# --- CLUSTERING ---
features = df[["X", "Y", "Z", "R", "G", "B"]]
kmeans = KMeans(n_clusters=n_clusters, random_state=0)
df["Cluster"] = kmeans.fit_predict(features)
sil = silhouette_score(features, df["Cluster"])

# --- DISPLAY METRICS ---
st.markdown(f"**Silhouette Score**: `{sil:.2f}` — Clustering cohesion")

# --- PLOTLY VISUALIZATION ---
fig = px.scatter_3d(df, x="X", y="Y", z="Z",
                    color=df["Cluster"].astype(str),
                    hover_data=["Label"],
                    title="3D Point Cloud Segmentation",
                    opacity=0.7)
st.plotly_chart(fig, use_container_width=True)

# --- REGULATORY RULE CHECK ---
st.subheader("⚖️ Turbine Siting Constraints")

summary = df.groupby("Cluster").agg({
    "Z": "mean", "G": "mean"
}).reset_index()

summary["Turbine_Excluded"] = (summary["Z"] < 2) | (summary["G"] > 100)
st.dataframe(summary.rename(columns={
    "Z": "Avg Height (Z)", "G": "Avg Green (G)", "Turbine_Excluded": "Excluded?"
}))

excluded = summary[summary["Turbine_Excluded"]]["Cluster"].tolist()
st.markdown(f"**Excluded Clusters** (too green or low): `{excluded}`")

# --- FOOTER ---
st.caption("Prototype by Savitha Narayana — Based on BIMSeg Concepts | Streamlit + Scikit-Learn + Plotly")
