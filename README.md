# AI-Based Geospatial Segmentation for Wind Turbine Planning
This project is a prototype inspired by my work on BIMSeg. It simulates AI-driven segmentation of geospatial data (e.g., buildings, vegetation, water) to support automated decision-making in wind turbine site selection.

✅ Built using **Streamlit**, **scikit-learn**, **NumPy**, **pandas**, and **Plotly**

## Project Overview

In real-world wind turbine approval workflows, decision-makers must analyze:
- **Spatial constraints** (e.g., vegetation, buildings, roads, protected areas)
- **Regulatory rules** (e.g., minimum distances to structures or nature zones)
- **Environmental data** (e.g., topography, land use)

This prototype:
- Simulates 3D geospatial point cloud data
- Uses **K-Means clustering** to segment features like buildings, vegetation, and water
- Applies **rule-based filtering** to identify areas **excluded from turbine installation**
- Visualizes results in 3D with interactive filtering

---

## Live Demo

👉 [Click here to try the app on Streamlit Cloud]
([https://YOUR-STREAMLIT-LINK-HERE](https://bimseg-app-wwdbdht2yxdcgrsaabwlkt.streamlit.app/))

---

## How It Works

- **Data Simulation**: Generates synthetic geospatial point cloud data with RGB and elevation
- **Clustering**: K-Means applied to spatial + color features
- **Evaluation**: Silhouette Score shown for clustering quality
- **Exclusion Rules**:
  - Exclude zones with average height `< 2m`
  - Exclude zones where vegetation (green channel) dominates
- **Visualization**: Interactive 3D plot using Plotly

---

## Technologies Used

- [Streamlit](https://streamlit.io/) – Web interface
- [Scikit-Learn](https://scikit-learn.org/) – Clustering & evaluation
- [Plotly](https://plotly.com/python/) – 3D visualization
- [NumPy](https://numpy.org/), [pandas](https://pandas.pydata.org/) – Data processing

---

## 📁 How to Run Locally

```bash
git clone https://github.com/savdn/bimseg-streamlit.git
cd bimseg-streamlit
pip install -r requirements.txt
streamlit run app.py
