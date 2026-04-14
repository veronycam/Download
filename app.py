import streamlit as st
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

# --- Page Config ---
st.set_page_config(page_title="Iris Dataset Explorer", page_icon="🌸", layout="wide")

st.title("🌸 Iris Dataset Explorer")
st.markdown("Interactive dashboard for exploring the classic Iris flower dataset.")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Iris.csv")
    return df

df = load_data()

# --- Sidebar Filters ---
st.sidebar.header("🔍 Filters")
species_options = df["Species"].unique().tolist()
selected_species = st.sidebar.multiselect("Select Species", species_options, default=species_options)

filtered_df = df[df["Species"].isin(selected_species)]

# --- Dataset Overview ---
st.subheader("📋 Dataset Preview")
col1, col2, col3 = st.columns(3)
col1.metric("Total Records", len(filtered_df))
col2.metric("Features", len(df.columns) - 2)
col3.metric("Species Selected", len(selected_species))

st.dataframe(filtered_df.drop(columns=["Id"]), use_container_width=True)

# --- Stats Summary ---
st.subheader("📊 Statistical Summary")
st.dataframe(filtered_df.drop(columns=["Id"]).describe().round(2), use_container_width=True)

# --- Visualizations ---
st.subheader("📈 Visualizations")

tab1, tab2, tab3, tab4 = st.tabs(["Distribution", "Scatter Plot", "Correlation Heatmap", "PCA"])

features = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
palette = {"Iris-setosa": "#4C9BE8", "Iris-versicolor": "#F28C28", "Iris-virginica": "#6DBF6B"}
st.markdown("---")
st.caption("Built with Streamlit 🌿 | Iris Dataset")
