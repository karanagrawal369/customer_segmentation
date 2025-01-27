import streamlit as st
import pickle
import numpy as np

# Load the pre-trained K-Means model
kmeans = pickle.load(open("k_means.pkl", "rb"))

# Cluster descriptions for better user understanding
cluster_descriptions = {
    0: "Cluster 0: Medium annual income and medium annual spend",
    1: "Cluster 1: High annual income and high annual spend",
    2: "Cluster 2: Low annual income and high annual spend",
    3: "Cluster 3: High annual income but low annual spend",
    4: "Cluster 4: Low annual income and low annual spend",
}

# App title and description
st.set_page_config(page_title="Mall Customer Segmentation", page_icon="📊", layout="centered")
st.title("📊 Mall Customer Segmentation")
st.markdown(
    """
    **Welcome!** This tool predicts customer segments based on their annual income and spending score.
    Use the sliders below to input customer data and discover their cluster group.
    """
)

# Sidebar for user inputs
st.sidebar.header("User Inputs")
annual_income = st.sidebar.slider("Annual Income ($):", 0, 200, 50, step=1)
spending_score = st.sidebar.slider("Spending Score (1-100):", 0, 100, 50, step=1)

# Predict button
if st.sidebar.button("Predict Cluster"):
    # Prepare user data for prediction
    user_data = np.array([[annual_income, spending_score]])
    predicted_cluster = kmeans.predict(user_data)[0]

    # Display prediction results
    st.subheader(f"Predicted Cluster: **{predicted_cluster}**")
    st.markdown(f"**Description:** {cluster_descriptions.get(predicted_cluster, 'Description not available.')}")
    st.balloons()  # Fun visual effect to celebrate predictions!

# Footer
st.markdown(
    """
    ---
    🛍️ **Mall Customer Segmentation Tool**  
    Created by Karan Agrawal
    """
)
