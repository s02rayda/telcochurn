import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
import shap

# --- 1. Page Configuration ---
st.set_page_config(
    page_title="Customer Churn Intelligence Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Caching and Data Loading ---
# Use caching for performance optimization
@st.cache_data
def load_artifacts():
    """Loads all the necessary data and model artifacts."""
    report_df = pd.read_csv('churn_dashboard_data.csv')
    _explainer, _shap_values, _X_test = joblib.load('shap_objects.pkl')
    return report_df, _explainer, _shap_values, _X_test

report_df, explainer, shap_values, X_test = load_artifacts()

# --- 3. Sidebar Navigation ---
st.sidebar.title("Churn Intelligence Dashboard")
st.sidebar.markdown("### Navigation")
page = st.sidebar.radio("Go to", ["Executive Summary", "Key Driver Analysis", "Model Interpretation", "Actionable Insights"])

# --- 4. Main Content ---

if page == "Executive Summary":
    # --- Header ---
    st.title("📈 Executive Summary: Customer Churn Analysis")
    st.markdown("""
    This dashboard provides a comprehensive overview of customer churn, leveraging a machine learning model to identify key drivers and at-risk customers.
    Our model, an XGBoost classifier, demonstrates a strong ability to predict churn, enabling proactive retention strategies.
    """)

    # --- High-Level KPIs ---
    st.markdown("---")
    st.header("Overall Business Health")
    
    col1, col2, col3 = st.columns(3)
    actual_churn_rate = report_df['Actual_Churn'].mean()
    predicted_churn_sum = report_df['Predicted_Churn'].sum()

    col1.metric("Total Customers Analyzed", f"{len(report_df):,}")
    col2.metric("Actual Churn Rate", f"{actual_churn_rate:.1%}")
    col3.metric("High-Risk Customers Flagged", f"{predicted_churn_sum:,}")

    st.markdown("""
    - **Actual Churn Rate:** Represents the percentage of customers in our test dataset who actually churned.
    - **High-Risk Customers Flagged:** The number of customers our model identified as likely to churn, allowing for targeted intervention.
    """)

elif page == "Key Driver Analysis":
    st.title("🔍 Key Driver Analysis: What Factors Influence Churn?")
    st.markdown("Visualizing the relationship between customer attributes and churn.")
    st.markdown("---")

    # --- Visualizations in Columns ---
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Churn Rate by Contract Type")
        fig1, ax1 = plt.subplots()
        sns.barplot(x='Contract', y='Actual_Churn', data=report_df, palette='viridis', ax=ax1, order=['Month-to-month', 'One year', 'Two year'])
        ax1.set_ylabel('Average Churn Rate')
        ax1.set_xlabel('Contract Type')
        st.pyplot(fig1)
        st.info("**Insight:** Customers on **Month-to-month** contracts are significantly more likely to churn. Securing longer-term contracts is a primary retention strategy.")

    with col2:
        st.subheader("Churn Rate by Internet Service")
        fig2, ax2 = plt.subplots()
        sns.barplot(x='InternetService', y='Actual_Churn', data=report_df, palette='plasma', ax=ax2)
        ax2.set_ylabel('Average Churn Rate')
        ax2.set_xlabel('Internet Service')
        st.pyplot(fig2)
        st.info("**Insight:** Customers with **Fiber optic** service exhibit a higher churn rate than those with DSL. This suggests a potential issue with service satisfaction, pricing, or competitor offerings for this premium service.")

elif page == "Model Interpretation":
    st.title("🧠 Model Interpretation: Inside the 'Black Box'")
    st.markdown("""
    Using SHAP (SHapley Additive exPlanations), we can understand exactly how our model makes its decisions. The plot below shows the top features and their impact on predicting churn.
    - **Red dots** represent high feature values.
    - **Blue dots** represent low feature values.
    - **Positive SHAP values** push the prediction towards **Churn**.
    """)
    st.markdown("---")

    # --- SHAP Beeswarm Plot ---
    st.subheader("Detailed Impact of Top Features on Churn Prediction")
    fig3, ax3 = plt.subplots(figsize=(10, 8)) # Make it larger for better readability
    shap.summary_plot(shap_values, X_test, max_display=10, show=False)
    st.pyplot(fig3)
    st.success("""
    **Key Takeaways from the Model's Logic:**
    1.  **Contract is King:** Having a `Two year` or `One year` contract strongly reduces churn risk.
    2.  **Loyalty Pays:** Low `tenure` (new customers) is a major risk factor.
    3.  **Price Matters:** High `MonthlyCharges` consistently push customers towards churning.
    """)

elif page == "Actionable Insights":
    st.title("🎯 Actionable Insights: High-Priority Customer List")
    st.markdown("""
    This table provides a prioritized list of customers that the model has identified as having the highest probability of churning. This is the starting point for a targeted retention campaign.
    """)
    st.markdown("---")

    # --- Top N Customers Table ---
    st.subheader("Top Customers to Contact")
    
    # Allow user to select how many top customers to sees
    top_n = st.slider("Select number of top customers to display:", 5, 50, 10)

    high_risk_customers = report_df.sort_values(by='Churn_Probability', ascending=False).head(top_n)
    report_table = high_risk_customers[[
        'customerID',
        'Churn_Probability',
        'tenure',
        'Contract',
        'InternetService',
        'MonthlyCharges'
    ]]

    st.dataframe(report_table.style.format({'Churn_Probability': '{:.2%}'}), use_container_width=True)
    st.warning("Recommendation: The retention team should prioritize contacting these customers, perhaps offering tailored incentives based on their characteristics (e.g., a discount for high `MonthlyCharges`).")