import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Set page config
st.set_page_config(page_title="AI Employee Attrition Predictor", layout="wide", page_icon="🏢")

st.title("🏢 Industry-Level Employee Attrition Predictor")
st.markdown("""
Welcome to the AI-powered HR Dashboard. This system has been trained on historical employee records to predict **who is likely to leave** and provides an AI-driven explanation of **why**, helping you take proactive retention measures.
""")

# Load resources
@st.cache_data
def load_data():
    df = pd.read_csv('employee_data.csv')
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    df['Emp_ID'] = ['EMP' + str(i).zfill(3) for i in range(1, len(df) + 1)]
    df_proc = pd.read_csv('employee_data_processed.csv')
    return df, df_proc

@st.cache_resource
def load_model():
    model = joblib.load('attrition_model.pkl')
    feature_names = joblib.load('feature_names.pkl')
    return model, feature_names

try:
    df, df_proc = load_data()
    model, feature_names = load_model()
except FileNotFoundError:
    st.error("Model or data files not found. Please run the training script first.")
    st.stop()

X = df_proc[feature_names]

# Predict probabilities
probs = model.predict_proba(X)[:, 1]
df['Risk_Score'] = probs * 100

# Sidebar filters
st.sidebar.header("⚙️ Filter Employees")
risk_threshold = st.sidebar.slider("Minimum Attrition Risk (%)", 0, 100, 50)
unique_depts = df['Department'].unique()
dept_filter = st.sidebar.multiselect("Filter by Department", unique_depts, default=unique_depts)

# Apply filters
filtered_df = df[(df['Risk_Score'] >= risk_threshold) & (df['Department'].isin(dept_filter))]
high_risk_df = filtered_df.sort_values(by='Risk_Score', ascending=False)

st.subheader(f"🚨 Employees at Risk of Leaving ({len(high_risk_df)} found)")

if len(high_risk_df) > 0:
    st.dataframe(
        high_risk_df[['Emp_ID', 'Age', 'Department', 'JobRole', 'Risk_Score', 'Attrition']]
        .style.background_gradient(subset=['Risk_Score'], cmap='Reds', vmin=0, vmax=100)
        .format({'Risk_Score': '{:.2f}%'})
    )

    st.divider()
    st.subheader("🧠 Deep Dive: Why are they leaving?")
    
    selected_emp = st.selectbox("Select an Employee to analyze their risk factors", high_risk_df['Emp_ID'].tolist())

    if selected_emp:
        emp_idx = df[df['Emp_ID'] == selected_emp].index[0]
        emp_data_encoded = X.iloc[[emp_idx]]

        col1, col2 = st.columns([1, 2])

        with col1:
            st.info(f"**Employee Profile: {selected_emp}**")
            st.write(f"**Age:** {df.loc[emp_idx, 'Age']}")
            st.write(f"**Department:** {df.loc[emp_idx, 'Department']}")
            st.write(f"**Role:** {df.loc[emp_idx, 'JobRole']}")
            st.write(f"**Monthly Income:** ${df.loc[emp_idx, 'MonthlyIncome']}")
            st.write(f"**Years at Company:** {df.loc[emp_idx, 'YearsAtCompany']}")
            
            is_left = df.loc[emp_idx, 'Attrition'] == 'Yes'
            st.write(f"**Current Status:** {'Left' if is_left else 'Active'}")
            st.metric(label="Predicted Risk of Leaving", value=f"{df.loc[emp_idx, 'Risk_Score']:.1f}%")

        with col2:
            st.warning("**AI Analysis: Key Drivers for this Employee**")
            st.markdown("Factors pushing the employee to leave are in **red** (pointing right). Factors making them stay are in **blue** (pointing left).")
            
            # SHAP Explanation
            explainer = shap.Explainer(model)
            shap_values = explainer(emp_data_encoded)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            shap.plots.waterfall(shap_values[0], show=False)
            plt.tight_layout()
            st.pyplot(fig)
else:
    st.success("No employees found matching the current risk filters. Great job!")
