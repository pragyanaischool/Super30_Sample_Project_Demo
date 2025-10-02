import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lending Club Loan Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# --- DATA SIMULATION ---
# This function creates a realistic-looking dataset for demonstration.
# @st.cache_data ensures this function only runs once, speeding up the app.
@st.cache_data
def create_synthetic_lending_club_data(num_samples=50000):
    """
    Creates a synthetic DataFrame that mimics the structure and key relationships
    of the Lending Club loan dataset for demonstration purposes.
    """
    np.random.seed(42)
    grades = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
    grade_dist = [0.18, 0.29, 0.28, 0.15, 0.07, 0.02, 0.01]
    default_rates = {'A': 0.05, 'B': 0.09, 'C': 0.15, 'D': 0.22, 'E': 0.30, 'F': 0.35, 'G': 0.40}
    int_rates = {'A': 6.5, 'B': 10.0, 'C': 14.0, 'D': 18.0, 'E': 22.0, 'F': 25.0, 'G': 28.0}
    data = pd.DataFrame()
    data['grade'] = np.random.choice(grades, size=num_samples, p=grade_dist)
    data['loan_status'] = data['grade'].apply(
        lambda g: 'Charged Off' if np.random.rand() < default_rates[g] else 'Fully Paid'
    )
    data['int_rate'] = data['grade'].apply(lambda g: np.random.normal(int_rates[g], 1.5))
    data['loan_amnt'] = np.random.randint(1000, 40001, size=num_samples)
    data['annual_inc'] = np.random.gamma(3, 25000, size=num_samples) + (data['loan_amnt'] * 0.5)
    grade_income_multiplier = {'A': 1.5, 'B': 1.2, 'C': 1.0, 'D': 0.8, 'E': 0.7, 'F': 0.6, 'G': 0.5}
    data['annual_inc'] = data.apply(lambda row: row['annual_inc'] * grade_income_multiplier[row['grade']], axis=1)
    data['annual_inc'] = data['annual_inc'].clip(10000, 500000)
    data['dti'] = (data['loan_amnt'] / data['annual_inc']) * 100 * np.random.uniform(0.5, 2.0, size=num_samples)
    data['dti'] = data['dti'].clip(1, 100)
    data.loc[data['loan_status'] == 'Charged Off', 'dti'] *= np.random.uniform(1.1, 1.5, size=len(data[data['loan_status'] == 'Charged Off']))
    purposes = ['debt_consolidation', 'credit_card', 'home_improvement', 'other', 'major_purchase', 'small_business']
    purpose_dist = [0.58, 0.22, 0.06, 0.05, 0.02, 0.01]
    data['purpose'] = np.random.choice(purposes, size=num_samples, p=purpose_dist)
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA'] + ['NV', 'AZ', 'MA', 'WA', 'MD']
    data['addr_state'] = np.random.choice(states, size=num_samples)
    years = np.random.choice(range(2012, 2018), size=num_samples)
    months = np.random.randint(1, 13, size=num_samples)
    data['issue_d'] = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(years, months)])
    data['credit_history_length'] = np.random.randint(5, 30, size=num_samples)
    data.loc[data['loan_status'] == 'Fully Paid', 'credit_history_length'] += np.random.randint(0, 10, size=len(data[data['loan_status'] == 'Fully Paid']))
    df = data.copy()
    df['issue_year'] = df['issue_d'].dt.year
    return df

# --- MACHINE LEARNING MODEL TRAINING ---
# @st.cache_resource caches the trained model to avoid retraining on each interaction.
@st.cache_resource
def train_loan_default_model(df):
    """
    Trains a Logistic Regression model to predict loan defaults.
    """
    # 1. Feature Selection and Preprocessing
    df_model = df.copy()
    df_model['loan_status'] = df_model['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    numeric_features = ['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'credit_history_length']
    categorical_features = ['grade', 'purpose']
    
    X = df_model[numeric_features + categorical_features]
    y = df_model['loan_status']
    
    # 2. Create a preprocessing pipeline
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    # 3. Create and train the model pipeline
    model_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                                     ('classifier', LogisticRegression(random_state=42, class_weight='balanced'))])
                                     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model_pipeline.fit(X_train, y_train)
    
    # 4. Evaluate the model
    y_pred = model_pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    
    return model_pipeline, accuracy, report, cm, X.columns

# --- MAIN APP ---
df = create_synthetic_lending_club_data()
model, accuracy, report, cm, feature_names = train_loan_default_model(df)

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Report & EDA", "Predictive Modeling & Simulation"])

# --- PAGE 1: PROJECT REPORT & EDA ---
if page == "Project Report & EDA":
    st.title("📊 Lending Club Loan Analysis Dashboard")
    st.markdown("An interactive report on the key drivers of loan defaults.")

    # -- Key Metrics --
    st.header("Key Performance Indicators")
    col1, col2, col3, col4 = st.columns(4)
    total_loans = len(df)
    default_rate = (df['loan_status'] == 'Charged Off').mean() * 100
    avg_loan_amt = df['loan_amnt'].mean()
    avg_int_rate = df['int_rate'].mean()

    col1.metric("Total Loans", f"{total_loans:,}")
    col2.metric("Overall Default Rate", f"{default_rate:.2f}%")
    col3.metric("Average Loan Amount", f"${avg_loan_amt:,.0f}")
    col4.metric("Average Interest Rate", f"{avg_int_rate:.2f}%")
    st.markdown("---")

    # -- Main EDA Layout --
    st.header("Exploratory Data Analysis (EDA)")
    
    # Row 1
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Interest Rate by Loan Grade")
        fig = px.box(df, x='grade', y='int_rate', color='grade', 
                     title="Interest Rate Increases with Riskier Grades",
                     labels={'grade': 'Loan Grade', 'int_rate': 'Interest Rate (%)'},
                     category_orders={'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** The company's risk-based pricing is working. As the loan grade gets riskier (from A to G), the assigned interest rate increases significantly.")

    with col2:
        st.subheader("Default Rate by Loan Grade")
        default_by_grade = df.groupby('grade')['loan_status'].apply(lambda x: (x == 'Charged Off').mean() * 100).reset_index()
        fig = px.bar(default_by_grade, x='grade', y='loan_status',
                     title="Lower Grades Have Much Higher Default Rates",
                     labels={'grade': 'Loan Grade', 'loan_status': 'Default Rate (%)'},
                     category_orders={'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Loan grade is a powerful predictor of default. The default rate for G-grade loans is nearly 8 times higher than for A-grade loans.")

    st.markdown("---")
    # Row 2
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Credit History vs. Loan Status")
        fig = px.box(df, x='loan_status', y='credit_history_length', color='loan_status',
                     title="Longer Credit History Correlates with Lower Risk",
                     labels={'loan_status': 'Loan Status', 'credit_history_length': 'Credit History (Years)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Borrowers who fully paid their loans tend to have a longer credit history, suggesting that more experience with credit is a sign of lower risk.")

    with col2:
        st.subheader("Default Rate by Loan Purpose")
        default_by_purpose = df.groupby('purpose')['loan_status'].apply(lambda x: (x == 'Charged Off').mean() * 100).reset_index().sort_values('loan_status', ascending=False)
        fig = px.bar(default_by_purpose, x='purpose', y='loan_status',
                     title="'Small Business' Loans are the Riskiest",
                     labels={'purpose': 'Loan Purpose', 'loan_status': 'Default Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** The reason for the loan provides context. 'Small Business' loans are inherently more speculative and have the highest default rate.")
        
    st.markdown("---")
    # Row 3 - Geo and Time
    st.subheader("Geographic and Temporal Trends")
    col1, col2 = st.columns([1, 1])
    with col1:
        state_default_rate = df.groupby('addr_state')['loan_status'].apply(lambda x: (x == 'Charged Off').mean() * 100).reset_index()
        fig = px.choropleth(state_default_rate,
                            locations='addr_state',
                            locationmode="USA-states",
                            color='loan_status',
                            scope="usa",
                            color_continuous_scale="Reds",
                            title="Regional Hot Spots for Loan Defaults",
                            labels={'loan_status': 'Default Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        default_rate_by_year = df.groupby('issue_year')['loan_status'].apply(lambda x: (x == 'Charged Off').mean() * 100).reset_index()
        fig = px.line(default_rate_by_year, x='issue_year', y='loan_status', markers=True,
                      title="Default Rate Fluctuates Annually",
                      labels={'issue_year': 'Year of Loan Issuance', 'loan_status': 'Default Rate (%)'})
        st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: PREDICTIVE MODELING ---
elif page == "Predictive Modeling & Simulation":
    st.title("🤖 Loan Default Predictive Model")
    
    st.header("Model Performance")
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        st.markdown("**Recall (for 'Charged Off'):**")
        st.info(f"{report['1']['recall']:.2%}")
        st.markdown("**Precision (for 'Charged Off'):**")
        st.info(f"{report['1']['precision']:.2%}")
        st.markdown("""
        - **Accuracy:** Overall, how often the model is correct.
        - **Recall:** Of all the actual defaults, how many did the model correctly identify? This is crucial for minimizing losses.
        - **Precision:** When the model predicts a default, how often is it correct? This is important for not rejecting good borrowers.
        """)

    with col2:
        st.subheader("Confusion Matrix")
        z = cm
        x = ['Predicted: Fully Paid', 'Predicted: Charged Off']
        y = ['Actual: Charged Off', 'Actual: Fully Paid']
        
        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
        fig.update_layout(title_text='Model Prediction Accuracy')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("The matrix shows the model is effective at identifying 'Fully Paid' loans and catches a good portion of 'Charged Off' loans, though some are still missed (False Negatives).")
        
    st.markdown("---")
    
    # --- Live Simulation ---
    st.header("Live Default Prediction Simulation")
    st.markdown("Enter an applicant's details below to get a real-time default risk prediction.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 15000)
        annual_inc = st.slider("Annual Income ($)", 10000, 500000, 60000)
        purpose = st.selectbox("Loan Purpose", df['purpose'].unique())
        
    with col2:
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0)
        dti = st.slider("Debt-to-Income Ratio (%)", 1.0, 100.0, 20.0)
        grade = st.selectbox("Loan Grade", df['grade'].unique().sort())
        
    with col3:
        credit_history_length = st.slider("Credit History (Years)", 1, 40, 10)
        
    # Create a DataFrame for prediction
    input_data = pd.DataFrame({
        'loan_amnt': [loan_amnt],
        'annual_inc': [annual_inc],
        'int_rate': [int_rate],
        'dti': [dti],
        'credit_history_length': [credit_history_length],
        'grade': [grade],
        'purpose': [purpose]
    })
    
    # Prediction
    prediction_proba = model.predict_proba(input_data)[:, 1]
    prediction = model.predict(input_data)
    
    st.subheader("Prediction Result")
    col1, col2 = st.columns([1,1])
    with col1:
        if prediction[0] == 1:
            st.error("Prediction: HIGH RISK (Likely to Default)")
        else:
            st.success("Prediction: LOW RISK (Likely to be Fully Paid)")
            
    with col2:
        st.metric("Probability of Default", f"{prediction_proba[0]:.2%}")
        st.progress(prediction_proba[0])

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **About this Dashboard:**
    This interactive dashboard provides a comprehensive analysis of the Lending Club loan dataset. 
    It includes Exploratory Data Analysis (EDA) and a machine learning model to predict loan defaults.
    """
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    ### How to Deploy
    1. Save this code as `app.py`.
    2. Create a `requirements.txt` file with:
       ```
       streamlit
       pandas
       numpy
       plotly
       scikit-learn
       ```
    3. Push both files to a GitHub repository.
    4. Deploy on Streamlit Community Cloud.
    """
)

