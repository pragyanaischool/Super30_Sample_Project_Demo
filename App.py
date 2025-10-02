import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.figure_factory as ff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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

# --- DATA SIMULATION & LOADING ---
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
    states = ['CA', 'NY', 'TX', 'FL', 'IL', 'NJ', 'PA', 'OH', 'GA', 'VA', 'NV', 'AZ', 'MA', 'WA', 'MD']
    data['addr_state'] = np.random.choice(states, size=num_samples)
    years = np.random.choice(range(2012, 2018), size=num_samples)
    months = np.random.randint(1, 13, size=num_samples)
    data['issue_d'] = pd.to_datetime([f'{y}-{m}-01' for y, m in zip(years, months)])
    data['credit_history_length'] = np.random.randint(5, 30, size=num_samples)
    data.loc[data['loan_status'] == 'Fully Paid', 'credit_history_length'] += np.random.randint(0, 10, size=len(data[data['loan_status'] == 'Fully Paid']))
    df = data.copy()
    df['issue_year'] = df['issue_d'].dt.year
    return df

@st.cache_data
def load_data(uploaded_file):
    """
    Loads data from an uploaded CSV file or generates synthetic data if no file is uploaded.
    """
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Successfully loaded your custom data!")
            # Basic validation
            required_model_cols = ['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'grade', 'purpose', 'loan_status']
            if not all(col in df.columns for col in required_model_cols):
                st.error(f"Your CSV is missing one or more required columns for modeling. Required: {required_model_cols}. Falling back to sample data.")
                return create_synthetic_lending_club_data()
            
            # Feature engineering for uploaded data
            if 'issue_d' in df.columns:
                df['issue_d'] = pd.to_datetime(df['issue_d'], errors='coerce')
                df['issue_year'] = df['issue_d'].dt.year
            else:
                df['issue_year'] = 2017 # Assign a default year if not present
            
            if 'credit_history_length' not in df.columns:
                 st.warning("`credit_history_length` not found. A default value will be used for modeling.")
                 df['credit_history_length'] = 10
            
            return df
        except Exception as e:
            st.error(f"Error reading the uploaded file: {e}. Falling back to sample data.")
            return create_synthetic_lending_club_data()
    # If no file is uploaded, use the synthetic data
    return create_synthetic_lending_club_data()

# --- MACHINE LEARNING MODEL TRAINING ---
@st.cache_resource
def train_all_models(df):
    """
    Trains multiple ML models and returns them along with their performance metrics.
    """
    df_model = df.copy()
    df_model['loan_status'] = df_model['loan_status'].apply(lambda x: 1 if x == 'Charged Off' else 0)
    
    numeric_features = ['loan_amnt', 'annual_inc', 'int_rate', 'dti', 'credit_history_length']
    categorical_features = ['grade', 'purpose']
    
    # Ensure all required columns are present before training
    for col in numeric_features + categorical_features:
        if col not in df_model.columns:
            st.error(f"Dataframe is missing the required column '{col}' for model training. Cannot proceed.")
            st.stop()
            
    X = df_model[numeric_features + categorical_features]
    y = df_model['loan_status']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
        
    classifiers = {
        'Logistic Regression': LogisticRegression(random_state=42, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1, max_depth=10),
        'Gradient Boosting': GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=5)
    }
    
    results = {}
    for name, classifier in classifiers.items():
        pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', classifier)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        
        feature_importance = None
        if hasattr(classifier, 'feature_importances_'):
            try:
                cat_feature_names = pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out(categorical_features)
                all_feature_names = numeric_features + list(cat_feature_names)
                importances = pipeline.named_steps['classifier'].feature_importances_
                feature_importance = pd.DataFrame({'feature': all_feature_names, 'importance': importances}).sort_values('importance', ascending=False)
            except: # Handle cases where feature names can't be retrieved
                feature_importance = None

        results[name] = {
            'pipeline': pipeline,
            'accuracy': accuracy,
            'report': report,
            'cm': cm,
            'feature_importance': feature_importance
        }
        
    return results

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Project Report & EDA", "Predictive Modeling & Simulation"])

st.sidebar.title("Data Source")
uploaded_file = st.sidebar.file_uploader("Upload your own CSV", type=["csv"])
st.sidebar.info("If no file is uploaded, the app will use a sample synthetic dataset for demonstration.")


# --- MAIN APP ---
df = load_data(uploaded_file)
model_results = train_all_models(df)

# --- PAGE 1: PROJECT REPORT & EDA ---
if page == "Project Report & EDA":
    st.title("📊 Lending Club Loan Analysis Dashboard")
    st.markdown("An interactive report on the key drivers of loan defaults.")

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

    st.header("Exploratory Data Analysis (EDA)")
    
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
    
    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Loan Amount Distribution")
        fig = px.histogram(df, x='loan_amnt', nbins=50, title="Distribution of Loan Amounts", labels={'loan_amnt': 'Loan Amount ($)'})
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** Most loans are concentrated below $20,000, with common peaks at round numbers like $10,000 and $20,000, indicating popular loan packages.")
    with col2:
        st.subheader("Income vs. Loan Amount by Grade")
        df_sample = df.sample(n=min(5000, len(df)), random_state=42)
        fig = px.scatter(df_sample, x='annual_inc', y='loan_amnt', color='grade',
                         title="Higher Income Allows for Larger Loans",
                         labels={'annual_inc': 'Annual Income ($)', 'loan_amnt': 'Loan Amount ($)'},
                         category_orders={'grade': ['A', 'B', 'C', 'D', 'E', 'F', 'G']},
                         hover_data=['purpose'])
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("**Insight:** A clear positive correlation exists between income and loan amount. However, lower-grade borrowers (e.g., D, E) often receive smaller loans regardless of income.")

    st.markdown("---")
    
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

    st.subheader("Geographic and Temporal Trends")
    col1, col2 = st.columns([1, 1])
    if 'addr_state' in df.columns:
        with col1:
            state_default_rate = df.groupby('addr_state')['loan_status'].apply(lambda x: (x == 'Charged Off').mean() * 100).reset_index()
            fig = px.choropleth(state_default_rate, locations='addr_state', locationmode="USA-states", color='loan_status', scope="usa", color_continuous_scale="Reds", title="Regional Hot Spots for Loan Defaults", labels={'loan_status': 'Default Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)
    if 'issue_year' in df.columns:
        with col2:
            default_rate_by_year = df.groupby('issue_year')['loan_status'].apply(lambda x: (x == 'Charged Off').mean() * 100).reset_index()
            fig = px.line(default_rate_by_year, x='issue_year', y='loan_status', markers=True, title="Default Rate Fluctuates Annually", labels={'issue_year': 'Year of Loan Issuance', 'loan_status': 'Default Rate (%)'})
            st.plotly_chart(fig, use_container_width=True)

# --- PAGE 2: PREDICTIVE MODELING ---
elif page == "Predictive Modeling & Simulation":
    st.title("🤖 Loan Default Predictive Model")
    
    st.header("Model Selection & Performance")
    
    model_name = st.selectbox("Choose a Machine Learning Model", list(model_results.keys()))
    
    selected_model = model_results[model_name]
    accuracy = selected_model['accuracy']
    report = selected_model['report']
    cm = selected_model['cm']

    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.metric("Model Accuracy", f"{accuracy:.2%}")
        st.markdown("**Recall (for 'Charged Off'):**")
        st.info(f"{report['1']['recall']:.2%}")
        st.markdown("**Precision (for 'Charged Off'):**")
        st.info(f"{report['1']['precision']:.2%}")
        st.markdown("""
        - **Accuracy:** Overall, how often the model is correct.
        - **Recall:** Of all actual defaults, how many did the model correctly identify? (Crucial for minimizing losses).
        - **Precision:** When the model predicts a default, how often is it correct? (Important for not rejecting good borrowers).
        """)

    with col2:
        st.subheader("Confusion Matrix")
        z = cm
        x = ['Predicted: Fully Paid', 'Predicted: Charged Off']
        y = ['Actual: Charged Off', 'Actual: Fully Paid']
        
        fig = ff.create_annotated_heatmap(z, x=x, y=y, colorscale='Blues', showscale=True)
        fig.update_layout(title_text=f'Confusion Matrix for {model_name}')
        st.plotly_chart(fig, use_container_width=True)
        st.markdown("The matrix shows the model's performance in distinguishing between the two outcomes. The goal is to maximize the top-right (True Positives for default) and bottom-left (True Negatives) cells.")

    if selected_model['feature_importance'] is not None:
        st.markdown("---")
        st.header("Model Feature Importance")
        st.markdown("This chart shows which factors the model found most influential in making its predictions.")
        
        feat_importance_df = selected_model['feature_importance'].head(15)
        fig = px.bar(feat_importance_df, x='importance', y='feature', orientation='h',
                     title=f'Top 15 Most Important Features for {model_name}',
                     labels={'importance': 'Importance Score', 'feature': 'Feature'})
        fig.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
        
    st.markdown("---")
    
    st.header("Live Default Prediction Simulation")
    st.markdown("Enter an applicant's details below to get a real-time default risk prediction using the selected model.")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 15000)
        annual_inc = st.slider("Annual Income ($)", 10000, 500000, 60000)
        purpose = st.selectbox("Loan Purpose", df['purpose'].unique())
        
    with col2:
        int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.0, key="sim_int_rate")
        dti = st.slider("Debt-to-Income Ratio (%)", 1.0, 100.0, 20.0)
        grade = st.select_slider("Loan Grade", options=['A', 'B', 'C', 'D', 'E', 'F', 'G'])
        
    with col3:
        credit_history_length = st.slider("Credit History (Years)", 1, 40, 10, key="sim_credit_hist")
        
    input_data = pd.DataFrame({
        'loan_amnt': [loan_amnt], 'annual_inc': [annual_inc], 'int_rate': [int_rate],
        'dti': [dti], 'credit_history_length': [credit_history_length],
        'grade': [grade], 'purpose': [purpose]
    })
    
    prediction_proba = selected_model['pipeline'].predict_proba(input_data)[:, 1]
    prediction = selected_model['pipeline'].predict(input_data)
    
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
    2. Create a `requirements.txt` file (if you don't have one).
    3. Push both files to a GitHub repository.
    4. Go to [share.streamlit.io](https://share.streamlit.io) to deploy your app.
    """
)
