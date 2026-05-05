"""
Streamlit UI for Customer Churn Pipeline
Run with: streamlit run streamlit_app.py
Features: CSV Upload, Interactive Charts, ML Training, Batch Analysis, Export
"""
import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import ensure_directories, PATHS
from src.utils import load_or_create_clean_data, get_data_summary
from src.agents import OrchestratorAgent
from src.models import ChurnModelTrainer, ChurnPredictor

# Page config
st.set_page_config(
    page_title="Customer Churn Analyzer",
    page_icon="📉",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize
ensure_directories()

@st.cache_data
def load_data():
    """Cache data loading."""
    return load_or_create_clean_data()

@st.cache_resource
def get_orchestrator():
    """Cache orchestrator."""
    return OrchestratorAgent()

# ============================================================
# SIDEBAR & NAVIGATION
# ============================================================
with st.sidebar:
    st.header("📉 Churn Analyzer")
    
    # Mode selection
    page = st.radio(
        "Navigation",
        ["📊 Dashboard", "👤 Single Customer", "📦 Batch Analysis", "🤖 ML Model", "📁 Upload CSV"],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Quick stats
    try:
        df = load_data()
        summary = get_data_summary(df)
        st.metric("Total Customers", f"{summary['total_rows']:,}")
        st.metric("Churn Rate", f"{summary['churn_rate']}%")
    except Exception as e:
        st.warning(f"Data not loaded: {e}")

    st.markdown("---")
    st.caption("Built with Agentic AI Pipeline")

# ============================================================
# PAGE 1: DASHBOARD
# ============================================================
if page == "📊 Dashboard":
    st.title("📊 Dataset Overview")
    st.markdown("Comprehensive analysis of customer churn patterns")
    
    try:
        df = load_data()
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()
    
    # KPI Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Customers", f"{len(df):,}")
    col2.metric("Churned", f"{int(df['Churn'].sum()):,}")
    col3.metric("Churn Rate", f"{df['Churn'].mean():.1%}")
    col4.metric("Avg Tenure", f"{df['Tenure'].mean():.0f} mo")
    
    st.markdown("---")
    
    # Charts Row 1
    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "🔍 Correlations", "📊 Risk Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Churn Distribution")
            churn_counts = df["Churn"].value_counts()
            fig = px.pie(
                values=churn_counts.values,
                names=["Retained", "Churned"],
                color_discrete_map={"Retained": "#4CAF50", "Churned": "#F44336"},
                hole=0.4
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Age Distribution")
            fig = px.histogram(df, x="Age", color="Churn", nbins=30, barmode="overlay", opacity=0.7)
            fig.update_layout(showlegend=True)
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Support Calls vs Churn")
            fig = px.box(df, x="Churn", y="Support Calls", color="Churn", points="outliers")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.subheader("Payment Delay vs Churn")
            fig = px.box(df, x="Churn", y="Payment Delay", color="Churn", points="outliers")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Feature Correlation Heatmap")
        numeric_df = df.select_dtypes(include=["number"])
        corr_matrix = numeric_df.corr()
        fig = px.imshow(
            corr_matrix,
            text_auto=True,
            color_continuous_scale="RdBu_r",
            zmin=-1, zmax=1,
            aspect="auto"
        )
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Quick Risk Analysis (Sample)")
        sample_size = st.slider("Sample Size", 100, 5000, 1000)
        
        if st.button("Run AI Agent Analysis"):
            with st.spinner("Analyzing customers..."):
                df_sample = df.head(sample_size)
                orch = get_orchestrator()
                result = orch.run(df_sample, sample_size=sample_size)
                
                st.success(f"✅ Analyzed {sample_size} customers")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Risk Distribution")
                    dist_df = pd.DataFrame(
                        list(result.risk_distribution.items()), 
                        columns=["Level", "Count"]
                    )
                    dist_df["Percentage"] = (dist_df["Count"] / result.total_customers * 100).round(1)
                    
                    fig = px.bar(
                        dist_df, x="Level", y="Count", color="Level",
                        color_discrete_map={"critical": "#F44336", "high": "#FF9800", "medium": "#FFEB3B", "low": "#8BC34A", "none": "#4CAF50"},
                        text="Percentage"
                    )
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dist_df, use_container_width=True)
                
                with col2:
                    st.subheader("Top 10 At-Risk Customers")
                    top_risk = sorted(result.results, key=lambda x: x.risk_score, reverse=True)[:10]
                    
                    risk_data = []
                    for r in top_risk:
                        risk_data.append({
                            "Customer ID": r.customer_id,
                            "Risk Level": r.risk_level.upper(),
                            "Score": r.risk_score,
                            "Churn Prob": f"{r.churn_probability:.1%}",
                            "Signals": r.signal_count
                        })
                    st.dataframe(pd.DataFrame(risk_data), use_container_width=True)
    
    # Feature summary table
    st.markdown("---")
    st.subheader("📋 Feature Summary")
    st.dataframe(df.describe().round(2), use_container_width=True)

# ============================================================
# PAGE 2: SINGLE CUSTOMER ANALYSIS
# ============================================================
elif page == "👤 Single Customer":
    st.title("👤 Single Customer Analysis")
    st.markdown("Enter customer details to get AI-powered churn prediction")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        age = st.number_input("Age", 18, 80, 35)
        gender = st.selectbox("Gender", ["Male", "Female"])
        tenure = st.number_input("Tenure (months)", 1, 60, 12)
    
    with col2:
        usage = st.number_input("Usage Frequency (days/month)", 1, 30, 15)
        support = st.number_input("Support Calls", 0, 20, 5)
        payment = st.number_input("Payment Delay (days)", 0, 30, 15)
    
    with col3:
        subscription = st.selectbox("Subscription Type", ["Basic", "Standard", "Premium"])
        contract = st.selectbox("Contract Length", ["Monthly", "Quarterly", "Annual"])
        spend = st.number_input("Total Spend ($)", 100, 1000, 500)
        interaction = st.number_input("Last Interaction (days ago)", 0, 30, 14)
    
    customer_data = {
        "Age": age,
        "Gender": gender,
        "Tenure": tenure,
        "Usage Frequency": usage,
        "Support Calls": support,
        "Payment Delay": payment,
        "Subscription Type": subscription,
        "Contract Length": contract,
        "Total Spend": spend,
        "Last Interaction": interaction,
        "CustomerID": 0
    }
    
    if st.button("🔍 Analyze Customer", type="primary", use_container_width=True):
        with st.spinner("Running AI Agent Analysis..."):
            orch = get_orchestrator()
            result = orch.run_single_customer(customer_data)
            
            assessment = result["assessment"]
            signals = result["signals"]
            actions = result["actions"]
            
            # Risk Gauge
            st.subheader("📊 Risk Assessment")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=assessment.risk_score,
                    title={"text": "Risk Score"},
                    gauge={
                        "axis": {"range": [0, 100]},
                        "bar": {"color": "darkblue"},
                        "steps": [
                            {"range": [0, 25], "color": "#4CAF50"},
                            {"range": [25, 50], "color": "#FFEB3B"},
                            {"range": [50, 75], "color": "#FF9800"},
                            {"range": [75, 100], "color": "#F44336"}
                        ],
                    }
                ))
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.metric("Risk Level", assessment.risk_level.upper())
                st.metric("Churn Probability", f"{assessment.churn_probability:.1%}")
                
                if assessment.risk_level in ["critical", "high"]:
                    st.error(f"🚨 {assessment.risk_level.upper()} risk of churning!")
                elif assessment.risk_level == "medium":
                    st.warning("⚠️ MEDIUM risk of churning")
                else:
                    st.success("✅ Low risk")
            
            with col3:
                if signals:
                    st.subheader("⚠️ Detected Signals")
                    for sig in signals:
                        icon = "🔴" if sig.severity == "CRITICAL" else "🟡" if sig.severity == "HIGH" else "🟢"
                        st.write(f"{icon} **{sig.name}**: {sig.description}")
            
            # Recommended Actions
            if actions:
                st.subheader("💡 Recommended Actions")
                for action in actions:
                    priority_icon = "🚨" if action.priority == "CRITICAL" else "⚠️" if action.priority == "HIGH" else "ℹ️"
                    with st.expander(f"{priority_icon} {action.action_type}"):
                        st.write(f"**Description:** {action.description}")
                        st.write(f"**Expected Impact:** {action.expected_impact}")
                        st.write(f"**Priority:** {action.priority}")

# ============================================================
# PAGE 3: BATCH ANALYSIS
# ============================================================
elif page == "📦 Batch Analysis":
    st.title("📦 Batch Customer Analysis")
    
    col1, col2 = st.columns(2)
    sample_size = col1.slider("Number of Customers", 100, 10000, 1000)
    use_llm = col2.checkbox("Enable LLM Enhancement (slower)")
    
    if st.button("Run Batch Analysis", type="primary", use_container_width=True):
        with st.spinner(f"Analyzing {sample_size} customers..."):
            df_sample = load_data().head(sample_size)
            orch = get_orchestrator()
            result = orch.run(df_sample, sample_size=sample_size, use_llm_enhancement=use_llm)
            
            st.success(f"✅ Analysis Complete!")
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total Analyzed", result.total_customers)
            col2.metric("High Risk", result.high_risk_count)
            col3.metric("Critical", result.critical_risk_count)
            col4.metric("Avg Risk Score", f"{sum(r.risk_score for r in result.results)/len(result.results):.0f}")
            
            st.markdown("---")
            
            # Visualization
            tab1, tab2 = st.tabs(["📊 Risk Distribution", "📋 Customer List"])
            
            with tab1:
                dist_df = pd.DataFrame(
                    list(result.risk_distribution.items()), 
                    columns=["Level", "Count"]
                )
                dist_df["Percentage"] = (dist_df["Count"] / result.total_customers * 100).round(1)
                
                col1, col2 = st.columns(2)
                with col1:
                    fig = px.pie(dist_df, values="Count", names="Level", hole=0.4)
                    fig.update_traces(textposition='inside', textinfo='percent+label')
                    st.plotly_chart(fig, use_container_width=True)
                with col2:
                    fig = px.bar(dist_df, x="Level", y="Count", color="Level", text="Percentage")
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(dist_df, use_container_width=True)
            
            with tab2:
                st.subheader("Top 20 At-Risk Customers")
                top_risk = sorted(result.results, key=lambda x: x.risk_score, reverse=True)[:20]
                
                risk_data = []
                for r in top_risk:
                    signals_desc = "; ".join([sig.description for sig in r.signals[:3]]) if r.signals else "None"
                    risk_data.append({
                        "Customer ID": r.customer_id,
                        "Risk Level": r.risk_level.upper(),
                        "Score": r.risk_score,
                        "Churn Prob": f"{r.churn_probability:.1%}",
                        "Signals": r.signal_count,
                        "Key Signals": signals_desc
                    })
                
                risk_df = pd.DataFrame(risk_data)
                st.dataframe(risk_df, use_container_width=True)
                
                # Export
                csv = risk_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Results CSV",
                    data=csv,
                    file_name="churn_analysis_results.csv",
                    mime="text/csv"
                )

# ============================================================
# PAGE 4: ML MODEL
# ============================================================
elif page == "🤖 ML Model":
    st.title("🤖 ML Model Training & Prediction")
    
    model_path = PATHS["models"] / "best_model.pkl"
    
    tab1, tab2 = st.tabs(["🔮 Prediction", "📊 Model Training"])
    
    with tab1:
        if model_path.exists():
            st.success("✅ Model loaded successfully")
            
            with st.form("prediction_form"):
                col1, col2 = st.columns(2)
                with col1:
                    age = st.number_input("Age", 18, 80, 35)
                    tenure = st.number_input("Tenure", 1, 60, 12)
                    usage = st.number_input("Usage Frequency", 1, 30, 15)
                    support = st.number_input("Support Calls", 0, 20, 5)
                    payment = st.number_input("Payment Delay", 0, 30, 15)
                
                with col2:
                    subscription = st.selectbox("Subscription", ["Basic", "Standard", "Premium"])
                    contract = st.selectbox("Contract", ["Monthly", "Quarterly", "Annual"])
                    spend = st.number_input("Total Spend", 100, 1000, 500)
                    interaction = st.number_input("Last Interaction", 0, 30, 14)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                
                submitted = st.form_submit_button("🔮 Predict", type="primary", use_container_width=True)
                
                if submitted:
                    customer = {
                        "Age": age, "Gender": gender, "Tenure": tenure,
                        "Usage Frequency": usage, "Support Calls": support,
                        "Payment Delay": payment, "Subscription Type": subscription,
                        "Contract Length": contract, "Total Spend": spend,
                        "Last Interaction": interaction,
                    }
                    
                    predictor = ChurnPredictor(model_path)
                    result = predictor.predict(customer)
                    
                    if result["churn_prediction"] == 1:
                        st.error(f"🚨 Customer will CHURN!")
                        st.metric("Churn Probability", f"{result['churn_probability']:.2%}")
                        
                        fig = go.Figure(go.Indicator(
                            mode="gauge+number",
                            value=result["churn_probability"] * 100,
                            title={"text": "Churn Probability"},
                            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "red"},
                                   "steps": [{"range": [0, 50], "color": "lightgreen"}, {"range": [50, 100], "color": "salmon"}]}
                        ))
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.success(f"✅ Customer will stay")
                        st.metric("Retention Probability", f"{result['retention_probability']:.2%}")
        else:
            st.warning("⚠️ No trained model found. Train a model in the 'Model Training' tab.")
    
    with tab2:
        st.subheader("Train New Model")
        st.markdown("This will train Logistic Regression, Random Forest, and Gradient Boosting models.")
        
        col1, col2 = st.columns(2)
        train_size = col1.slider("Training Sample Size", 1000, 50000, 10000)
        
        if col2.button("🚀 Train Model", type="primary"):
            with st.spinner("Training models..."):
                df_sample = load_data().sample(min(train_size, len(load_data())), random_state=42)
                sample_path = PROJECT_ROOT / "data" / "processed" / "train_sample.csv"
                df_sample.to_csv(sample_path, index=False)
                
                trainer = ChurnModelTrainer(sample_path, PATHS["models"])
                
                progress_bar = st.progress(0)
                
                # Step 1: Preprocess
                st.write("📦 Preprocessing data...")
                trainer.preprocess()
                progress_bar.progress(25)
                
                # Step 2: Train
                st.write("🏋️ Training models...")
                results = trainer.train_models()
                progress_bar.progress(75)
                
                # Step 3: Select best
                st.write("🏆 Selecting best model...")
                trainer.select_best_model()
                model_file = trainer.save_model("best_model.pkl")
                progress_bar.progress(100)
                
                st.success(f"✅ Model saved to: {model_file}")
                
                # Show results table
                st.subheader("Model Comparison")
                comparison_data = []
                for name, metrics in results.items():
                    comparison_data.append({
                        "Model": name,
                        "Accuracy": f"{metrics['accuracy']:.4f}",
                        "Precision": f"{metrics['precision']:.4f}",
                        "Recall": f"{metrics['recall']:.4f}",
                        "F1 Score": f"{metrics['f1']:.4f}",
                        "AUC-ROC": f"{metrics['auc']:.4f}"
                    })
                
                st.dataframe(pd.DataFrame(comparison_data), use_container_width=True)

# ============================================================
# PAGE 5: UPLOAD CSV
# ============================================================
elif page == "📁 Upload CSV":
    st.title("📁 Upload Custom Dataset")
    st.markdown("Upload your own customer data for analysis")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df_upload):,} rows, {len(df_upload.columns)} columns")
            
            st.subheader("Data Preview")
            st.dataframe(df_upload.head(10), use_container_width=True)
            
            st.markdown("---")
            
            # Auto-detect columns
            st.subheader("Column Mapping")
            col_names = df_upload.columns.tolist()
            
            mapping = {
                "CustomerID": st.selectbox("Customer ID", col_names, index=0 if "CustomerID" in col_names else 0),
                "Churn": st.selectbox("Churn Label", col_names, index=col_names.index("Churn") if "Churn" in col_names else 0),
            }
            
            if st.button("Analyze Uploaded Data"):
                with st.spinner("Analyzing..."):
                    # Quick stats
                    col1, col2, col3 = st.columns(3)
                    
                    if "Churn" in df_upload.columns:
                        churn_col = df_upload["Churn"]
                        col1.metric("Churn Rate", f"{churn_col.mean():.1%}")
                        col2.metric("Churned", f"{int(churn_col.sum()):,}")
                        col3.metric("Retained", f"{int(len(churn_col) - churn_col.sum()):,}")
                        
                        # Churn pie chart
                        fig = px.pie(
                            names=["Retained", "Churned"],
                            values=[int(len(churn_col) - churn_col.sum()), int(churn_col.sum())]
                        )
                        st.plotly_chart(fig)
                    else:
                        st.warning("⚠️ No 'Churn' column found. Showing basic stats only.")
                        col1.metric("Total Rows", f"{len(df_upload):,}")
                        col2.metric("Columns", f"{len(df_upload.columns)}")
                    
                    # Full data table
                    st.subheader("Complete Dataset")
                    st.dataframe(df_upload, use_container_width=True)
                    
                    # Download processed
                    csv = df_upload.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Dataset",
                        data=csv,
                        file_name="processed_data.csv",
                        mime="text/csv"
                    )
        except Exception as e:
            st.error(f"❌ Error reading file: {e}")
    else:
        st.info("👆 Upload a CSV file to get started")
        
        # Template download
        template_data = {
            "CustomerID": [1001, 1002, 1003],
            "Age": [35, 42, 28],
            "Gender": ["Male", "Female", "Male"],
            "Tenure": [12, 24, 6],
            "Usage Frequency": [15, 20, 8],
            "Support Calls": [5, 2, 10],
            "Payment Delay": [15, 5, 25],
            "Subscription Type": ["Standard", "Premium", "Basic"],
            "Contract Length": ["Monthly", "Annual", "Monthly"],
            "Total Spend": [500, 800, 200],
            "Last Interaction": [14, 5, 28],
            "Churn": [1, 0, 1]
        }
        template_csv = pd.DataFrame(template_data).to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Template",
            data=template_csv,
            file_name="churn_template.csv",
            mime="text/csv"
        )

# Footer
st.markdown("---")
st.caption("Customer Churn Pipeline | Built with Agentic AI")
