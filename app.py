import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import streamlit as st
from sklearn.metrics import accuracy_score,log_loss,classification_report,roc_auc_score, roc_curve,confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


# LOAD PIPELINE
loaded_pipeline = joblib.load("telecom_churn_pipeline.pkl")

# Access each component
model = loaded_pipeline["model"]
scaler = loaded_pipeline["scaler"]
trained_features = loaded_pipeline["trained_features"]
benchmarks = loaded_pipeline["benchmark_metrics"]

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Telecom Churn Prediction",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# HEADER
st.title("📊 Telecom Customer Churn Prediction Dashboard")
st.caption(
    "Predict churn probability, identify customer risk, and improve retention strategy."
)


# CUSTOMER PREDICTION PAGE
def customer_page():

    st.subheader("👤 Customer Information Input")

    col1, col2, col3 = st.columns(3)

    # ---------------- COLUMN 1 ----------------
    with col1:
        account_length = st.number_input(
            "Account Length",
            min_value=0,
            max_value=300,
            value=100
        )

        voice_plan = st.selectbox(
            "Voice Plan",
            ['Yes', 'No']
        )

        voice_messages = st.number_input(
            "Voice Messages",
            min_value=0,
            max_value=100,
            value=10
        )

        intl_plan = st.selectbox(
            "International Plan",
            ['Yes', 'No']
        )

        intl_mins = st.number_input(
            "International Minutes",
            min_value=0.0,
            max_value=50.0,
            value=10.0
        )

    # ---------------- COLUMN 2 ----------------
    with col2:
        intl_calls = st.number_input(
            "International Calls",
            min_value=0,
            max_value=30,
            value=5
        )

        day_mins = st.number_input(
            "Day Minutes",
            min_value=0.0,
            max_value=400.0,
            value=180.0
        )

        day_calls = st.number_input(
            "Day Calls",
            min_value=0,
            max_value=200,
            value=100
        )

        eve_mins = st.number_input(
            "Evening Minutes",
            min_value=0.0,
            max_value=400.0,
            value=200.0
        )

        eve_calls = st.number_input(
            "Evening Calls",
            min_value=0,
            max_value=200,
            value=100
        )

    # ---------------- COLUMN 3 ----------------
    with col3:
        night_mins = st.number_input(
            "Night Minutes",
            min_value=0.0,
            max_value=400.0,
            value=200.0
        )

        night_calls = st.number_input(
            "Night Calls",
            min_value=0,
            max_value=200,
            value=100
        )

        customer_calls = st.number_input(
            "Customer Service Calls",
            min_value=0,
            max_value=20,
            value=1
        )

    # ---------------- PREDICT BUTTON ----------------
    if st.button("🚀 Predict Churn"):
        # Build input dataframe
        input_data = pd.DataFrame([{
          "account.length": account_length,
          "voice.plan": voice_plan,
          "voice.messages": voice_messages,
          "intl.plan": intl_plan,
          "intl.mins": intl_mins,
          "intl.calls": intl_calls,
          "day.mins": day_mins,
          "day.calls": day_calls,
          "eve.mins": eve_mins,
          "eve.calls": eve_calls,
          "night.mins": night_mins,
          "night.calls": night_calls,
          "customer.calls": customer_calls
        }])

        # Keep exact trained order
        input_data = input_data[trained_features]

        # Scale
        num_cols=input_data.select_dtypes(include=['int64', 'float64']).columns
        input_data[num_cols] = scaler.transform(input_data[num_cols])

        #Encoding
        input_data["voice.plan"] = input_data["voice.plan"].map({"Yes": 1, "No": 0})
        input_data["intl.plan"] = input_data["intl.plan"].map({"Yes": 1,"No": 0})

        # Predict
        prediction = model.predict(input_data)[0]
        churn_prob = model.predict_proba(input_data)[0][1]

        # ---------------- RISK CATEGORY ----------------
        if churn_prob > 0.7:
            risk = "🔴 High Risk"
            suggestion = [
                "Offer loyalty discounts",
                "Provide premium customer retention support",
                "Review international/voice plan pricing",
                "Reduce customer service friction"
            ]

        elif churn_prob > 0.3:
            risk = "🟡 Medium Risk"
            suggestion = [
                "Personalized retention offers",
                "Monitor customer service calls",
                "Encourage plan optimization"
            ]

        else:
            risk = "🟢 Low Risk"
            suggestion = [
                "Maintain current satisfaction",
                "Upsell suitable premium plans",
                "Reward customer loyalty"
            ]

        # ---------------- DISPLAY ----------------
        churn_label = "Yes" if prediction == 1 else "No"

        st.success(f"Predicted Churn: {churn_label}")
        st.info(f"Risk Level: {risk}")
        st.info(f"Churn Probability: {churn_prob:.2%}")

        st.progress(int(churn_prob * 100))

        st.subheader("💡 Retention Strategy")
        for item in suggestion:
            st.write(f"- {item}")

# BATCH PREDICTION PAGE
def batch_page():

    st.subheader("📂 Upload File for Batch Prediction")

    uploaded_file = st.file_uploader(
        "Upload File (CSV or Excel)",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:
            # Detect file type
            file_name = uploaded_file.name.lower()

            if file_name.endswith(".csv"):
                data = pd.read_csv(uploaded_file)

            elif file_name.endswith(".xlsx"):
                data = pd.read_excel(uploaded_file)

            else:
                st.error(
                    "Unsupported file format. Please upload CSV or XLSX."
                )
                st.stop()

            st.success(
                f"File uploaded successfully: {uploaded_file.name}"
            )

            # File Summary
            col1, col2, col3 = st.columns(3)

            col1.metric("Rows", data.shape[0])
            col2.metric("Columns", data.shape[1])
            col3.metric(
                "File Type",
                file_name.split(".")[-1].upper()
            )

            # Preview
            st.write("### Preview of Uploaded Data")
            st.dataframe(data.head())


            data.columns = data.columns.str.strip().str.lower()
            trained_features = [col.lower() for col in trained_features]
            # VALIDATE REQUIRED FEATURES
            missing_cols = [
                col for col in trained_features
                if col not in data.columns
            ]

            extra_cols = [
                col for col in data.columns
                if col not in trained_features
            ]

            # Missing Columns
            if missing_cols:
                st.error(
                    f"❌ Missing Required Columns: {missing_cols}"
                )
                st.stop()
            # Extra Columns Warning
            if extra_cols:
                st.warning(
                    f"⚠ Extra Columns Found (will be ignored): {extra_cols}"
                )

            # Keep only trained features
            batch_data = data[trained_features].copy()

            # OPTIONAL BINARY MAPPING
            if batch_data["voice.plan"].dtype == "object":
                batch_data["voice.plan"] = (batch_data["voice.plan"].map({"Yes": 1, "No": 0}))

            if batch_data["intl.plan"].dtype == "object":
                batch_data["intl.plan"] = (batch_data["intl.plan"].map({"Yes": 1, "No": 0}))

            #Fixing Incorrect Data Types in Uploaded Data
            cols_to_convert=[]
            for col in batch_data.select_dtypes(include='object').columns:
                converted = pd.to_numeric(batch_data[col], errors='coerce')
                numeric_ratio = converted.notnull().sum() / len(batch_data)

                if numeric_ratio > 0.8:
                  cols_to_convert.append(col)

            for col in cols_to_convert:
                batch_data[col] = pd.to_numeric(batch_data[col].replace(['Nan', 'nan', 'NA', ''], pd.NA), errors='coerce')

            # Check missing values
            if batch_data.isnull().sum().sum() > 0:
               st.warning("Missing values found. Filling with safe defaults.")
               binary_cols = ["voice.plan", "intl.plan"]
               batch_data[binary_cols] = batch_data[binary_cols].fillna(0)

               numeric_cols = batch_data.columns.difference(binary_cols)
               batch_data[numeric_cols] = batch_data[numeric_cols].fillna(
                batch_data[numeric_cols].median())

            # Scale
            binary_cols = ["voice.plan", "intl.plan"]
            numeric_cols = batch_data.columns.difference(binary_cols)
            batch_scaled = batch_data.copy()
            batch_scaled[numeric_cols] = scaler.transform(batch_scaled[numeric_cols])

            # Run Prediction

            if st.button("🚀 Run Batch Prediction"):
                # Predict
                predictions = model.predict(batch_scaled)
                probabilities = model.predict_proba(batch_scaled)[:, 1]

                # Add Prediction Columns
                data["Predicted_Churn"] = predictions

                data["Predicted_Churn_Label"] = (data["Predicted_Churn"].map({1: "Yes",0: "No"}))

                data["Churn_Probability"] = probabilities

                # Success Message
                st.success(
                    "✅ Batch Prediction Completed!"
                )

                # SUMMARY METRICS
                churn_count = (data["Predicted_Churn"] == 1).sum()

                churn_rate = (churn_count / len(data)) * 100

                col1, col2, col3 = st.columns(3)

                col1.metric("Total Customers",len(data))

                col2.metric("Predicted Churners",churn_count)

                col3.metric("Predicted Churn Rate",f"{churn_rate:.2f}%")

                # Average Probability
                st.info(
                    f"Average Churn Probability: "
                    f"{data['Churn_Probability'].mean():.2%}"
                )

                # RESULTS TABLE
                st.write("### Prediction Results")
                st.dataframe(data)

                # HIGH RISK CUSTOMERS
                high_risk = data[
                    data["Churn_Probability"] > 0.7
                ]

                st.write(
                    "### 🔴 High Risk Customers (>70%)"
                )

                if len(high_risk) > 0:
                    st.dataframe(high_risk)
                else:
                    st.write(
                        "No high-risk customers found."
                    )

                #chart
                st.markdown("---")

                total = len(data)
                churn_counts=data["Predicted_Churn"].value_counts()
                churn_pct = (churn_counts.get(1, 0) / total) * 100
                non_churn_pct = (churn_counts.get(0, 0) / total) * 100

                fig, ax = plt.subplots()

                ax.bar(
                    ["Non-Churn %", "Churn %"],
                    [non_churn_pct, churn_pct]
                )

                ax.set_ylabel("Percentage")
                ax.set_title("Churn Rate Percentage")

                st.subheader("📊 Churn Rate Percentage")
                st.pyplot(fig)

                st.markdown("---")

                # Business recommendations
                st.markdown("""
                ### 💡 Business Recommendations:

                - 🎯 Focus retention campaigns on **high-risk customers (>70%)**
                - 💰 Offer discounts or loyalty rewards to reduce churn
                - 📞 Increase customer support engagement for medium-risk users
                - 📊 Monitor churn probability trends monthly
                - 🔍 Identify top features influencing churn for strategy improvement
                """)

                st.markdown("---")

                # Model note
                st.info("""
                🤖 Model Note:
                Predictions are based on trained machine learning model.
                Probability values represent churn risk score, not certainty.
                """)
                # DOWNLOAD RESULTS
                csv_output = (data.to_csv(index=False).encode("utf-8"))

                st.download_button(
                    label="📥 Download Prediction Results (CSV)",
                    data=csv_output,
                    file_name="batch_predictions.csv",
                    mime="text/csv"
                )

        # ERROR HANDLING
        except Exception as e:
            st.error(
                f"❌ Error processing file: {str(e)}"
            )


def modelperf_page():
    st.subheader("🤖 Model Evaluation Metrics")

    metrics = pd.DataFrame({
        "Metric": [
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC-AUC"
        ],
        "Score": [
            benchmarks["accuracy"],
            benchmarks["precision"],
            benchmarks["recall"],
            benchmarks["f1_score"],
            benchmarks["roc_auc"]
        ]
    })

    st.table(metrics)



def about_page():
    st.subheader("📘 About This Project")

    st.write("**Telecom Customer Churn Prediction** helps telecom companies:")
    st.write("- Predict customer churn")
    st.write("- Identify high-risk users")
    st.write("- Improve retention strategies")
    st.write("- Reduce customer loss")

    st.write("### Tech Stack")
    st.write("- Pandas")
    st.write("- Python")
    st.write("- XGBoost")
    st.write("- Scikit-learn")
    st.write("- Streamlit")


# SIDEBAR NAVIGATION
st.sidebar.title("📌 Navigation")

pages = {
    "Customer Prediction": customer_page,
    "Batch Prediction": batch_page,
    "Model Performance":modelperf_page,
    "About Project":about_page
}
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
