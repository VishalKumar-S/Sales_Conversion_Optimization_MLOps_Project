import streamlit as st
import pandas as pd
import h2o
from h2o.automl import H2OAutoML
import time
import matplotlib.pyplot as plt
import streamlit.components.v1 as components
from datetime import datetime, timedelta
from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.metric_preset import TargetDriftPreset
from evidently.metric_preset import DataQualityPreset
from evidently.metric_preset.regression_performance import RegressionPreset
import time
from src.clean_data import DataPreprocessor
from pipelines.ci_cd_pipeline import ci_cd_pipeline
from email_validator import validate_email, EmailNotValidError





# Initialize H2O
h2o.init()

# Load trained H2O AutoML model
model_path = "models/best_model.zip/StackedEnsemble_BestOfFamily_1_AutoML_1_20240122_204822.zip"
best_tree_model_path = "models/best_tree_model.zip/GBM_2_AutoML_5_20240118_62124.zip"

model = h2o.import_mojo(model_path)
best_tree_model = h2o.import_mojo(best_tree_model_path)

# Sidebar to switch between Prediction App and Interpretability
app_selector = st.sidebar.radio("Select App", ("Prediction App", "Interpretability", "Data and Model Reports", "Test Your Batch Data"))
input_data_file = "data/input_data.csv"


if app_selector == "Prediction App":
    # Prediction App
    st.title('Prediction App')
    st.header('User Input Data')

    input_data = {}

    # Input fields
    input_data['Impressions'] = st.number_input('Impressions', min_value=0)
    input_data['Clicks'] = st.number_input('Clicks', min_value=0)
    input_data['Spent'] = st.number_input('Spent', min_value=0.0)
    input_data['Total_Conversion'] = st.number_input('Total number of queries', min_value=0)
    input_data['CPC'] = st.number_input('Cost Per Click (CPC)', min_value=0.0)

    with st.expander("Help"):
        st.write("""
            - Impressions: Number of times the ad was shown
            - Clicks: Clicks on the ad  
            - Spent: Amount paid to show the ad
            - Total_Conversion: People who inquired after seeing the ad 
            - CPC: Cost per click
        """)

    # Save input data to a DataFrame and store it in a file
    input_df = pd.DataFrame([input_data])
    input_df.to_csv(input_data_file, index=False)

    # Make predictions on user input
    if st.button('Predict'):
        prediction = model.predict(h2o.H2OFrame(pd.DataFrame([input_data]))).as_data_frame()['predict'][0]
        st.write(f"Predicted Approved Conversion: **{round(prediction, 2)}**")
        st.write("To understand why the model made this prediction, explore the 'Interpretability' section.")
        

    def redirect(url):
        st.write(f"Redirecting to Neptune.ai...")
        st.write(f"Please wait...")
        st.write("Please note that you’ll need to sign in or register with a Neptune.ai account to view the full details of my project.")
        time.sleep(3)  # 3-second delay for demonstration
        st.markdown(f'<a href="{url}" target="_blank">Click here to go to Neptune.ai metrics</a>', unsafe_allow_html=True)

    # Button for redirection
    if st.button('See Model Metrics on Neptune.ai'):
        redirect("https://app.neptune.ai/Vishal-Kumar-S/Sales-Conversion-Optimisation-MLOps-Project")


elif app_selector == "Interpretability":
    # Interpretability
    st.title('Interpretability')
    st.write("Explore the interpretability of the model predictions. "
             "Click on the buttons in the sidebar to view interpretability reports and visualizations.")
    st.write("Note: For the SHAP Local Plot, ensure you have entered values for all user input fields in the Prediction App.")


    # Detailed Interpretability Report
    if st.sidebar.button('Detailed Interpretability Report'):
        # Read the HTML file
        with open('Interpretability_reports/Global_interpretability.html', 'r', encoding='utf-8') as f:
            html_string = f.read()

        # Embed the HTML into the Streamlit app
        components.html(html_string, width=850, height=850, scrolling=True)

    # SHAP Global Plot
    if st.sidebar.button('SHAP Global Plot'):
        # Load and display the pre-generated SHAP Global Plot
        shap_global_plot_path = "Interpretability_reports/Shap_interpretability.png"
        st.image(shap_global_plot_path, use_column_width=True)

    # SHAP Local Plot
    if st.sidebar.button('SHAP Local Plot'):
        # Load input data from the saved file
        input_df = pd.read_csv(input_data_file)

        # Convert input data to H2O DataFrame
        input_frame = h2o.H2OFrame(input_df)

        # Calculate SHAP values for the provided input data
        shap_values_local = best_tree_model.predict_contributions(input_frame).as_data_frame().iloc[0, 1:]

        # Include 'Impressions' in case it's missing
        if 'Impressions' not in shap_values_local.index:
            shap_values_local['Impressions'] = 0

        # Exclude 'BiasTerm' from the index
        shap_values_local = shap_values_local[shap_values_local.index != 'BiasTerm']

        # Plot
        fig, ax = plt.subplots()

        # Plot bars using live SHAP values with both negative and positive values
        ax.barh(shap_values_local.index, shap_values_local, color=['red' if val < 0 else 'blue' for val in shap_values_local])

        # Set x-axis labels with SHAP values
        ax.set_xlabel("SHAP Value")

        # Set y-axis labels with feature names and values
        ax.set_yticklabels([f"{feat} ({input_df[feat].iloc[0]})" for feat in shap_values_local.index])

        # Set y-axis ticks
        ax.set_yticks(range(len(shap_values_local)))

        # Set labels and title
        ax.set_ylabel("Feature (Value)")
        plt.title("SHAP Values")

        # Display the plot in Streamlit
        st.pyplot(fig) 


elif app_selector == "Data and Model Reports":
    def data_and_model_reports(path):
        numerical_features = ['ad_id', 'xyz_campaign_id', 'fb_campaign_id',
       'interest', 'Impressions', 'Clicks', 'Spent', 'Total_Conversion',
       'Approved_Conversion', 'CPC']

        categorical_features = ['age','gender']

        column_mapping = ColumnMapping()
        column_mapping.target = 'Approved_Conversion'
        column_mapping.prediction = 'prediction'
        column_mapping.numerical_features = numerical_features
        #column_mapping.categorical_features = categorical_features


        st.title("Data & Model Monitoring App")
        st.write("You are in the Data & Model Monitoring App. Select one of the options given below")


        st.subheader("Select Reports to Generate")
        generate_data_quality = st.checkbox("Generate Data Quality Report")        
        generate_data_drift = st.checkbox("Generate Data Drift Report")
        generate_target_drift = st.checkbox("Generate Target Drift Report")
        generate_model_report = st.checkbox("Generate Model Performance Report")
        
        
        

        if st.button("Submit"):
            st.write("Fetching your current batch data...")
            df=pd.read_csv(path)

            reference_raw_data = pd.read_csv("https://sale2.s3.us-east-2.amazonaws.com/KAG_conversion_data.csv")
            current_raw_data = df

            preprocessor = DataPreprocessor(reference_raw_data)
            reference_data = preprocessor.clean_data()

            preprocessor = DataPreprocessor(current_raw_data)
            current_data = preprocessor.clean_data()
            
            h2o_ref_data = h2o.H2OFrame(reference_data)
            h2o_curr_data = h2o.H2OFrame(current_data)

            # Get predictions for train and test datasets
            ref_pred = model.predict(h2o_ref_data)
            curr_pred = model.predict(h2o_curr_data)

            # Convert H2OFrames to pandas DataFrames
            ref_df = ref_pred.as_data_frame()
            curr_df = curr_pred.as_data_frame()



            st.write("Please scroll down to see your report")
            st.write("Reference Dataset Shape:", reference_data.shape)
            st.write("Current Dataset Shape:", current_data.shape)


            
            # Generate selected reports and display them
            
            def display_report(report, report_name: str):
                """
                Displays a report generated by Evidently.

                Args:
                    report (Report): The Evidently report to display.
                    report_name (str): Name of the report (e.g., "Model Performance Report").
                """
                st.write(f"{report_name}")
                st.components.v1.html(report.get_html(), width=850, height=850, scrolling=True)

            
            if generate_data_quality:
                st.session_state.button_1_Clicked = False
                st.session_state.button_2_Clicked = False
                st.write("### Data Quality Report")
                st.write("Generating Data Quality Report...")
                data_quality_report = Report(metrics=[DataQualityPreset()])
                data_quality_report.run(
                    reference_data=reference_raw_data,
                    current_data=current_raw_data,
                    column_mapping=column_mapping
                )
                
                display_report(data_quality_report, "Data Quality Report")
            
       



            if generate_data_drift:
                st.session_state.button_1_Clicked = False
                st.session_state.button_2_Clicked = False
                st.write("### Data Drift Report")
                st.write("Generating Data Drift Report...")
                data_drift_report = Report(metrics=[DataDriftPreset()])
                data_drift_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping
                )

                display_report(data_drift_report, "Data Drift Report")


            # Add predictions to the original train and test datasets
            reference_data['prediction'] = ref_df
            current_data['prediction'] = curr_df
            
            if generate_target_drift:
                st.session_state.button_1_Clicked = False
                st.session_state.button_2_Clicked = False
                st.write("### Target Drift Report")
                st.write("Generating Target Drift Report...")
                target_drift_report = Report(metrics=[TargetDriftPreset()])
                target_drift_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping
                )
            
                display_report(target_drift_report, "Target Drift Report")
            

            if generate_model_report:
                st.session_state.button_1_Clicked = False
                st.session_state.button_2_Clicked = False
                st.write("### Model Performance Report")
                st.write("Generating Model Performance Report...")
                regression_performance_report = Report(metrics=[RegressionPreset()])
                regression_performance_report.run(
                    reference_data=reference_data,
                    current_data=current_data,
                    column_mapping=column_mapping
                )
                display_report(regression_performance_report, "Model Performance Report")
                
    st.write("Click any one of the buttons in the sidebar to view the reports")
    if "button_1_Clicked" not in st.session_state:
        st.session_state.button_1_Clicked = False
    
    def callback1():
        st.session_state.button_1_Clicked = True
    
    if "button_2_Clicked" not in st.session_state:
        st.session_state.button_2_Clicked = False
    
    def callback2():
        st.session_state.button_2_Clicked = True
    

    # Button for existing batch data
    if st.sidebar.button('Use Existing Batch Data', on_click= callback1) or st.session_state.button_1_Clicked:
        st.session_state.button_2_Clicked = False
        # Determine the date for the existing production batch
        # You can customize this logic based on your production schedule
        current_date = datetime.now().strftime("%Y-%m-%d")  # Current date

        # Assuming production batches are generated daily, you can adjust this based on your production cycle
        production_batch_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

        # Provide a dynamic message with the existing batch date
        st.write(f"Using existing batch data for the production batch of {production_batch_date}.")

        path="Production_data/synthetic_data.csv"
        data_and_model_reports(path)



    # Button for uploading dataset
    if st.sidebar.button('Upload Your Own Dataset', on_click= callback2) or st.session_state.button_2_Clicked:
        st.session_state.button_1_Clicked = False
        uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

        if uploaded_file is not None:
            # Save the uploaded file
            with open("Production_data/user_uploaded_dataset.csv", "wb") as f:
                f.write(uploaded_file.getvalue())

            # Provide a confirmation message
            path="Production_data/user_uploaded_dataset.csv"
            st.success("Dataset uploaded successfully!")
            st.write("Now you can explore reports and visualizations based on your uploaded dataset.")
            data_and_model_reports(path)
        else:
            # Prompt the user to upload a valid CSV file
            st.warning("Please upload a valid CSV file to proceed.")



elif app_selector == "Test Your Batch Data":
    st.title("Batch Data Validation")
    st.write("Welcome to the Batch Data Validation section. We'll assess the quality and integrity of your dataset "
             "through a comprehensive set of 67 validation tests. Please follow the steps below:")

    # Step 1: Dataset Upload
    st.subheader("Step 1: Upload Your Batch Dataset")
    st.write("Upload the CSV file containing your batch dataset. Ensure the file includes all relevant features required for validation.")

    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

    if uploaded_file is not None:
        # Save the uploaded file
        with open("Production_data/batch_validation_data.csv", "wb") as f:
            f.write(uploaded_file.getvalue())

        # Provide a confirmation message
        path = "Production_data/batch_validation_data.csv"
        st.success("📂 Dataset uploaded successfully!")

    if not uploaded_file:
        st.warning("⚠️ Please upload a valid CSV file to initiate the validation process.")
        st.stop()

    # Step to Get Email Address
    st.subheader("Step 2: Provide Email Address for Alerts")
    st.write("""Please enter your email address below.  
           If any of the validation tests fail, you will receive 
           an email alert with the test report attached. This can
           help you investigate and fix any data issues.""")
           
    email = st.text_input("Email Address")
    
    # Check if the email is provided before attempting validation
    if email:
        try:
            validate_email(email)
            st.success("Valid Email Address", icon="✅")
            st.write(f"We will send an alert to {email} if any tests fail")   
        except EmailNotValidError:
            st.error("Invalid Email", icon="🚨")
            st.stop()

        st.write("In case of any failed tests, alerts will also be posted to:") 

        st.markdown("Discord: [#failed-alerts](https://discord.gg/bxZx6EGVMD)") 

        st.markdown("Slack: [#sales-conversion-test-failures](https://join.slack.com/t/vishalsworkspaceco/shared_invite/zt-2b00eaite-KHPsBmlsM2JtsmR2oN0qrQ)") 

        st.write("You can check the above channels at any time to view the latest failed validation results.")
        st.write("Now, we'll proceed with the validation tests.")

        # Step 2: Data Validation Progress
        st.subheader("Step 3: Data Validation Progress")
        st.write("The dataset is undergoing 67 validation tests to ensure its quality and adherence to expected standards. "
                 "Please wait while the validation process is in progress.....")

        # Run the validation tests
        ci_cd_pipeline(path,email)
        st.write("🎉 All tests are validated successfully for the batch data. Your data is ready to go!")

        # Displaying visualizations
        st.subheader("Visualizations")
        st.write("Below are two visualizations to help you understand the validation results:")
        
        # Prediction vs Actual Scatter Plot
        predictions_scatter_plot_path = "CML_Reports/predictions_scatter_plot.png"
        st.image(predictions_scatter_plot_path, caption="Predictions vs Actuals", use_column_width=True)

        # Residuals Plot
        residuals_plot_path = "CML_Reports/residuals_plot.png"
        st.image(residuals_plot_path, caption="Residuals Plot", use_column_width=True)


