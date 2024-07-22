import numpy as np
import pandas as pd
import streamlit as st
#from sklearn import preprocessing
#import pickle
import joblib
import xgboost
import matplotlib.pyplot as plt

# Load model
with open('base_biofilter_no_dup2.pkl', 'rb') as file:
    model = pickle.load(file)

# Define inputs
cols = ['W1 Total Disinfection Credit Virus', 'Hours Between Washes', 'W1 Pre Ozone Turbidity', 'W1 Mix 2 Chlorine', 
        'W1 Chloramine Disinfection Credit Giardia', 'W2 Raw Calcium Hardness', 'W1 Mix 3 Chlorine', 
        'W1 Chloramine Disinfection Credit Inactivation Ratio', 'W1 Combined Filter Effluent Turbidity', 
        'W1 Chloramine Disinfection Credit Virus', 'W1 Total Disinfection Credit Giardia', 'W2 Raw Fluoride AM', 
        'Final NTU', 'W2 Raw Phopshate', 'W2 Post Clear Well % Oxygen Saturation', 'MG Since las BW']

def main():
    st.title("Biofilter Final Headloss Predictor")

    # Load your DataFrame here
    df =joblib.load('unit10_for_app.pkl')   # Replace with your actual data file

    # Create tabs
    tab1, tab2 = st.tabs(["Prediction", "Sensitivity Analysis"])

    with tab1:
        st.header("Prediction")
        # Create sliders for each column
        user_inputs = {}
        for col in cols:
            if col in df.columns:
                median_value = df[col].median()
                min_value = df[col].min()
                max_value = df[col].max()
                
                # Create a slider for each column
                user_input = st.slider(
                    label=col,
                    min_value=float(min_value),
                    max_value=float(max_value),
                    value=float(median_value),
                    step=0.01
                )
                
                # Store the user input in a dictionary
                user_inputs[col] = user_input
            else:
                st.warning(f"Column '{col}' not found in the DataFrame. Skipping this input.")

        # Convert user inputs to DataFrame for prediction
        input_df = pd.DataFrame([user_inputs])
        
        if st.button('Predict'):
            prediction = model.predict(input_df)
            st.write(f'Predicted Final Headloss: {prediction[0]}')

    with tab2:
        st.header("Sensitivity Analysis")
        # Select variable for sensitivity analysis
        variable = st.selectbox("Select variable for sensitivity analysis", cols)
        
        if variable in df.columns:
            # Use the inputs from the first tab as basis
            base_values = user_inputs.copy()
            
            # Generate a range of values for the selected variable
            min_value = df[variable].min()
            max_value = df[variable].max()
            values = np.linspace(min_value, max_value, 100)
            
            # Store predictions
            predictions = []
            for value in values:
                base_values[variable] = value
                input_df = pd.DataFrame([base_values])
                prediction = model.predict(input_df)
                predictions.append(prediction[0])
            
            # Plot the results
            fig, ax = plt.subplots(2, 1, figsize=(10, 12))
            
            # Sensitivity analysis plot
            ax[0].plot(values, predictions, label=f'Sensitivity Analysis for {variable}')
            ax[0].set_xlabel(variable)
            ax[0].set_ylabel('Predicted Final Headloss')
            ax[0].set_title('Sensitivity Analysis')
            ax[0].legend()
            
            # Boxplot
            ax[1].boxplot(df[variable].dropna(),vert=False)
            ax[1].set_title(f'Boxplot of {variable}')
            ax[1].set_ylabel(variable)
            
            st.pyplot(fig)
        else:
            st.warning(f"Column '{variable}' not found in the DataFrame. Cannot perform sensitivity analysis.")

if __name__ == '__main__':
    main()
