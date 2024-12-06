import streamlit as st
import pandas as pd
import joblib
import base64

# Streamlit App Layout and Design
st.set_page_config(page_title="Basketball Player Longevity Prediction", layout="centered")

# Function to handle the background image
def set_background_image(image_path):
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

# Set the background image
set_background_image('basketball.jpg')  # Ensure 'basketball.jpg' is in the same folder

# Load the pre-trained model and scaler
model_path = "stack_model_best_solution.pkl"
scaler_path = "scaler.pkl"

stack_model = joblib.load(model_path)  # Load the trained model
scaler = joblib.load(scaler_path)     # Load the saved scaler

# Streamlit GUI
def main():
    # App Title and Description
    st.markdown(
        """
        <h1 style='text-align: center; color: #ffff;'>Basketball Player Longevity Prediction 🏀</h1>
        """, 
        unsafe_allow_html=True
    )
    
    # Collapsible section for "About This App"
    with st.expander("ℹ️ About This App"):
        st.write("""
        This app predicts the career longevity of basketball players based on their game statistics. 
        It uses a **Stack Ensemble Model** trained on standardized basketball data to make predictions.
        
        **How it works:**
        - Enter the player's stats below.
        - The model will evaluate the data and predict if the player will have a **long** or **short** career.
        """)

    # Player Name Input
    user_name = st.text_input("Enter your name:", placeholder="e.g., John Doe")
    st.markdown("<hr>", unsafe_allow_html=True)

    # Player Data Input Section
    st.header("Enter Player Data 📊")

    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)

    # Inputs inside columns with tooltips for each statistic
    with col1:
        injury = st.radio(
            "Injury History", 
            ["No", "Yes"], 
            index=0, 
            help="Indicate if the player has a history of injuries (Yes = 1, No = 0)"
        )
        three_point_percent = st.slider(
            "3-Point Percentage (%)", 
            min_value=0.0, max_value=100.0, step=0.1, value=45.0,
            help="Percentage of successful 3-point attempts."
        )
        turnovers = st.slider(
            "Turnovers (per game)", 
            min_value=0.0, max_value=10.0, step=0.1, value=2.0,
            help="Average number of turnovers per game."
        )

    with col2:
        free_throw_made = st.slider(
            "Free Throws Made (per game)", 
            min_value=0.0, max_value=10.0, step=0.1, value=1.5,
            help="Number of free throws made on average per game."
        )
        blocks = st.slider(
            "Blocks (per game)", 
            min_value=0.0, max_value=10.0, step=0.1, value=0.8,
            help="Average number of blocks per game."
        )
        offensive_rebounds = st.slider(
            "Offensive Rebounds (per game)", 
            min_value=0.0, max_value=10.0, step=0.1, value=1.2,
            help="Average number of offensive rebounds per game."
        )

    with col3:
        free_throw_attempts = st.slider(
            "Free Throw Attempts (per game)", 
            min_value=0.0, max_value=15.0, step=0.1, value=2.3,
            help="Number of free throw attempts on average per game."
        )
        rebounds = st.slider(
            "Total Rebounds (per game)", 
            min_value=0.0, max_value=20.0, step=0.1, value=5.4,
            help="Average number of total rebounds per game."
        )
        field_goals_made = st.slider(
            "Field Goals Made (per game)", 
            min_value=0.0, max_value=20.0, step=0.1, value=4.6,
            help="Number of field goals made on average per game."
        )

    # Prepare input data
    player_data = pd.DataFrame({
        'injury': [1 if injury == "Yes" else 0],
        '3_point_percent': [three_point_percent],
        'turnovers': [turnovers],
        'free_throw_made': [free_throw_made],
        'blocks': [blocks],
        'offensive_rebounds': [offensive_rebounds],
        'free_throw_attempts': [free_throw_attempts],
        'rebounds': [rebounds],
        'field_goals_made': [field_goals_made],
    })

    # Prediction button
    if st.button("Predict Longevity", use_container_width=True):
        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(player_data)

        # Make prediction
        prediction = stack_model.predict(scaled_data)[0]

        # Display results with styled output
        st.markdown("<hr>", unsafe_allow_html=True)
        st.subheader("Prediction Results 🏆")
        if prediction == 1:
            st.success(f"Hello **{user_name}**, the player is likely to have a **long career**. 🏀🎉")
        else:
            st.warning(f"Hello **{user_name}**, the player may have a **short career**. ⚠️")

# Run the app
if __name__ == "__main__":
    main()
