import streamlit as st
import pandas as pd
import joblib
import base64

# Streamlit App Layout and Design
st.set_page_config(page_title="Basketball Player Longevity Prediction", layout="centered")

# Function to handle the background image
def set_background_image(image_path):
    # Read the image file in binary mode and encode it as base64
    with open(image_path, "rb") as file:
        encoded_image = base64.b64encode(file.read()).decode("utf-8")
    
    # Add the background image to the CSS
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{encoded_image}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        
        /* CSS for moving text */
        .moving-text {{
            font-size: 20px;
            font-weight: bold;
            color: #ff6600;
            white-space: nowrap;
            overflow: hidden;
            display: inline-block;
            animation: moveText 15s linear infinite;
        }}
        
        /* Animation for horizontal movement */
        @keyframes moveText {{
            0% {{
                transform: translateX(100%);
            }}
            100% {{
                transform: translateX(-100%);
            }}
        }}
        </style>
        """, 
        unsafe_allow_html=True
    )

# Set the background image
set_background_image('basketball.jpg')  # Make sure basketball.jpg is in the same folder as the script

# Load the pre-trained model
model_path = "stack_model_best_solution.pkl"
stack_model = joblib.load(model_path)  # Load the trained model

# Load the saved scaler
scaler = joblib.load("scaler.pkl")  # Assuming scaler was saved earlier

# Streamlit GUI
def main():
    # App Title and Description with enhanced markdown
    st.markdown(
        """
        <h1 style='text-align: center; color: #ff6600;'>Basketball Player Longevity Prediction 🏀</h1>
        <div class="moving-text" style="text-align: center;">
            Welcome to the **Basketball Player Longevity Prediction** app! This tool uses player game statistics to predict whether a player will have a **long** or **short** career. Please enter the required information below and click "Predict Longevity" to see the results.
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Player Name Input (moved to the main page)
    user_name = st.text_input("Enter your name:", placeholder="e.g., John Doe", label_visibility="collapsed")

    # Horizontal rule to separate sections
    st.markdown("<hr>", unsafe_allow_html=True)

    # Player Data Input Section
    st.header("Enter Player Data 📊")

    # Create three columns for input fields
    col1, col2, col3 = st.columns(3)

    # Inputs inside columns with added units and icons
    with col1:
        injury = st.radio("Injury History", ["No", "Yes"], index=0)
        three_point_percent = st.slider("3-Point Percentage (%)", min_value=0.0, max_value=100.0, step=0.1, value=45.0)
        turnovers = st.slider("Turnovers (per game)", min_value=0.0, max_value=10.0, step=0.1, value=2.0)

    with col2:
        free_throw_made = st.slider("Free Throws Made (per game)", min_value=0.0, max_value=10.0, step=0.1, value=1.5)
        blocks = st.slider("Blocks (per game)", min_value=0.0, max_value=10.0, step=0.1, value=0.8)
        offensive_rebounds = st.slider("Offensive Rebounds (per game)", min_value=0.0, max_value=10.0, step=0.1, value=1.2)

    with col3:
        free_throw_attempts = st.slider("Free Throw Attempts (per game)", min_value=0.0, max_value=15.0, step=0.1, value=2.3)
        rebounds = st.slider("Total Rebounds (per game)", min_value=0.0, max_value=20.0, step=0.1, value=5.4)
        field_goals_made = st.slider("Field Goals Made (per game)", min_value=0.0, max_value=20.0, step=0.1, value=4.6)

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
