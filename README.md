## Cardio Risk Prediction using KNN

This project uses a K-Nearest Neighbors (KNN) model to predict the risk of cardiovascular disease based on various health factors. The model is trained on a dataset containing information about individuals' age, gender, cholesterol levels, blood pressure, etc., and their corresponding cardiovascular health status.

### Features:

- **Data Preprocessing:**  Handles missing values, scales features, and prepares the data for model training.
- **KNN Model Training:** Trains a KNN classifier with optimized parameters (including calibration) to improve prediction accuracy.
- **Model Persistence:** Saves the trained model and scaler for future use without retraining.
- **Interactive Web App (Streamlit):** Allows users to input their health information and receive a prediction of their cardiovascular risk.

### How to Run:

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/rohit-krish/work-cardio-risk-prediction
   ```

2. **Navigate to the Project Directory:**
   ```bash
   cd work-cardio-risk-prediction
   ```

3. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv 
   ```
   - **Linux:**
     ```bash
     source venv/bin/activate
     ```
   - **Windows:**
     ```bash
     venv/Scripts/Activate.bat
     ```

4. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

5. **Train the Model (Run Once):**
   ```bash
   python model.py
   ```

6. **Launch the Streamlit App:**
   ```bash
   streamlit run app.py 
   ```

### Usage:

- Once the Streamlit app is running, access it in your web browser at the address provided in the console output (usually `http://localhost:8501`).
- Enter your health information into the input fields in the app.
- Click the "Predict" button to receive a prediction of your cardiovascular risk.

### Notes:

- The accuracy of the prediction depends on the quality and representativeness of the training data.
- This model is intended for informational purposes only and should not be used as a substitute for professional medical advice. Consult with a healthcare professional for any concerns about your cardiovascular health.

