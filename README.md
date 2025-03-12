# Bank-Customer-Churn-using-Deep-learning


Bank Customer Churn Prediction using Deep Learning

📌 Overview

This project aims to predict customer churn in a bank using deep learning techniques. It utilizes TensorFlow/Keras for model building, along with pandas, scikit-learn, and Matplotlib for data preprocessing and visualization. The dataset contains customer details such as credit score, geography, gender, age, tenure, balance, and other banking-related features.

📂 Project Structure

📦 Bank-Customer-Churn-using-Deep-learning
├── 📁 dataset             # Dataset used for training
├── 📁 models              # Trained model files
├── 📁 notebooks           # Jupyter notebooks for analysis and training
├── 📁 src                 # Source code for preprocessing and model training
├── 📄 app.py              # Deployment script (Flask or Streamlit)
├── 📄 model.h5            # Trained deep learning model
├── 📄 requirements.txt    # Dependencies required to run the project
├── 📄 README.md           # Project documentation

📊 Dataset

The dataset consists of bank customer records with the following features:

CreditScore - Customer's credit score

Geography - Customer's country (France, Germany, Spain)


Gender - Male or Female

Age - Customer's age

Tenure - Number of years with the bank

Balance - Account balance

NumOfProducts - Number of bank products owned

HasCrCard - Whether the customer has a credit card (1 = Yes, 0 = No)

IsActiveMember - Whether the customer is an active bank member (1 = Yes, 0 = No)

EstimatedSalary - Customer's estimated salary

Exited (Target Variable) - Whether the customer left the bank (1 = Yes, 0 = No)

🚀 Installation

1️⃣ Clone the Repository

git clone https://github.com/rahulpoojith/Bank-Customer-Churn-using-Deep-learning.git
cd Bank-Customer-Churn-using-Deep-learning

2️⃣ Create a Virtual Environment (Recommended)

python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

3️⃣ Install Dependencies

pip install -r requirements.txt

🛠️ Usage

1️⃣ Train the Model

python src/train_model.py

This script preprocesses the data, trains a deep learning model using Keras/TensorFlow, and saves the trained model.

2️⃣ Evaluate the Model

python src/evaluate_model.py

This script loads the trained model and evaluates its performance on test data.

3️⃣ Make Predictions

To predict whether a customer will churn, run:

python src/predict.py --input sample_input.json

Replace sample_input.json with a JSON file containing customer data.

4️⃣ Deploy the Model

For deployment, use Flask or Streamlit:

python app.py

Then open http://127.0.0.1:5000/ in your browser to interact with the model.

📈 Model Architecture

The deep learning model is a fully connected neural network (ANN) built using TensorFlow/Keras:

Input layer: 10 features (after encoding categorical variables)

Hidden layers: Dense layers with ReLU activation and dropout

Output layer: Single neuron with sigmoid activation (binary classification)

Optimizer: Adam

Loss function: Binary cross-entropy

📊 Results

The model achieves an accuracy of X% on the test dataset. Precision, recall, and F1-score are used to evaluate its performance.

📌 Technologies Used

Python

TensorFlow/Keras (Deep Learning)

Scikit-learn (Data Preprocessing)

Pandas & NumPy (Data Handling)

Matplotlib & Seaborn (Data Visualization)

Flask/Streamlit (Model Deployment)

🏆 Future Improvements

Improve feature selection and hyperparameter tuning

Experiment with different architectures (CNN, RNN, etc.)

Deploy model as a web service (FastAPI, Flask, or Streamlit)

Implement a dashboard for interactive visualization

🤝 Contributing

Feel free to fork this repository and submit pull requests!

📜 License

This project is licensed under the MIT License.

🔗 Author: Rahul Poojith


