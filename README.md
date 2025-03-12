# Bank-Customer-Churn-using-Deep-learning

## Bank Customer Churn Prediction using Deep Learning

### 📌 Overview
This project aims to predict customer churn in a bank using deep learning techniques. It utilizes **TensorFlow/Keras** for model building, along with **pandas, scikit-learn, and Matplotlib** for data preprocessing and visualization. The dataset contains customer details such as **credit score, geography, gender, age, tenure, balance, and other banking-related features**.

### 📂 Project Structure
```
📦 Bank-Customer-Churn-using-Deep-learning
├── 📁 dataset              # Dataset used for training
├── 📁 models               # Trained model files
├── 📜 app.py               # Deployment script (Flask or Streamlit)
├── 📜 model.h5             # Trained deep learning model
├── 📜 requirements.txt     # Dependencies required to run the project
└── 📜 README.md            # Project documentation
```

### 📊 Dataset
The dataset consists of bank customer records with the following features:
- **CreditScore** - Customer's credit score
- **Geography** - Customer's country (France, Germany, Spain)
- **Gender** - Male or Female
- **Age** - Customer's age
- **Tenure** - Number of years with the bank
- **Balance** - Account balance
- **NumOfProducts** - Number of bank products owned
- **HasCrCard** - Whether the customer has a credit card (1 = Yes, 0 = No)
- **IsActiveMember** - Whether the customer is an active bank member (1 = Yes, 0 = No)
- **EstimatedSalary** - Estimated salary of the customer
- **Exited** - Churn status (1 = Customer left the bank, 0 = Customer stayed)

### 🏗️ Model Training
The model follows these steps:
1. **Data Preprocessing**: Handling missing values, encoding categorical features, and scaling numerical values.
2. **Feature Engineering**: One-hot encoding categorical data and normalizing numerical data.
3. **Building the Model**: Using **TensorFlow/Keras** with multiple dense layers, batch normalization, and dropout for regularization.
4. **Training**: Optimized using Adam optimizer and binary cross-entropy loss function.
5. **Evaluation**: Analyzing model performance using accuracy, precision, recall, and F1-score.

### 🚀 How to Run the Project
#### 1️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

#### 2️⃣ Run the Training Script
```bash
python src/train_model.py
```

#### 3️⃣ Run the Web App (Flask or Streamlit)
```bash
streamlit run app.py  # For Streamlit
# OR
python app.py         # For Flask
```

### 📈 Model Performance
- Accuracy: **~85%**
- Precision, Recall, and F1-score were used to evaluate performance.
- Model was fine-tuned using **hyperparameter tuning** and **early stopping**.

### 🛠️ Future Improvements
- **Feature Engineering**: Try more advanced features.
- **Hyperparameter Tuning**: Improve model generalization.
- **Deployment**: Deploy using **Docker** or **Cloud services**.

### 🤝 Contribution
Feel free to open issues or submit pull requests. Contributions are welcome!

### 📜 License
This project is open-source and available under the **MIT License**.


### Here is the link to the streamlit  website
https://bankcustomerschurnprediction.streamlit.app

---
🚀 Happy Coding! 🎯

