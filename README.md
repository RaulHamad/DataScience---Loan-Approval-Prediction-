# DataScience- Loan Approval Prediction

Welcome to the Loan Approval Prediction App, a smart web interface built using Python and Shiny for Python that simulates a loan evaluation system for financial institutions.

🚀 What This App Does
This app helps users determine the likelihood of loan approval based on a variety of personal and financial features. Powered by a Random Forest Classifier and enhanced with pre-trained One-Hot Encoding and Feature Scaling, the app delivers quick and interpretable predictions using an intuitive UI.

🔍 How It Works
🧾 Input Data: Users enter key information such as education level, income, credit score, loan amount, and asset values.

⚙️ Preprocessing: Inputs are encoded and scaled to match the model's training pipeline.

🧠 Prediction: A trained Random Forest model processes the inputs and returns a simple status:
✅ Approved or ❌ Not Approved.

🛠️ Tech Stack
Python

Shiny for Python (UI Framework)

scikit-learn (Model + Preprocessing)

Pandas / NumPy

faicons for clean iconography

📦 Files Included
app.py – the main Shiny app with UI and prediction logic

model_aprendizaje.sav – the trained Random Forest model

scaler.pkl & one_hot_encoder.pkl – data transformation tools

loan_approval_dataset.csv – dataset used for training

#-------------------------------------------

Bienvenido a la aplicación de predicción de aprobación de préstamos, una interfaz web inteligente creada con Python y Shiny para Python que simula un sistema de evaluación de préstamos para instituciones financieras.

🚀 Qué hace esta aplicación
Esta aplicación ayuda a los usuarios a determinar la probabilidad de aprobación de un préstamo en función de una variedad de características personales y financieras. Impulsada por un clasificador de bosque aleatorio y mejorada con codificación One-Hot entrenada previamente y escalamiento de características, la aplicación ofrece predicciones rápidas e interpretables mediante una interfaz de usuario intuitiva.

🔍 Cómo funciona
🧾 Datos de entrada: Los usuarios ingresan información clave como el nivel de educación, los ingresos, el puntaje crediticio, el monto del préstamo y los valores de los activos.

⚙️ Preprocesamiento: las entradas se codifican y escalan para que coincidan con el proceso de entrenamiento del modelo.

🧠 Predicción: Un modelo de Random Forest entrenado procesa las entradas y devuelve un estado simple:
✅ Aprobado o ❌ No Aprobado.

Pila tecnológica
Pitón

Shiny para Python (marco de interfaz de usuario)

scikit-learn (modelo + preprocesamiento)

Pandas / NumPy

faicons para una iconografía limpia

📦 Archivos incluidos
app.py: la aplicación principal de Shiny con interfaz de usuario y lógica de predicción

model_aprendizaje.sav – el modelo de Bosque aleatorio entrenado

scaler.pkl y one_hot_encoder.pkl: herramientas de transformación de datos

loan_approval_dataset.csv – conjunto de datos utilizado para la capacitación
