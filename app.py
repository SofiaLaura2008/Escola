import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

st.title('Modelo de Regressão Linear')
st.markdown("Este é um Data App para prever notas com base em variáveis selecionadas e classificar como alta ou baixa.")

dataset_path = 'StudentsPerformance_with_headers (1).csv'
data = pd.read_csv(dataset_path)

st.write("### Visualização do Dataset:")
st.write(data.head())

x_columns = ['Student Age', 'Weekly study hours', 'Attendance to classes', 'Mother’s education', 'Father’s education ']  
y_column = df['GRADE']

X = data[x_columns]
y = data[y_column]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Resultados do Modelo:")
st.write(f"Erro Quadrático Médio (MSE): {mse}")
st.write(f"Coeficiente de Determinação (R²): {r2}")

st.write("### Coeficientes do Modelo:")
for i, col in enumerate(x_columns):
    st.write(f"{col}: {model.coef_[i]}")

st.sidebar.subheader("Prever Nota Alta ou Baixa")
threshold = st.sidebar.slider("Defina o limite para considerar uma nota alta:", float(y.min()), float(y.max()), float(y.mean()))

input_data = {}
for col in x_columns:
    input_data[col] = st.sidebar.number_input(f"Insira o valor para {col}", value=float(X[col].mean()))
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    classification = "Alta" if prediction >= threshold else "Baixa"

    if st.sidebar.button("Realizar Predição"):
        st.write(f"### Previsão para os valores inseridos:")
        st.write(f"Nota Prevista: {prediction:.2f}")
        st.write(f"Classificação: {classification}")

        st.write("### Classificação de Notas Testadas:")
        predictions = ["Alta" if pred >= threshold else "Baixa" for pred in y_pred]
        results = pd.DataFrame({"Nota Real": y_test, "Nota Prevista": y_pred, "Classificação": predictions})
        st.write(results)

        st.subheader("Distribuição das Notas Previstas")
        hist_data = results["Nota Prevista"]
        st.bar_chart(hist_data.value_counts())
