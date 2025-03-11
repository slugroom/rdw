import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
from xgboost import XGBClassifier

# 📌 1. Carregar os datasets
df_veiculos_motoristas = pd.read_csv("veiculos_motoristas.csv")  # Dados dos veículos e motoristas
df_corridas = pd.read_csv("historico_corridas.csv")  # Histórico de corridas

# 📌 2. Juntar os datasets com base no identificador do motorista
df = df_corridas.merge(df_veiculos_motoristas, on="id_motorista", how="left")

# Remover colunas desnecessárias (ex: IDs que não ajudam na previsão)
df = df.drop(columns=["id_corrida", "nome_motorista"])

# Definir a variável-alvo (1 = vencedor, 0 = não vencedor)
y = df["vencedor"]
X = df.drop(columns=["vencedor"])

# 📌 3. Pré-processamento dos dados

# Identificar colunas categóricas
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# Aplicar Label Encoding para variáveis categóricas
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    label_encoders[col] = le  # Guardar o encoder caso precise inverter depois

# 📌 4. Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 📌 5. Treinar os Modelos

# CatBoost
catboost_model = CatBoostClassifier(iterations=500, learning_rate=0.05, depth=6, verbose=0)
catboost_model.fit(X_train, y_train)

# XGBoost
xgb_model = XGBClassifier(n_estimators=500, learning_rate=0.05, max_depth=6, use_label_encoder=False, eval_metric="logloss")
xgb_model.fit(X_train, y_train)

# 📌 6. Fazer Previsões
catboost_preds = catboost_model.predict(X_test)
xgb_preds = xgb_model.predict(X_test)

# 📌 7. Avaliação dos Modelos
catboost_acc = accuracy_score(y_test, catboost_preds)
xgb_acc = accuracy_score(y_test, xgb_preds)

print(f"🏎 Acurácia CatBoost: {catboost_acc:.4f}")
print(f"🏎 Acurácia XGBoost: {xgb_acc:.4f}")
