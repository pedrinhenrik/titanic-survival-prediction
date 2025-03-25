# -*- coding: utf-8 -*-
"""
Created on Tue Mar 25 06:04:57 2025

@author: https://github.com/pedrinhenrik
"""

# Bibliotecas
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

titanic = sns.load_dataset('titanic')
titanic.head()

titanic.count()

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']
target = 'survived'

X = titanic[features]
y = titanic[target]

# Distribuição das classes
classe_counts = titanic['survived'].value_counts()
classe_percentual = titanic['survived'].value_counts(normalize=True) * 100

print("Contagem absoluta:")
print(classe_counts)
print("\nPercentual por classe:")
print(classe_percentual)

# Visualização
sns.countplot(x='survived', data=titanic)
plt.title('Distribuição das Classes - Sobrevivência')
plt.xlabel('Sobreviveu (0 = Não, 1 = Sim)')
plt.ylabel('Quantidade')
plt.show()

# Separando os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Verificando os tamanhos
print(f"Tamanho do conjunto de treino: {X_train.shape[0]} amostras")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]} amostras")

# Criação de pipelines de pré processamento
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

features = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'class', 'who', 'adult_male', 'alone']

# Definição de classes númericas e de categoria
numerical_features = ['age', 'sibsp', 'parch', 'fare', 'pclass']
categorical_features = ['sex', 'class', 'who', 'adult_male', 'alone']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Pipeline de classificação
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Definição de grade
param_grid = {
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [None, 10, 20],
    'classifier__min_samples_split': [2, 5]
}

# Estratégia de validação cruzada estratificada
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Grid search com validação cruzada
grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=cv,
    scoring='accuracy',
    n_jobs=-1  # usa todos os núcleos disponíveis
)

# Ajusta aos dados de treino
grid_search.fit(X_train, y_train)

# Resultados
print("Melhores parâmetros encontrados:")
print(grid_search.best_params_)

print("\nMelhor score de validação cruzada:")
print(grid_search.best_score_)

# Treinando o modelo diretamente com o pipeline
pipeline.fit(X_train, y_train)

from sklearn.metrics import classification_report, confusion_matrix

y_pred = pipeline.predict(X_test)

print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

print("Matriz de Confusão:")
print(confusion_matrix(y_test, y_pred))

# Obter o melhor modelo encontrado pela busca em grade
best_model = grid_search.best_estimator_

# Fazer previsões nos dados de teste
y_pred = best_model.predict(X_test)

# Avaliação das previsões
# Relatório de métricas
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred))

# Matriz de confusão com visualização
disp = ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
disp.ax_.set_title("Matriz de Confusão - Dados de Teste")
plt.show()

from sklearn.metrics import ConfusionMatrixDisplay

# Usando as previsões do melhor modelo nos dados de teste
ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

# Título e exibição
plt.title("Matriz de Confusão - Dados de Teste")
plt.show()

# Recuperar o melhor modelo do Grid Search
best_model = grid_search.best_estimator_

# Obter os nomes das variáveis categóricas transformadas pelo OneHotEncoder
categorical_encoded_names = list(
    best_model.named_steps['preprocessor']
              .named_transformers_['cat']
              .named_steps['onehot']
              .get_feature_names_out(categorical_features)
)

# Combinar com as variáveis numéricas
feature_names = numerical_features + categorical_encoded_names

# Importância das features no Random Forest
feature_importances = best_model.named_steps['classifier'].feature_importances_

# Criar um DataFrame com os nomes e importâncias
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': feature_importances
}).sort_values(by='Importance', ascending=False)

# Plotando grafico de importancia
plt.figure(figsize=(10, 6))
plt.barh(importance_df['Feature'], importance_df['Importance'], color='skyblue')
plt.gca().invert_yaxis()
plt.title('Features mais importantes para prever a sobrevivência')
plt.xlabel('Importância')
plt.tight_layout()
plt.show()

# Ver acuracia do conjunto
test_score = best_model.score(X_test, y_test)
print(f"\nAcurácia no conjunto de teste: {test_score:.2%}")