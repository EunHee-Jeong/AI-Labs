import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings('ignore')

#
print("------------------------------------------------")
print("SelectKBest 와 GridSearchCV 를 활용한 feature 선택")

# 데이터 로드
housing = fetch_california_housing()
X = housing.data
y = housing.target

# 파이프라인 구성
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('feature_selection', SelectKBest(score_func=f_regression)),
    ('regressor', LinearRegression())
])

# GridSearchCV를 통해 선택할 best k개 feature 수를 튜닝
param_grid = {
    'feature_selection__k': [5, 7, 10, 12]  # 전체 feature 수는 13개
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid.fit(X, y)

print(f"Best number of features: {grid.best_params_['feature_selection__k']}")
print(f"Best CV R2 score: {grid.best_score_:.4f}")

# best estimator에서 선택된 feature 인덱스 확인
selected_mask = grid.best_estimator_.named_steps['feature_selection'].get_support()
selected_features = np.array(housing.feature_names)[selected_mask]

print("Selected features:", selected_features)
print("------------------------------------------------")

print()

print("RFE 와 GridSearchCV 를 활용한 feature 선택")
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge

pipeline = Pipeline([
    ('feature_selection', RFE(estimator=Ridge(), step=1)),
    ('regressor', Ridge())
])

param_grid = {
    'feature_selection__n_features_to_select': [5, 7, 10, 12],
    'regressor__alpha': [0.1, 1.0, 10.0]
}

grid = GridSearchCV(pipeline, param_grid, cv=5, scoring='r2')
grid.fit(X, y)

print("Best parameters:", grid.best_params_)

# best estimator에서 선택된 feature 인덱스 확인
selected_mask = grid.best_estimator_.named_steps['feature_selection'].get_support()
selected_features = np.array(housing.feature_names)[selected_mask]

print("Selected features:", selected_features)