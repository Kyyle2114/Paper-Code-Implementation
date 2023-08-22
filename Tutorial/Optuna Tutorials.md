![Optuna_logo](./Images/Optuna.jpg)

# Optuna

머신러닝 / 딥러닝 모델의 하이퍼파라미터를 탐색하는 방법은 여러 가지가 있습니다. 사이킷런에서 제공하는 GridSearchCV, RandomizedSearchCV 등이 있으나, 이번 포스팅에서는 하이퍼파라미터 최적화를 위한 프레임워크, **Optuna**를 소개합니다. 

지난번 업로드한 **Revisiting Deep Learning Models for Tabular Data** 논문에서, 논문의 본문 중 "For most algorithms, we use the Optuna library (Akiba et al., 2019) to run Bayesian optimization (the Tree-Structured Parzen Estimator algorithm), which is reported to be superior to random search (Turner et al., 2021)." 이라는 내용이 있어 Optuna를 알게 되었고, 간단히 사용법을 정리해 보려 합니다.

사용법은 크게 어렵지 않으며, [공식 문서](https://optuna.org/)를 확인하시면 더 많은 정보를 확인하실 수 있습니다. 

# Default

하이퍼파라미터 튜닝 전후를 비교하기 위해, 우선 간단한 LGBMRegressor를 훈련해 보겠습니다. 데이터는 Colab을 실행하면 기본으로 제공해 주는 california_housing 데이터셋을 사용하였습니다. 

각 특성엔 별다른 전처리(StandardScaler 등)를 진행하지 않고, 모델이 median_house_value를 예측하도록 합니다. 회귀에서 다양한 평가 지표가 있겠지만, 간단히 mse와 모델의 score 값만 확인하도록 하겠습니다. 


```python
import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error

X_train = pd.read_csv('/content/sample_data/california_housing_train.csv')
X_test = pd.read_csv('/content/sample_data/california_housing_test.csv')

X_train.shape, X_test.shape
```
**출력**
((17000, 9), (3000, 9))


```python
# 데이터 예시 
X_train
```
**출력**
![](https://velog.velcdn.com/images/kyyle/post/0c701993-fbc0-4fb3-abba-0400a3f98141/image.png)

```python
# 결측치 존재하지 않음 
X_train.isnull().sum().sum(), X_test.isnull().sum().sum()
```
**출력**
(0, 0)

```python
X_train, y_train = X_train.iloc[:, :-1].values, X_train.median_house_value.values
X_test, y_test = X_test.iloc[:, :-1].values, X_test.median_house_value.values

lgbm = LGBMRegressor(random_state=21)
lgbm.fit(X_train, y_train)

y_pred = lgbm.predict(X_test)

error = mean_squared_error(y_test, y_pred)

print("MSE(default LGBM):", error)
print("Score(default LGBM):", lgbm.score(X_test, y_test))
```
**출력**
MSE(default LGBM): 2331923006.477407
Score(default LGBM): 0.8177017066139438

# Tuned

## MSE(Minimize)

이제 Optuna를 사용하여 하이퍼파라미터를 튜닝합니다. Colab을 사용하시는 경우, 우선 Optuna를 설치해야 합니다. 


```python
!pip install optuna
```

필요한 라이브러리를 불러오고, LGBMRegressor의 하이퍼파라미터를 탐색합니다. 

```python
import optuna
from optuna import Trial
from optuna.samplers import TPESampler
from sklearn.model_selection import train_test_split

X_train = pd.read_csv('/content/sample_data/california_housing_train.csv')
X_test = pd.read_csv('/content/sample_data/california_housing_test.csv')

X_train, y_train = X_train.iloc[:, :-1].values, X_train.median_house_value.values
X_test, y_test = X_test.iloc[:, :-1].values, X_test.median_house_value.values
```

최적화를 위한 함수 objective를 정의합니다. param에 탐색할 하이퍼파라미터와 그 범위를 지정합니다. 

suggest_int, suggest_float 외에도 suggest_categorical, suggest_uniform, suggest_discrete_uniform, suggest_loguniform 등의 함수가 존재합니다. 자세한 것은 공식 문서를 확인해 주세요. 

```python
def objective(trial, X, y):
  param = {"max_depth": trial.suggest_int("max_depth", 1, 20),
           "num_leaves": trial.suggest_int("num_leaves", 2, 256),
           "subsample": trial.suggest_float("subsample", 0.3, 1.0),
           "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
           "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),}

  model = LGBMRegressor(random_state=21, **param)

  # 20% for validation
  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

  model.fit(X_train, y_train)
  y_pred = model.predict(X_val)
  mse_score = mean_squared_error(y_val, y_pred)

  return mse_score
```

정의한 objective를 바탕으로 최적화를 수행합니다. 리턴하는 값이 mse_score, 즉 오차이므로 오차가 작아지는 방향으로 탐색해야 합니다. 이를 위해 direction='minimize'를 전달합니다. 30회의 탐색을 수행하겠습니다.

```python
study = optuna.create_study(study_name='LGBMRegressor', direction='minimize', sampler=TPESampler(seed=21))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)

print()
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)
```
**출력**
```
[I 2023-07-31 10:18:40,722] A new study created in memory with name: LGBMRegressor
[I 2023-07-31 10:18:40,832] Trial 0 finished with value: 5595256700.657137 and parameters: {'max_depth': 1, 'num_leaves': 75, 'subsample': 0.8046764427818609, 'subsample_freq': 1, 'min_child_samples': 24}. Best is trial 0 with value: 5595256700.657137.
[I 2023-07-31 10:18:40,953] Trial 1 finished with value: 3590148420.763507 and parameters: {'max_depth': 2, 'num_leaves': 79, 'subsample': 0.7647372062372899, 'subsample_freq': 4, 'min_child_samples': 61}. Best is trial 1 with value: 3590148420.763507.
[I 2023-07-31 10:18:41,071] Trial 2 finished with value: 3767313437.1282096 and parameters: {'max_depth': 2, 'num_leaves': 223, 'subsample': 0.3932683634762234, 'subsample_freq': 2, 'min_child_samples': 52}. Best is trial 1 with value: 3590148420.763507.
[I 2023-07-31 10:18:42,269] Trial 3 finished with value: 2338607537.4790854 and parameters: {'max_depth': 18, 'num_leaves': 195, 'subsample': 0.9793395877444102, 'subsample_freq': 8, 'min_child_samples': 41}. Best is trial 3 with value: 2338607537.4790854.
[I 2023-07-31 10:18:42,577] Trial 4 finished with value: 2402433367.394679 and parameters: {'max_depth': 9, 'num_leaves': 183, 'subsample': 0.4894688408711936, 'subsample_freq': 9, 'min_child_samples': 92}. Best is trial 3 with value: 2338607537.4790854.
[I 2023-07-31 10:18:43,458] Trial 5 finished with value: 2307347500.980729 and parameters: {'max_depth': 16, 'num_leaves': 133, 'subsample': 0.4174850356550551, 'subsample_freq': 3, 'min_child_samples': 32}. Best is trial 5 with value: 2307347500.980729.
[I 2023-07-31 10:18:43,776] Trial 6 finished with value: 2462636598.7491384 and parameters: {'max_depth': 7, 'num_leaves': 119, 'subsample': 0.6810578516203463, 'subsample_freq': 3, 'min_child_samples': 81}. Best is trial 5 with value: 2307347500.980729.
[I 2023-07-31 10:18:44,229] Trial 7 finished with value: 2316151273.839512 and parameters: {'max_depth': 15, 'num_leaves': 218, 'subsample': 0.48726526320745056, 'subsample_freq': 7, 'min_child_samples': 67}. Best is trial 5 with value: 2307347500.980729.
[I 2023-07-31 10:18:47,758] Trial 8 finished with value: 2240259489.2824745 and parameters: {'max_depth': 8, 'num_leaves': 106, 'subsample': 0.867110284644846, 'subsample_freq': 7, 'min_child_samples': 91}. Best is trial 8 with value: 2240259489.2824745.
[I 2023-07-31 10:18:49,105] Trial 9 finished with value: 2383484183.6954217 and parameters: {'max_depth': 10, 'num_leaves': 213, 'subsample': 0.9012226298177672, 'subsample_freq': 8, 'min_child_samples': 90}. Best is trial 8 with value: 2240259489.2824745.
[I 2023-07-31 10:18:49,897] Trial 10 finished with value: 2569804823.3527393 and parameters: {'max_depth': 6, 'num_leaves': 16, 'subsample': 0.6070518179740391, 'subsample_freq': 6, 'min_child_samples': 7}. Best is trial 8 with value: 2240259489.2824745.
[I 2023-07-31 10:18:54,904] Trial 11 finished with value: 2576790064.605227 and parameters: {'max_depth': 14, 'num_leaves': 137, 'subsample': 0.3155218884927118, 'subsample_freq': 5, 'min_child_samples': 34}. Best is trial 8 with value: 2240259489.2824745.
[I 2023-07-31 10:19:04,763] Trial 12 finished with value: 2293313963.811297 and parameters: {'max_depth': 20, 'num_leaves': 134, 'subsample': 0.6053987370060777, 'subsample_freq': 10, 'min_child_samples': 17}. Best is trial 8 with value: 2240259489.2824745.
[I 2023-07-31 10:19:08,281] Trial 13 finished with value: 2222724118.4966326 and parameters: {'max_depth': 20, 'num_leaves': 92, 'subsample': 0.6261293977333131, 'subsample_freq': 10, 'min_child_samples': 5}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:09,265] Trial 14 finished with value: 2340014877.1865854 and parameters: {'max_depth': 13, 'num_leaves': 59, 'subsample': 0.8018082313235951, 'subsample_freq': 10, 'min_child_samples': 100}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:09,537] Trial 15 finished with value: 2478170653.731526 and parameters: {'max_depth': 5, 'num_leaves': 27, 'subsample': 0.9948400331659695, 'subsample_freq': 6, 'min_child_samples': 74}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:10,024] Trial 16 finished with value: 2390005488.8445864 and parameters: {'max_depth': 11, 'num_leaves': 106, 'subsample': 0.6745794515866604, 'subsample_freq': 8, 'min_child_samples': 46}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:10,872] Trial 17 finished with value: 2242792631.6559186 and parameters: {'max_depth': 12, 'num_leaves': 163, 'subsample': 0.8668028604170925, 'subsample_freq': 9, 'min_child_samples': 60}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:13,056] Trial 18 finished with value: 2296830985.594768 and parameters: {'max_depth': 8, 'num_leaves': 100, 'subsample': 0.7275223463529371, 'subsample_freq': 5, 'min_child_samples': 14}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:13,596] Trial 19 finished with value: 2290046671.230836 and parameters: {'max_depth': 20, 'num_leaves': 247, 'subsample': 0.5947467973288103, 'subsample_freq': 7, 'min_child_samples': 75}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:13,809] Trial 20 finished with value: 2680240305.366035 and parameters: {'max_depth': 4, 'num_leaves': 44, 'subsample': 0.7222693213658331, 'subsample_freq': 10, 'min_child_samples': 27}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:14,413] Trial 21 finished with value: 2230503963.7173557 and parameters: {'max_depth': 12, 'num_leaves': 161, 'subsample': 0.8815095259590999, 'subsample_freq': 9, 'min_child_samples': 57}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:15,174] Trial 22 finished with value: 2271786082.160187 and parameters: {'max_depth': 17, 'num_leaves': 170, 'subsample': 0.8795547879562502, 'subsample_freq': 9, 'min_child_samples': 43}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:15,597] Trial 23 finished with value: 2351965469.5089245 and parameters: {'max_depth': 9, 'num_leaves': 154, 'subsample': 0.9323825722823451, 'subsample_freq': 7, 'min_child_samples': 84}. Best is trial 13 with value: 2222724118.4966326.
[I 2023-07-31 10:19:16,112] Trial 24 finished with value: 2170161457.2279835 and parameters: {'max_depth': 11, 'num_leaves': 88, 'subsample': 0.8382206515474253, 'subsample_freq': 9, 'min_child_samples': 61}. Best is trial 24 with value: 2170161457.2279835.
[I 2023-07-31 10:19:16,648] Trial 25 finished with value: 2218372487.2746024 and parameters: {'max_depth': 12, 'num_leaves': 88, 'subsample': 0.9489483147375105, 'subsample_freq': 9, 'min_child_samples': 54}. Best is trial 24 with value: 2170161457.2279835.
[I 2023-07-31 10:19:17,201] Trial 26 finished with value: 2161086350.932649 and parameters: {'max_depth': 14, 'num_leaves': 82, 'subsample': 0.8146928767601797, 'subsample_freq': 10, 'min_child_samples': 51}. Best is trial 26 with value: 2161086350.932649.
[I 2023-07-31 10:19:17,660] Trial 27 finished with value: 2260759733.8170314 and parameters: {'max_depth': 14, 'num_leaves': 54, 'subsample': 0.939256440739107, 'subsample_freq': 8, 'min_child_samples': 69}. Best is trial 26 with value: 2161086350.932649.
[I 2023-07-31 10:19:17,791] Trial 28 finished with value: 5626705319.81021 and parameters: {'max_depth': 11, 'num_leaves': 2, 'subsample': 0.8223457048802809, 'subsample_freq': 9, 'min_child_samples': 50}. Best is trial 26 with value: 2161086350.932649.
[I 2023-07-31 10:19:18,362] Trial 29 finished with value: 2189353404.8122654 and parameters: {'max_depth': 13, 'num_leaves': 81, 'subsample': 0.8347725594891964, 'subsample_freq': 10, 'min_child_samples': 36}. Best is trial 26 with value: 2161086350.932649.

Best Score: 2161086350.932649
Best trial: {'max_depth': 14, 'num_leaves': 82, 'subsample': 0.8146928767601797, 'subsample_freq': 10, 'min_child_samples': 51}
```


최적화 과정과 하이퍼파라미터의 중요도를 확인할 수 있습니다.


```python
optuna.visualization.plot_optimization_history(study)
```
**출력**
![](https://velog.velcdn.com/images/kyyle/post/2db3a107-5f83-45cb-b0d3-69666c47361f/image.png)



```python
optuna.visualization.plot_param_importances(study)
```
**출력**
![](https://velog.velcdn.com/images/kyyle/post/6909e73a-8746-4692-95e7-36c77c125909/image.png)


최적의 하이퍼파라미터 조합으로 모델을 훈련시킨 뒤, mse와 score가 어떻게 변하는지 확인합니다.

```python
lgbm_tuned = LGBMRegressor(random_state=21, **study.best_trial.params)
lgbm_tuned.fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)

error = mean_squared_error(y_test, y_pred)

print("MSE(Tuned LGBM):", error)
print("Score(Tuned LGBM):", lgbm_tuned.score(X_test, y_test))
```
**출력**
MSE(Tuned LGBM): 2291835215.062972
Score(Tuned LGBM): 0.8208355733583297

## Score(Maximize)

혹은, 다음과 같이 score를 리턴한 뒤 score를 최대화하는 방향으로 탐색을 진행할 수 있습니다. 

```python
def objective(trial, X, y):
  param = {"max_depth": trial.suggest_int("max_depth", 1, 20),
           "num_leaves": trial.suggest_int("num_leaves", 2, 256),
           "subsample": trial.suggest_float("subsample", 0.3, 1.0),
           "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
           "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),}

  model = LGBMRegressor(random_state=21, **param)

  X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

  model.fit(X_train, y_train)
  score = model.score(X_val, y_val)

  return score

# direction='maximize'
study = optuna.create_study(study_name='LGBMRegressor', direction='maximize', sampler=TPESampler(seed=21))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)

print()
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)
```
**출력**
```
[I 2023-07-31 10:19:19,975] A new study created in memory with name: LGBMRegressor
[I 2023-07-31 10:19:20,082] Trial 0 finished with value: 0.5939765922648081 and parameters: {'max_depth': 1, 'num_leaves': 75, 'subsample': 0.8046764427818609, 'subsample_freq': 1, 'min_child_samples': 24}. Best is trial 0 with value: 0.5939765922648081.
[I 2023-07-31 10:19:20,219] Trial 1 finished with value: 0.7373185011830732 and parameters: {'max_depth': 2, 'num_leaves': 79, 'subsample': 0.7647372062372899, 'subsample_freq': 4, 'min_child_samples': 61}. Best is trial 1 with value: 0.7373185011830732.
[I 2023-07-31 10:19:20,328] Trial 2 finished with value: 0.7151051447992078 and parameters: {'max_depth': 2, 'num_leaves': 223, 'subsample': 0.3932683634762234, 'subsample_freq': 2, 'min_child_samples': 52}. Best is trial 1 with value: 0.7373185011830732.
[I 2023-07-31 10:19:21,474] Trial 3 finished with value: 0.8463342137384563 and parameters: {'max_depth': 18, 'num_leaves': 195, 'subsample': 0.9793395877444102, 'subsample_freq': 8, 'min_child_samples': 41}. Best is trial 3 with value: 0.8463342137384563.
[I 2023-07-31 10:19:21,784] Trial 4 finished with value: 0.8082077602394131 and parameters: {'max_depth': 9, 'num_leaves': 183, 'subsample': 0.4894688408711936, 'subsample_freq': 9, 'min_child_samples': 92}. Best is trial 3 with value: 0.8463342137384563.
[I 2023-07-31 10:19:22,387] Trial 5 finished with value: 0.8194834758951564 and parameters: {'max_depth': 16, 'num_leaves': 133, 'subsample': 0.4174850356550551, 'subsample_freq': 3, 'min_child_samples': 32}. Best is trial 3 with value: 0.8463342137384563.
[I 2023-07-31 10:19:22,679] Trial 6 finished with value: 0.824332739258592 and parameters: {'max_depth': 7, 'num_leaves': 119, 'subsample': 0.6810578516203463, 'subsample_freq': 3, 'min_child_samples': 81}. Best is trial 3 with value: 0.8463342137384563.
[I 2023-07-31 10:19:26,890] Trial 7 finished with value: 0.8205850736737004 and parameters: {'max_depth': 15, 'num_leaves': 218, 'subsample': 0.48726526320745056, 'subsample_freq': 7, 'min_child_samples': 67}. Best is trial 3 with value: 0.8463342137384563.
[I 2023-07-31 10:19:28,116] Trial 8 finished with value: 0.8269941908171182 and parameters: {'max_depth': 8, 'num_leaves': 106, 'subsample': 0.867110284644846, 'subsample_freq': 7, 'min_child_samples': 91}. Best is trial 3 with value: 0.8463342137384563.
[I 2023-07-31 10:19:29,656] Trial 9 finished with value: 0.8307739071991467 and parameters: {'max_depth': 10, 'num_leaves': 213, 'subsample': 0.9012226298177672, 'subsample_freq': 8, 'min_child_samples': 90}. Best is trial 3 with value: 0.8463342137384563.
[I 2023-07-31 10:19:39,997] Trial 10 finished with value: 0.846764596096845 and parameters: {'max_depth': 19, 'num_leaves': 172, 'subsample': 0.9687414314720535, 'subsample_freq': 10, 'min_child_samples': 5}. Best is trial 10 with value: 0.846764596096845.
[I 2023-07-31 10:19:41,344] Trial 11 finished with value: 0.8246611910919582 and parameters: {'max_depth': 20, 'num_leaves': 20, 'subsample': 0.9988688269981433, 'subsample_freq': 10, 'min_child_samples': 9}. Best is trial 10 with value: 0.846764596096845.
[I 2023-07-31 10:19:44,461] Trial 12 finished with value: 0.8472850368743177 and parameters: {'max_depth': 20, 'num_leaves': 166, 'subsample': 0.9941647837170454, 'subsample_freq': 10, 'min_child_samples': 6}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:45,321] Trial 13 finished with value: 0.8333005334941017 and parameters: {'max_depth': 14, 'num_leaves': 161, 'subsample': 0.9084200293624082, 'subsample_freq': 10, 'min_child_samples': 5}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:46,442] Trial 14 finished with value: 0.8155880586718758 and parameters: {'max_depth': 20, 'num_leaves': 245, 'subsample': 0.995761572995347, 'subsample_freq': 5, 'min_child_samples': 18}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:47,178] Trial 15 finished with value: 0.8379078863850803 and parameters: {'max_depth': 13, 'num_leaves': 153, 'subsample': 0.7714380073095753, 'subsample_freq': 10, 'min_child_samples': 18}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:47,932] Trial 16 finished with value: 0.8456967302834562 and parameters: {'max_depth': 17, 'num_leaves': 171, 'subsample': 0.690278590934378, 'subsample_freq': 6, 'min_child_samples': 36}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:50,714] Trial 17 finished with value: 0.8389232660505666 and parameters: {'max_depth': 12, 'num_leaves': 249, 'subsample': 0.8668028604170925, 'subsample_freq': 9, 'min_child_samples': 13}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:51,544] Trial 18 finished with value: 0.8292229323068254 and parameters: {'max_depth': 18, 'num_leaves': 139, 'subsample': 0.6043628525274626, 'subsample_freq': 8, 'min_child_samples': 27}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:51,867] Trial 19 finished with value: 0.8251875812697315 and parameters: {'max_depth': 6, 'num_leaves': 84, 'subsample': 0.9362314832626815, 'subsample_freq': 9, 'min_child_samples': 43}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:52,159] Trial 20 finished with value: 0.805694867687349 and parameters: {'max_depth': 20, 'num_leaves': 20, 'subsample': 0.8455329446030667, 'subsample_freq': 5, 'min_child_samples': 5}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:52,972] Trial 21 finished with value: 0.8360074437101335 and parameters: {'max_depth': 18, 'num_leaves': 194, 'subsample': 0.9617919869309447, 'subsample_freq': 8, 'min_child_samples': 44}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:53,941] Trial 22 finished with value: 0.8272993348986579 and parameters: {'max_depth': 18, 'num_leaves': 195, 'subsample': 0.9436115339886938, 'subsample_freq': 10, 'min_child_samples': 19}. Best is trial 12 with value: 0.8472850368743177.
[I 2023-07-31 10:19:54,647] Trial 23 finished with value: 0.8550398696369568 and parameters: {'max_depth': 16, 'num_leaves': 172, 'subsample': 0.9897273690346913, 'subsample_freq': 9, 'min_child_samples': 68}. Best is trial 23 with value: 0.8550398696369568.
[I 2023-07-31 10:19:55,324] Trial 24 finished with value: 0.8320267775446628 and parameters: {'max_depth': 16, 'num_leaves': 149, 'subsample': 0.9208545441035725, 'subsample_freq': 9, 'min_child_samples': 74}. Best is trial 23 with value: 0.8550398696369568.
[I 2023-07-31 10:19:55,921] Trial 25 finished with value: 0.8201682655709802 and parameters: {'max_depth': 12, 'num_leaves': 174, 'subsample': 0.9868705857736527, 'subsample_freq': 7, 'min_child_samples': 53}. Best is trial 23 with value: 0.8550398696369568.
[I 2023-07-31 10:19:56,559] Trial 26 finished with value: 0.8364351853512805 and parameters: {'max_depth': 20, 'num_leaves': 96, 'subsample': 0.832591003227128, 'subsample_freq': 10, 'min_child_samples': 75}. Best is trial 23 with value: 0.8550398696369568.
[I 2023-07-31 10:19:57,038] Trial 27 finished with value: 0.8345168123184077 and parameters: {'max_depth': 15, 'num_leaves': 55, 'subsample': 0.9020183811177751, 'subsample_freq': 9, 'min_child_samples': 61}. Best is trial 23 with value: 0.8550398696369568.
[I 2023-07-31 10:19:57,771] Trial 28 finished with value: 0.8298180363404931 and parameters: {'max_depth': 19, 'num_leaves': 119, 'subsample': 0.9506102455300931, 'subsample_freq': 6, 'min_child_samples': 12}. Best is trial 23 with value: 0.8550398696369568.
[I 2023-07-31 10:19:58,563] Trial 29 finished with value: 0.8182428989648495 and parameters: {'max_depth': 16, 'num_leaves': 164, 'subsample': 0.7785843105810236, 'subsample_freq': 10, 'min_child_samples': 29}. Best is trial 23 with value: 0.8550398696369568.

Best Score: 0.8550398696369568
Best trial: {'max_depth': 16, 'num_leaves': 172, 'subsample': 0.9897273690346913, 'subsample_freq': 9, 'min_child_samples': 68}
```

```python
lgbm_tuned = LGBMRegressor(random_state=21, **study.best_trial.params)
lgbm_tuned.fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)

error = mean_squared_error(y_test, y_pred)

print("MSE(Tuned LGBM):", error)
print("Score(Tuned LGBM):", lgbm_tuned.score(X_test, y_test))
```
**출력**
MSE(Tuned LGBM): 2262972240.8683677
Score(Tuned LGBM): 0.8230919389943763

## KFold 

지금까지는 1회 탐색 당 전체 훈련 데이터에서 20%를 검증 데이터로 사용하였습니다. 다음과 같은 방법을 사용하여 1회 탐색 당 k-fold 교차 검증을 사용할 수 있습니다. 

이번 포스팅의 경우 회귀 문제이기에 KFold를 사용하였으며, 분류의 경우 StratifiedKFold 등 적절한 CV splitter를 사용하시면 되겠습니다. 

```python
from sklearn.model_selection import KFold

X_train = pd.read_csv('/content/sample_data/california_housing_train.csv')
X_test = pd.read_csv('/content/sample_data/california_housing_test.csv')

X_train, y_train = X_train.iloc[:, :-1], X_train.median_house_value
X_test, y_test = X_test.iloc[:, :-1], X_test.median_house_value

def objective(trial, X, y):
  param = {"max_depth": trial.suggest_int("max_depth", 1, 20),
           "num_leaves": trial.suggest_int("num_leaves", 2, 256),
           "subsample": trial.suggest_float("subsample", 0.3, 1.0),
           "subsample_freq": trial.suggest_int("subsample_freq", 1, 10),
           "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),}

  cv = KFold(n_splits=5, shuffle=True)
  cv_scores = []

  for idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        model = LGBMRegressor(random_state=21, **param)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)
        cv_scores.append(mean_squared_error(y_val, y_pred))

  mse_score = np.mean(cv_scores)
  return mse_score
```

```python
study = optuna.create_study(study_name='LGBMRegressor_KFold', direction='minimize', sampler=TPESampler(seed=21))
study.optimize(lambda trial: objective(trial, X_train, y_train), n_trials=30)

print()
print("Best Score:", study.best_value)
print("Best trial:", study.best_trial.params)
```
**출력**
```
[I 2023-07-31 10:19:59,464] A new study created in memory with name: LGBMRegressor_KFold
[I 2023-07-31 10:19:59,883] Trial 0 finished with value: 5442107872.6467285 and parameters: {'max_depth': 1, 'num_leaves': 75, 'subsample': 0.8046764427818609, 'subsample_freq': 1, 'min_child_samples': 24}. Best is trial 0 with value: 5442107872.6467285.
[I 2023-07-31 10:20:00,473] Trial 1 finished with value: 3598939857.77954 and parameters: {'max_depth': 2, 'num_leaves': 79, 'subsample': 0.7647372062372899, 'subsample_freq': 4, 'min_child_samples': 61}. Best is trial 1 with value: 3598939857.77954.
[I 2023-07-31 10:20:01,067] Trial 2 finished with value: 3581939742.051656 and parameters: {'max_depth': 2, 'num_leaves': 223, 'subsample': 0.3932683634762234, 'subsample_freq': 2, 'min_child_samples': 52}. Best is trial 2 with value: 3581939742.051656.
[I 2023-07-31 10:20:07,023] Trial 3 finished with value: 2189918097.714978 and parameters: {'max_depth': 18, 'num_leaves': 195, 'subsample': 0.9793395877444102, 'subsample_freq': 8, 'min_child_samples': 41}. Best is trial 3 with value: 2189918097.714978.
[I 2023-07-31 10:20:08,446] Trial 4 finished with value: 2466550995.3818603 and parameters: {'max_depth': 9, 'num_leaves': 183, 'subsample': 0.4894688408711936, 'subsample_freq': 9, 'min_child_samples': 92}. Best is trial 3 with value: 2189918097.714978.
[I 2023-07-31 10:20:11,123] Trial 5 finished with value: 2326803173.9647207 and parameters: {'max_depth': 16, 'num_leaves': 133, 'subsample': 0.4174850356550551, 'subsample_freq': 3, 'min_child_samples': 32}. Best is trial 3 with value: 2189918097.714978.
[I 2023-07-31 10:20:12,505] Trial 6 finished with value: 2417012540.3175125 and parameters: {'max_depth': 7, 'num_leaves': 119, 'subsample': 0.6810578516203463, 'subsample_freq': 3, 'min_child_samples': 81}. Best is trial 3 with value: 2189918097.714978.
[I 2023-07-31 10:20:16,572] Trial 7 finished with value: 2384762935.6466856 and parameters: {'max_depth': 15, 'num_leaves': 218, 'subsample': 0.48726526320745056, 'subsample_freq': 7, 'min_child_samples': 67}. Best is trial 3 with value: 2189918097.714978.
[I 2023-07-31 10:20:18,213] Trial 8 finished with value: 2380953073.1223745 and parameters: {'max_depth': 8, 'num_leaves': 106, 'subsample': 0.867110284644846, 'subsample_freq': 7, 'min_child_samples': 91}. Best is trial 3 with value: 2189918097.714978.
[I 2023-07-31 10:20:20,211] Trial 9 finished with value: 2330832414.754014 and parameters: {'max_depth': 10, 'num_leaves': 213, 'subsample': 0.9012226298177672, 'subsample_freq': 8, 'min_child_samples': 90}. Best is trial 3 with value: 2189918097.714978.
[I 2023-07-31 10:20:24,574] Trial 10 finished with value: 2180980278.7969046 and parameters: {'max_depth': 19, 'num_leaves': 172, 'subsample': 0.9687414314720535, 'subsample_freq': 10, 'min_child_samples': 5}. Best is trial 10 with value: 2180980278.7969046.
[I 2023-07-31 10:20:27,801] Trial 11 finished with value: 2378250147.24029 and parameters: {'max_depth': 20, 'num_leaves': 20, 'subsample': 0.9988688269981433, 'subsample_freq': 10, 'min_child_samples': 9}. Best is trial 10 with value: 2180980278.7969046.
[I 2023-07-31 10:20:32,082] Trial 12 finished with value: 2166031828.166163 and parameters: {'max_depth': 20, 'num_leaves': 166, 'subsample': 0.9941647837170454, 'subsample_freq': 10, 'min_child_samples': 6}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:20:36,230] Trial 13 finished with value: 2201647456.354946 and parameters: {'max_depth': 14, 'num_leaves': 161, 'subsample': 0.9084200293624082, 'subsample_freq': 10, 'min_child_samples': 5}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:20:43,569] Trial 14 finished with value: 2219554312.343604 and parameters: {'max_depth': 20, 'num_leaves': 245, 'subsample': 0.995761572995347, 'subsample_freq': 5, 'min_child_samples': 18}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:20:47,005] Trial 15 finished with value: 2237517492.215838 and parameters: {'max_depth': 13, 'num_leaves': 153, 'subsample': 0.7714380073095753, 'subsample_freq': 10, 'min_child_samples': 18}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:20:52,618] Trial 16 finished with value: 2252514345.0862594 and parameters: {'max_depth': 17, 'num_leaves': 171, 'subsample': 0.690278590934378, 'subsample_freq': 6, 'min_child_samples': 36}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:20:56,832] Trial 17 finished with value: 2239481015.856851 and parameters: {'max_depth': 12, 'num_leaves': 249, 'subsample': 0.8668028604170925, 'subsample_freq': 9, 'min_child_samples': 13}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:21:00,241] Trial 18 finished with value: 2263231329.9307537 and parameters: {'max_depth': 18, 'num_leaves': 139, 'subsample': 0.6043628525274626, 'subsample_freq': 8, 'min_child_samples': 27}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:21:01,671] Trial 19 finished with value: 2335863993.2346125 and parameters: {'max_depth': 6, 'num_leaves': 84, 'subsample': 0.9362314832626815, 'subsample_freq': 9, 'min_child_samples': 43}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:21:04,899] Trial 20 finished with value: 2411666865.702531 and parameters: {'max_depth': 20, 'num_leaves': 20, 'subsample': 0.8455329446030667, 'subsample_freq': 5, 'min_child_samples': 5}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:21:08,757] Trial 21 finished with value: 2187874745.9446306 and parameters: {'max_depth': 18, 'num_leaves': 194, 'subsample': 0.9617919869309447, 'subsample_freq': 8, 'min_child_samples': 44}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:21:11,769] Trial 22 finished with value: 2212285504.4707527 and parameters: {'max_depth': 18, 'num_leaves': 202, 'subsample': 0.9406540259101459, 'subsample_freq': 10, 'min_child_samples': 72}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:21:17,475] Trial 23 finished with value: 2206672331.376862 and parameters: {'max_depth': 16, 'num_leaves': 177, 'subsample': 0.9467286566555275, 'subsample_freq': 9, 'min_child_samples': 52}. Best is trial 12 with value: 2166031828.166163.
[I 2023-07-31 10:21:21,446] Trial 24 finished with value: 2163017573.1307216 and parameters: {'max_depth': 19, 'num_leaves': 149, 'subsample': 0.9933785263233815, 'subsample_freq': 7, 'min_child_samples': 21}. Best is trial 24 with value: 2163017573.1307216.
[I 2023-07-31 10:21:25,466] Trial 25 finished with value: 2177964447.6762953 and parameters: {'max_depth': 20, 'num_leaves': 153, 'subsample': 0.8967128503263655, 'subsample_freq': 7, 'min_child_samples': 16}. Best is trial 24 with value: 2163017573.1307216.
[I 2023-07-31 10:21:32,044] Trial 26 finished with value: 2180588239.411641 and parameters: {'max_depth': 15, 'num_leaves': 145, 'subsample': 0.897504519442375, 'subsample_freq': 6, 'min_child_samples': 17}. Best is trial 24 with value: 2163017573.1307216.
[I 2023-07-31 10:21:34,814] Trial 27 finished with value: 2252408664.353419 and parameters: {'max_depth': 12, 'num_leaves': 120, 'subsample': 0.8334963950774881, 'subsample_freq': 7, 'min_child_samples': 26}. Best is trial 24 with value: 2163017573.1307216.
[I 2023-07-31 10:21:35,991] Trial 28 finished with value: 2448371696.7732024 and parameters: {'max_depth': 5, 'num_leaves': 105, 'subsample': 0.9076143330359145, 'subsample_freq': 6, 'min_child_samples': 21}. Best is trial 24 with value: 2163017573.1307216.
[I 2023-07-31 10:21:38,950] Trial 29 finished with value: 2219831590.7570343 and parameters: {'max_depth': 20, 'num_leaves': 100, 'subsample': 0.8047325827847878, 'subsample_freq': 7, 'min_child_samples': 31}. Best is trial 24 with value: 2163017573.1307216.

Best Score: 2163017573.1307216
Best trial: {'max_depth': 19, 'num_leaves': 149, 'subsample': 0.9933785263233815, 'subsample_freq': 7, 'min_child_samples': 21}
```

```python
lgbm_tuned = LGBMRegressor(random_state=21, **study.best_trial.params)
lgbm_tuned.fit(X_train, y_train)

y_pred = lgbm_tuned.predict(X_test)

error = mean_squared_error(y_test, y_pred)

print("MSE(Tuned LGBM):", error)
print("Score(Tuned LGBM):", lgbm_tuned.score(X_test, y_test))
```
**출력**
MSE(Tuned LGBM): 2192256796.0408244
Score(Tuned LGBM): 0.8286201253334143

***

이것으로 Optuna 기초 사용 방법을 알아보았습니다. 굉장히 기초적인 부분만 정리한 것이므로, 추가적인 정보를 원하신다면 [공식 문서](https://optuna.org/)를 확인해 주세요.
