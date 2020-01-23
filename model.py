from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_squared_log_error, SCORERS


from yellowbrick.datasets import load_concrete
from yellowbrick.regressor import ResidualsPlot, PredictionError



import numpy as np
import pandas as pd


class ModelTry():

  def __init__(self, X, y, test_size=.3):
    self.X = X
    self.y = y

    if test_size < 1:
      self.X_train , self.X_test, self.y_train, self.y_test = train_test_split(X, y,
        test_size=test_size, random_state=42)




  def print_results(self, X_train, X_test, y_train, y_test, model):

    train_r2 = model.score(X_train, y_train)
    test_r2 = model.score(X_test, y_test)

    print('Training r^2 %:', round(train_r2*100,3))
    print('Testing r^2 %:', round(test_r2*100,3))
    print('Training MSE (mil):', mean_squared_error(y_train, model.predict(X_train))/1_000_000)
    print('Testing MSE (mil):', mean_squared_error(y_test, model.predict(X_test))/1_000_000)




  def data_transform(self, only_cont = False, poly = True):

    X_train , X_test, y_train, y_test = self.X_train , self.X_test, self.y_train, self.y_test

    # remove "object"-type features and y from `X`
    con_features = [ col for col in self.X.columns if self.X[col].dtype in ['int64','float64']]
    X_train_con = X_train.loc[:,con_features]
    X_test_con = X_test.loc[:,con_features]

    # Scale the train and test data
    scaler = StandardScaler()
    scaler.fit(X_train_con)

    X_train_sca = scaler.transform(X_train_con)
    X_test_sca = scaler.transform(X_test_con)

    if poly:
    # add in the polynomial interations for the cont. variables
      poly = PolynomialFeatures(2)
      poly.fit(X_train_sca)
      X_train_poly = poly.transform(X_train_sca)
      X_test_poly = poly.transform(X_test_sca)
      X_train_cols = poly.get_feature_names(X_train.columns)
    else:
      X_train_poly, X_test_poly = X_train_sca, X_test_sca
      X_train_cols = X_train_con.columns



    # Create X_cat which contains only the categorical variables
    cat_features = [ col for col in self.X.columns if self.X[col].dtype == np.object]
    X_train_cat = X_train.loc[:,cat_features]
    X_test_cat = X_test.loc[:,cat_features]


    # OneHotEncode Categorical variables
    ohe = OneHotEncoder(handle_unknown='ignore')
    ohe.fit(X_train_cat)

    X_train_ohe = ohe.transform(X_train_cat)
    X_test_ohe = ohe.transform(X_test_cat)

    columns = ohe.get_feature_names(input_features=X_train_cat.columns)
    cat_train_df = pd.DataFrame(X_train_ohe.todense(), columns=columns)
    cat_test_df = pd.DataFrame(X_test_ohe.todense(), columns=columns)

    X_train_all = pd.concat([pd.DataFrame(X_train_poly, columns=X_train_cols), cat_train_df], axis = 1)
    X_test_all = pd.concat([pd.DataFrame(X_test_poly, columns=X_train_cols), cat_test_df], axis = 1)

    if only_cont:
      X_train_tr, X_test_tr = X_train_sca, X_test_sca
    else:
      X_train_tr, X_test_tr = X_train_all, X_test_all

    return X_train_tr, X_test_tr, y_train, y_test, scaler, poly, ohe




  def run_model(self, model = 1, run_vis = False, poly = False, alpha = 1):
    """
    Intake: model code, run_visibility, poly, alpha level
    :return: best_alpha
    ------------------
    model code - (1 is LinearRegression, 3 is Lasso, 4 is Ridge)

    """
    only_cont = False
    lin_reg = LinearRegression()


    if model == 1:
      only_cont = True
    elif model == 3:
      lin_reg = Lasso(alpha)
    elif model == 4:
      lin_reg = Ridge(alpha)


    X_train , X_test, y_train, y_test, scaler, poly, ohe = self.data_transform(only_cont, poly)
    lin_reg.fit(X_train, y_train)

    self.print_results(X_train, X_test, y_train, y_test, lin_reg)

    model_coef = lin_reg.coef_
    if model != 1:
      model_coef = pd.DataFrame(model_coef, index=X_train.columns, columns=['coef_value'])
      model_coef['coef_abs'] = model_coef['coef_value'].apply(lambda x: np.abs(x))

    if run_vis:
      visualizer = ResidualsPlot(lin_reg)
      visualizer.fit(X_train, y_train)  # Fit the training data to the visualizer
      visualizer.score(X_test, y_test)  # Evaluate the model on the test data
      visualizer.show();

      vis_pred = PredictionError(lin_reg)
      vis_pred.fit(X_train, y_train)  # Fit the training data to the visualizer
      vis_pred.score(X_test, y_test)  # Evaluate the model on the test data
      vis_pred.show();


    return lin_reg, model_coef, X_train, y_train, scaler, poly, ohe


def grid_search(X_train, y_train, model, score = 'r2', cv=5):
  """
  Assumption: takes X, y train data to apply grid search to find best hyperparameter
  based on given score measure.
  :return: best_alpha

  """

  param_grid = [
    {'alpha' : np.linspace(-2, 2, 81)
    }
  ]

  gs_clf = GridSearchCV(model, param_grid = param_grid, scoring = score, cv=cv, return_train_score=True);
  best_clf = gs_clf.fit(X_train, y_train)

  bmodels = pd.DataFrame(best_clf.cv_results_).sort_values('rank_test_score')

  for i in range(0,3):
      bmodels[f'split{i}_drop'] = bmodels[f'split{i}_train_score']-bmodels[f'split{i}_test_score']

  best_alpha = best_clf.best_params_['alpha']

  return best_alpha, bmodels





