import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn import model_selection, preprocessing, linear_model
import xgboost as xgb

data = pd.read_csv('data.csv')
data.columns = ['x', 'y']
train = data[:900]
test = data[900:]
train_X = train['x'].as_matrix()
train_y = train['y'].as_matrix()
test_X = test['x'].as_matrix()
test_y = test['y'].as_matrix()
train_X.shape = (len(train_X), 1)
train_y.shape = (len(train_y), 1)
test_X.shape = (len(test_X), 1)
test_y.shape = (len(test_y), 1)

pred_y_linear = [0] * len(test_X)

regr = linear_model.LinearRegression()
regr.fit(train_X, train_y) 
pred_y_linear = regr.predict(test_X)
# regr.score(test_X, pred_y_linear) 
def calc_rmse(predict):
	sigma = 0
	for i in range(0, len(predict)):
		diff = predict[i] - test_y[i];
		sigma += diff * diff
	return np.sqrt(sigma / len(predict))

print calc_rmse(pred_y_linear)



xgb_params = {
    'eta': 0.1,
    'max_depth': 5,
    'subsample': 0.7,
    'colsample_bytree': 1.0,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain = xgb.DMatrix(train_X, train_y)
model = xgb.train(xgb_params, dtrain, num_boost_round=100)
dtest = xgb.DMatrix(test_X)
pred_test_y = model.predict(dtest)

print calc_rmse(pred_test_y)
