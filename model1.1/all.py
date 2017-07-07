import numpy as np
import pandas as pd
from sklearn import model_selection, preprocessing
import xgboost as xgb
import datetime


train_df = pd.read_csv('../input/train.csv')
test_df = pd.read_csv('../input/test.csv')

# brings error down a lot by removing extreme price per sqm
train_full_sq_mean = 54.21
test_fill_sq_mean = 53.70
train_df.loc[train_df.full_sq == 0, 'full_sq'] = train_full_sq_mean
test_df.loc[test_df.full_sq == 0, 'full_sq'] = test_fill_sq_mean
train_df = train_df[train_df.price_doc/train_df.full_sq <= 600000]
train_df = train_df[train_df.price_doc/train_df.full_sq >= 10000]
train_df = train_df.reset_index()

train_df['yearmonth'] = train_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
test_df['yearmonth'] = test_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
train_df['ppa'] = train_df['price_doc'] / train_df['full_sq']

g_ppa_df = train_df.groupby('yearmonth')['ppa'].aggregate(np.mean).reset_index()


macro_df = pd.read_csv("../input/macro.csv")
macro_df_ym_1 = macro_df['timestamp'].apply(lambda x: x[:4]+x[5:7])
macro_df = macro_df.drop(['timestamp'], axis=1)
for f in macro_df.columns:
    if macro_df[f].dtype=='object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(macro_df[f].values)) 
        macro_df[f] = lbl.transform(list(macro_df[f].values))
# print macro_df.dtypes.reset_index()
macro_df['yearmonth'] = macro_df_ym_1
g_macro_data = macro_df.groupby('yearmonth').aggregate(np.mean).reset_index()
g_macro_data_1 = g_macro_data.loc[19:65,:].reset_index() # train
g_macro_data_2 = g_macro_data.loc[66:76,:].reset_index() # test

train_y = g_ppa_df['ppa']
train_X = g_macro_data_1.drop(['index', 'yearmonth'], axis=1)
test_X = g_macro_data_2.drop(['index', 'yearmonth'], axis=1)


print('Running Model 1_1...')
xgb_params = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.5,
    'colsample_bytree': 0.7,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}
dtrain = xgb.DMatrix(train_X, train_y, feature_names=train_X.columns.values)
dtest = xgb.DMatrix(test_X, feature_names=test_X.columns.values)

model = xgb.train(dict(xgb_params, silent=0), dtrain, num_boost_round=100)
pred_test_y = model.predict(dtest)


#clean data
print('Data Clean...')
bad_index = train_df[train_df.life_sq > train_df.full_sq].index
train_df.loc[bad_index, "life_sq"] = np.NaN
equal_index = [601,1896,2791]
test_df.loc[equal_index, "life_sq"] = test_df.loc[equal_index, "full_sq"]
bad_index = test_df[test_df.life_sq > test_df.full_sq].index
test_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = train_df[train_df.life_sq < 5].index
train_df.loc[bad_index, "life_sq"] = np.NaN
bad_index = test_df[test_df.life_sq < 5].index
test_df.loc[bad_index, "life_sq"] = np.NaN
# bad_index = train_df[train_df.full_sq < 5].index
# train_df.loc[bad_index, "full_sq"] = np.NaN
# bad_index = test_df[test_df.full_sq < 5].index
# test_df.loc[bad_index, "full_sq"] = np.NaN
kitch_is_build_year = [13117]
train_df.loc[kitch_is_build_year, "build_year"] = train_df.loc[kitch_is_build_year, "kitch_sq"]
bad_index = train_df[train_df.kitch_sq >= train_df.life_sq].index
train_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test_df[test_df.kitch_sq >= test_df.life_sq].index
test_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = train_df[(train_df.kitch_sq == 0).values + (train_df.kitch_sq == 1).values].index
train_df.loc[bad_index, "kitch_sq"] = np.NaN
bad_index = test_df[(test_df.kitch_sq == 0).values + (test_df.kitch_sq == 1).values].index
test_df.loc[bad_index, "kitch_sq"] = np.NaN
# bad_index = train_df[(train_df.full_sq > 210) & (train_df.life_sq / train_df.full_sq < 0.3)].index
# train_df.loc[bad_index, "full_sq"] = np.NaN
# bad_index = test_df[(test_df.full_sq > 150) & (test_df.life_sq / test_df.full_sq < 0.3)].index
# test_df.loc[bad_index, "full_sq"] = np.NaN
# bad_index = train_df[train_df.life_sq > 300].index
# train_df.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
# bad_index = test_df[test_df.life_sq > 200].index
# test_df.loc[bad_index, ["life_sq", "full_sq"]] = np.NaN
# train_df.product_type.value_counts(normalize= True)
# test_df.product_type.value_counts(normalize= True)
bad_index = train_df[train_df.build_year < 1500].index
train_df.loc[bad_index, "build_year"] = np.NaN
bad_index = test_df[test_df.build_year < 1500].index
test_df.loc[bad_index, "build_year"] = np.NaN
bad_index = train_df[train_df.num_room == 0].index
train_df.loc[bad_index, "num_room"] = np.NaN
bad_index = test_df[test_df.num_room == 0].index
test_df.loc[bad_index, "num_room"] = np.NaN
bad_index = [10076, 11621, 17764, 19390, 24007, 26713, 29172]
train_df.loc[bad_index, "num_room"] = np.NaN
bad_index = [3174, 7313]
test_df.loc[bad_index, "num_room"] = np.NaN
bad_index = train_df[(train_df.floor == 0).values * (train_df.max_floor == 0).values].index
train_df.loc[bad_index, ["max_floor", "floor"]] = np.NaN
bad_index = train_df[train_df.floor == 0].index
train_df.loc[bad_index, "floor"] = np.NaN
bad_index = train_df[train_df.max_floor == 0].index
train_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = test_df[test_df.max_floor == 0].index
test_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = train_df[train_df.floor > train_df.max_floor].index
train_df.loc[bad_index, "max_floor"] = np.NaN
bad_index = test_df[test_df.floor > test_df.max_floor].index
test_df.loc[bad_index, "max_floor"] = np.NaN
train_df.floor.describe(percentiles= [0.9999])
bad_index = [23584]
train_df.loc[bad_index, "floor"] = np.NaN
train_df.material.value_counts()
test_df.material.value_counts()
train_df.state.value_counts()
bad_index = train_df[train_df.state == 33].index
train_df.loc[bad_index, "state"] = np.NaN
test_df.state.value_counts()

print('Feature Engineering...')
# # Add month-year
# month_year = (train_df.timestamp.dt.month*30 + train_df.timestamp.dt.year * 365)
# month_year_cnt_map = month_year.value_counts().to_dict()
# train_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

# month_year = (test_df.timestamp.dt.month*30 + test_df.timestamp.dt.year * 365)
# month_year_cnt_map = month_year.value_counts().to_dict()
# test_df['month_year_cnt'] = month_year.map(month_year_cnt_map)

# # Add week-year count
# week_year = (train_df.timestamp.dt.weekofyear*7 + train_df.timestamp.dt.year * 365)
# week_year_cnt_map = week_year.value_counts().to_dict()
# train_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

# week_year = (test_df.timestamp.dt.weekofyear*7 + test_df.timestamp.dt.year * 365)
# week_year_cnt_map = week_year.value_counts().to_dict()
# test_df['week_year_cnt'] = week_year.map(week_year_cnt_map)

# # Add month and day-of-week
# train_df['month'] = train_df.timestamp.dt.month
# train_df['dow'] = train_df.timestamp.dt.dayofweek

# test_df['month'] = test_df.timestamp.dt.month
# test_df['dow'] = test_df.timestamp.dt.dayofweek

# Other feature engineering
train_df['rel_floor'] = 0.05+train_df['floor'] / train_df['max_floor'].astype(float)
train_df['rel_kitch_sq'] = 0.05+train_df['kitch_sq'] / train_df['full_sq'].astype(float)

test_df['rel_floor'] = 0.05+test_df['floor'] / test_df['max_floor'].astype(float)
test_df['rel_kitch_sq'] = 0.05+test_df['kitch_sq'] / test_df['full_sq'].astype(float)

train_df.apartment_name=train_df.sub_area + train_df['metro_km_avto'].astype(str)
test_df.apartment_name=test_df.sub_area + train_df['metro_km_avto'].astype(str)

train_df['area_per_room'] = train_df['life_sq'] / train_df['num_room'].astype(float)
test_df['area_per_room'] = test_df['life_sq'] / test_df['num_room'].astype(float)

# train_df['area_per_room'] = train_df['life_sq'] / train_df['num_room'].astype(float) #rough area per room
train_df['livArea_ratio'] = train_df['life_sq'] / train_df['full_sq'].astype(float) #rough living area
train_df['yrs_old'] = 2017 - train_df['build_year'].astype(float) #years old from 2017
train_df['avgfloor_sq'] = train_df['life_sq']/train_df['max_floor'].astype(float) #living area per floor
train_df['pts_floor_ratio'] = train_df['public_transport_station_km']/train_df['max_floor'].astype(float)
# looking for significance of apartment buildings near public t 
# train_df['room_size'] = train_df['life_sq'] / train_df['num_room'].astype(float)
# doubled a var by accident
# when removing one score did not improve...
train_df['gender_ratio'] = train_df['male_f']/train_df['female_f'].astype(float)
train_df['kg_park_ratio'] = train_df['kindergarten_km']/train_df['park_km'].astype(float) #significance of children?
train_df['high_ed_extent'] = train_df['school_km'] / train_df['kindergarten_km'] #schooling
train_df['pts_x_state'] = train_df['public_transport_station_km'] * train_df['state'].astype(float) #public trans * state of listing
train_df['lifesq_x_state'] = train_df['life_sq'] * train_df['state'].astype(float) #life_sq times the state of the place
train_df['floor_x_state'] = train_df['floor'] * train_df['state'].astype(float) #relative floor * the state of the place

test_df['area_per_room'] = test_df['life_sq'] / test_df['num_room'].astype(float)
test_df['livArea_ratio'] = test_df['life_sq'] / test_df['full_sq'].astype(float)
test_df['yrs_old'] = 2017 - test_df['build_year'].astype(float)
test_df['avgfloor_sq'] = test_df['life_sq']/test_df['max_floor'].astype(float) #living area per floor
test_df['pts_floor_ratio'] = test_df['public_transport_station_km']/test_df['max_floor'].astype(float) #apartments near public t?
test_df['room_size'] = test_df['life_sq'] / test_df['num_room'].astype(float)
test_df['gender_ratio'] = test_df['male_f']/test_df['female_f'].astype(float)
test_df['kg_park_ratio'] = test_df['kindergarten_km']/test_df['park_km'].astype(float)
test_df['high_ed_extent'] = test_df['school_km'] / test_df['kindergarten_km']
test_df['pts_x_state'] = test_df['public_transport_station_km'] * test_df['state'].astype(float) #public trans * state of listing
test_df['lifesq_x_state'] = test_df['life_sq'] * test_df['state'].astype(float)
test_df['floor_x_state'] = test_df['floor'] * test_df['state'].astype(float)


print('Running Model 1_2...')
def ymid(ymstr):
	year = int(ymstr[0:4]) - 2011
	month = int(ymstr[4:6]) - 8
	if month < 0:
		month += 12
		year -= 1
	return year * 12 + month

train_y_2 = []
for i in range(0, len(train_df['ppa'])):
	mppa = g_ppa_df.loc[ymid(train_df.loc[i, 'yearmonth']), 'ppa']
	train_y_2.append(train_df.loc[i, 'ppa'] - mppa)

train_X_2 = train_df.drop(['index', 'id', 'timestamp', 'price_doc', 'ppa'], axis=1)
test_X_2 = test_df.drop(['id', 'timestamp'], axis=1)

for c in train_X_2.columns:
    if train_X_2[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(train_X_2[c].values))
        train_X_2[c] = lbl.transform(list(train_X_2[c].values))
for c in test_X_2.columns:
    if test_X_2[c].dtype == 'object':
        lbl = preprocessing.LabelEncoder()
        lbl.fit(list(test_X_2[c].values))
        test_X_2[c] = lbl.transform(list(test_X_2[c].values))


xgb_params_2 = {
    'eta': 0.05,
    'max_depth': 6,
    'subsample': 0.7,
    'colsample_bytree': 1,
    'objective': 'reg:linear',
    'eval_metric': 'rmse',
    'silent': 1
}

dtrain_2 = xgb.DMatrix(train_X_2, train_y_2)
dtest_2 = xgb.DMatrix(test_X_2)

num_boost_rounds = 420
model_2 = xgb.train(dict(xgb_params_2, silent=0), dtrain_2, num_boost_round=num_boost_rounds)
pred_test_y_2 = model_2.predict(dtest_2)

pred_price = []

_shift = ymid('201507')
for i in range(0, len(pred_test_y_2)):
	ppa = pred_test_y_2[i]
	ppa += pred_test_y[ymid(test_df.loc[i, 'yearmonth']) - _shift]
	pred_price.append(ppa * test_df.loc[i, 'full_sq'])

out_df = pd.DataFrame(test_df['id'])
out_df.columns = ['id']
out_df['price_doc'] = pred_price
out_df.to_csv('ppa.csv', index=False)
