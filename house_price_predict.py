import os, time, pickle
import pandas as pd
from pandas.plotting import scatter_matrix
import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, GridSearchCV

'''
url: the url which fetches the data
id: the id of transaction
Lng: and Lat coordinates, using the BD09 protocol.
Cid: community id
tradeTime: the time of transaction
DOM: active days on market.Know more in https://en.wikipedia.org/wiki/Days_on_market
followers: the number of people follow the transaction.
totalPrice: the total price
price: the average price by square
square: the square of house
livingRoom: the number of living room
drawingRoom: the number of drawing room
kitchen: the number of kitchen
bathroom the number of bathroom
floor: the height of the house.
buildingType: including tower( 1 ) , bungalow( 2 )，combination of plate and tower( 3 ), plate( 4 ).
constructionTime: the time of construction
renovationCondition: including other( 1 ), rough( 2 ),Simplicity( 3 ), hardcover( 4 )
buildingStructure: including unknown( 1 ), mixed( 2 ), brick and wood( 3 ), brick and concrete( 4 ),steel( 5 ) and steel-concrete composite ( 6 ).
ladderRatio: the proportion between number of elevator of ladder and number of residents on the same floor. It describes how many ladders a resident have on average.
elevator: have ( 1 ) or not have elevator( 0 )
fiveYearsProperty: if the owner have the property for less than 5 years
'''


def load_housing_data(housing_path="./data"):
    csv_path = os.path.join(housing_path, "raw_data.csv")
    housing_raw_data_frame = pd.read_csv(csv_path, low_memory=False)

    return housing_raw_data_frame


def data_prep(housing_raw_data_frame):
    housing_data_frame = housing_raw_data_frame.drop(labels=["url", "id", "Cid", "followers", "DOM", "price"],
                                                     axis=1)  # 去掉无用列
    housing_data_frame = housing_data_frame[~housing_data_frame["buildingType"].lt(1)] # 处理sb的buildingType列的异常值(<1)
    # 将缺失值设置为某个值 （中位数）
    imputer = SimpleImputer(strategy="median")
    housing_data_numpy = imputer.fit_transform(housing_data_frame)
    housing_data_frame = pd.DataFrame(housing_data_numpy, columns=housing_data_frame.columns)
    '''
    # 丢弃个别含缺失值的样例
    nullValueCount = housing_raw_data_frame.isnull().sum(axis=0)
    for index in nullValueCount.index:
        if 500 > nullValueCount[index] > 0:
            housing_raw_data_frame.dropna(subset=[index])
    '''
    return housing_data_frame


def data_statistics(housing_data_frame):
    with open("data/data_statistics", "w+", encoding="utf-8") as statistics:
        statistics.readline()
        for column in list(housing_data_frame.columns):
            if not column in ["Lng", "Lat", "totalPrice", "square", "ladderRatio", "communityAverage"]:
                statistics.write(str(housing_data_frame[column].value_counts()) + "\n\n")
        statistics.write(str(housing_data_frame.describe()) + "\n\n")
    print("Data statistics computed and restored in the file: data/data_statistics")


def split_train_test(housing_data, test_ratio):
    np.random.seed(42)
    # 随机抽样
    '''
    shuffled_indices = np.random.permutation(len(housing_data))
    test_set_size = int(len(housing_data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    
    return housing_data.iloc[train_indices], housing_data.iloc[test_indices]
    '''

    # 根据tradeTime分层抽样
    housing_data.sort_values(by="tradeTime", inplace=True)

    classified_samples = {}
    test = pd.DataFrame()
    train = pd.DataFrame()
    for year in housing_data["tradeTime"].value_counts().index:
        classified_samples.setdefault(year, housing_data.loc[housing_data["tradeTime"] == year])
        shuffled_indices = np.random.permutation(len(classified_samples[year]))
        test_set_size = int(len(classified_samples[year]) * test_ratio)
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        test = test.append(housing_data.iloc[test_indices], ignore_index=True)
        train = train.append(housing_data.iloc[train_indices], ignore_index=True)

    return train, test


def train_test_prep(housing_data, test_ratio):
    train_set, test_set = split_train_test(housing_data, test_ratio)
    print(len(train_set), "training data +", len(test_set), "test data.", len(train_set) + len(test_set),
          "data in total.")

    return train_set, test_set


def data_inspective(housing_data, attributes, geo_price=False, corr=False):
    housing = housing_data.copy()
    # 地理位置与房价关系
    if geo_price:
        housing.plot(kind="scatter", x="Lng", y="Lat", alpha=0.009, c="totalPrice", cmap=plt.get_cmap("ocean"),
                     colorbar=True)
        plt.show()
    # 各属性相关系数
    if corr:
        corr_matrix = housing.corr()
        print(corr_matrix)
    # 具体属性见相关性
    scatter_matrix(housing[attributes], figsize=(24, 8), alpha=0.1)
    plt.show()


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


def data_pipeline(data_set):
    num_attributes = ["Lng", "Lat", "tradeTime", "square", "livingRoom", "drawingRoom", "kitchen", "bathRoom", "floor",
                      "constructionTime", "ladderRatio", "elevator", "fiveYearsProperty", "subway", "district",
                      "communityAverage"]
    cat_attributes = ["buildingType", "renovationCondition", "buildingStructure"]

    num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_attributes)),
    ])
    cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(cat_attributes)),
        ("one_hot_encoder", OneHotEncoder())
    ])

    full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline)
    ])

    housing_prepared = full_pipeline.fit_transform(data_set)
    housing_labels = data_set["totalPrice"].copy()
    print("Data completely prepared!")
    return housing_prepared, housing_labels


def model_training(train_set_prepared, train_set_labels, model="LR"):
    print("Model Training Started...")
    if model == "LR":
        lin_reg = LinearRegression()
        lin_reg.fit(train_set_prepared, train_set_labels)
        return lin_reg
    if model == "DT":
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(train_set_prepared, train_set_labels)
        return tree_reg
    if model == "RF":
        forest_reg = RandomForestRegressor(max_features=8, n_estimators=30)
        forest_reg.fit(train_set_prepared, train_set_labels)
        print("Model Training Completed!")
        return forest_reg


def rmse(predictions, labels):
    print("Calculating Root Mean Squared Error...")
    lin_mse = mean_squared_error(labels, predictions)
    lin_rmse = np.sqrt(lin_mse)
    return lin_rmse


def cross_validation(model, data_set, labels):
    print("Using Cross Validation to measure model performance...")
    scores = cross_val_score(model, data_set, labels, scoring="neg_mean_squared_error", cv=10)
    rmse_scores = np.sqrt(-scores)
    return rmse_scores


def save_model(model, file):
    output = open(file, 'wb')
    pickle.dump(model, output)
    output.close()
    print("Model Saved!")


def load_model(file):
    model_file = open(file, 'rb')
    model = pickle.load(model_file)
    return model


def hyperparam_tuning(method, model, train_set_prepared, train_set_labels):
    if method == "GS":
        print("Starting Grid Search to find the best parameters...")
        param_grid = [
            {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
            {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]}
        ]
        grid_search = GridSearchCV(model, param_grid, cv=5, scoring='neg_mean_squared_error', refit=True)
        grid_search.fit(train_set_prepared, train_set_labels)
        best_params = grid_search.best_params_
        best_estimator = grid_search.best_estimator_
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
            print(np.sqrt(-mean_score), params)
        return best_params, best_estimator

def result_visual(final_predictions, labels):
    final_predictions = final_predictions[:100]
    labels = labels[:100]

    fig, ax = plt.subplots()
    plt.xlabel("Test Samples")
    plt.ylabel("Pricing Range Group")

    """Set interval for y label"""
    yticks = range(0, 1000, 50)
    ax.set_yticks(yticks)
    xticks = range(0, 100, 1)
    ax.set_xticks(xticks)

    """Set min and max value for axes"""
    ax.set_ylim([0, 1000])
    ax.set_xlim([0, 100])

    x = range(len(final_predictions))
    plt.plot(x, final_predictions, color="red", label="Predictions")
    plt.plot(x, labels, color="blue", label="Price")

    """Open the grid"""
    plt.grid(True)
    plt.legend(bbox_to_anchor=(1.0, 1), loc=1, borderaxespad=0.)
    plt.show()


if __name__ == '__main__':
    # 显示所有列
    # pd.set_option('display.max_columns', None)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    housing_raw_data_frame = load_housing_data()
    housing_data_frame = data_prep(housing_raw_data_frame)
    # data_statistics(housing_data_frame)

    train_set, test_set = train_test_prep(housing_data_frame, 0.15)
    # data_inspective(train_set, ["totalPrice", "tradeTime", "square", "livingRoom", "communityAverage"])


    train_set_prepared, train_set_labels = data_pipeline(train_set)
    # 训练模型
    '''
    model_name = "DT"

    model = model_training(train_set_prepared, train_set_labels, model_name)
    predictions = model.predict(train_set_prepared)

    lin_rmse = rmse(predictions, train_set_labels)
    print("RMSE Scores:", lin_rmse)

    cross_val_scores = cross_validation(model, train_set_prepared, train_set_labels)
    print("Scores:", cross_val_scores)
    print("Mean:", cross_val_scores.mean())
    print("Standard Deviation:", cross_val_scores.std())

    save_model(model, "model/" + model_name + "_" + time.strftime("%Y%m%d", time.localtime()) + ".model")
    '''

    # 载入模型

    model = load_model("model/RF_GS_20191220.model")

    # 利用Grid Search调参
    '''
    best_params, best_estimator = hyperparam_tuning("GS", model, train_set_prepared, train_set_labels)
    model_gridSearch = best_estimator
    save_model(model_gridSearch, "model/RF_GS_20191220.model")
    '''


    # Final test
    test_set_prepared, test_set_labels = data_pipeline(test_set)

    final_predictions = model.predict(test_set_prepared)
    final_mse = mean_squared_error(test_set_labels, final_predictions)
    final_rmse = np.sqrt(final_mse)

    print("The result for model final test:")
    print("MSE:",final_mse)
    print("RMSE",final_rmse)

    result_visual(final_predictions, test_set_labels)
    '''
    print(housing_raw_data_frame.head())
    print(housing_raw_data_frame.info())
    print(housing_raw_data_frame.describe())
    '''

    # housing_raw_data_frame.hist(bins=100, figsize=(100,11.5), density=True)
    # plt.show()
