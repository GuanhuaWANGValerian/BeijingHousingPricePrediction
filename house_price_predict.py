import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer

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

    # 分层抽样
    # 根据tradeTime分层抽样
    # 未完

    for i, v in zip(housing_data["tradeTime"].value_counts().index, housing_data["tradeTime"].value_counts().values):
        if v < 100:
            housing_data.drop(housing_data.where(housing_data["tradeTime"] == i), axis=1)

    housing_data.sort_values(by="tradeTime", inplace=True)

    classified_samples = {}
    test, train = pd.DataFrame()
    for year in housing_data["tradeTime"].value_counts().index:
        classified_samples.setdefault(year, housing_data.where(housing_data["tradeTime"] == year))
        shuffled_indices = np.random.permutation(len(classified_samples[year]))
        test_set_size = int(len(classified_samples[year] * test_ratio))
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        test.append(housing_data.iloc[test_indices])
        train.append(housing_data.iloc[train_indices])

    return test, train






def train_test_prep(housing_data, test_ratio):
    train_set, test_set = split_train_test(housing_data, test_ratio)
    print(len(train_set), "training data +", len(test_set), "test data")

    return train_set, test_set


if __name__ == '__main__':
    # 显示所有列
    # pd.set_option('display.max_columns', None)
    pd.options.display.max_columns = None
    pd.options.display.max_rows = None

    housing_raw_data_frame = load_housing_data()
    housing_data_frame = data_prep(housing_raw_data_frame)
    #data_statistics(housing_data_frame)

    train_set, test_set = train_test_prep(housing_data_frame, 0.15)
    print(test_set.head())

    '''
    print(housing_raw_data_frame.head())
    print(housing_raw_data_frame.info())
    print(housing_raw_data_frame.describe())
    '''

    # housing_raw_data_frame.hist(bins=100, figsize=(100,11.5), density=True)
    # plt.show()
