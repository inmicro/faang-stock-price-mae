#importing libraries
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import numpy as np

def predict_apple():
    #reading data
    apple_df = dataframe1 = pd.read_csv("AAPL_data.csv")

    #cleaning out values with no data
    apple_df = apple_df.dropna(axis=0)

    #setting target and features
    target = apple_df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = apple_df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    apple_model = RandomForestRegressor(random_state = 0)
    apple_model.fit(train_markers, train_target)
    predictions = apple_model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_fb():
    #reading data
    fb_df = dataframe1 = pd.read_csv("FB_data.csv")

    #cleaning out values with no data
    fb_df = fb_df.dropna(axis=0)

    #setting target and features
    target = fb_df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = fb_df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    fb_model = RandomForestRegressor(random_state = 0)
    fb_model.fit(train_markers, train_target)
    predictions = fb_model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_msft():
    #reading data
    df = dataframe1 = pd.read_csv("MSFT_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = RandomForestRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_google():
    #reading data
    df = dataframe1 = pd.read_csv("GOOGL_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = RandomForestRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_orcl():
    #reading data
    df = dataframe1 = pd.read_csv("ORCL_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = RandomForestRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_amzn():
    #reading data
    df = dataframe1 = pd.read_csv("AMZN_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = RandomForestRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)



def predict_apple_dtr():
    #reading data
    apple_df = dataframe1 = pd.read_csv("AAPL_data.csv")

    #cleaning out values with no data
    apple_df = apple_df.dropna(axis=0)

    #setting target and features
    target = apple_df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = apple_df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    apple_model = DecisionTreeRegressor(random_state = 0)
    apple_model.fit(train_markers, train_target)
    predictions = apple_model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_fb_dtr():
    #reading data
    fb_df = dataframe1 = pd.read_csv("FB_data.csv")

    #cleaning out values with no data
    fb_df = fb_df.dropna(axis=0)

    #setting target and features
    target = fb_df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = fb_df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    fb_model = DecisionTreeRegressor(random_state = 0)
    fb_model.fit(train_markers, train_target)
    predictions = fb_model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_msft_dtr():
    #reading data
    df = dataframe1 = pd.read_csv("MSFT_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = DecisionTreeRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_google_dtr():
    #reading data
    df = dataframe1 = pd.read_csv("GOOGL_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = DecisionTreeRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_orcl_dtr():
    #reading data
    df = dataframe1 = pd.read_csv("ORCL_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = DecisionTreeRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

def predict_amzn_dtr():
    #reading data
    df = dataframe1 = pd.read_csv("AMZN_data.csv")

    #cleaning out values with no data
    df = df.dropna(axis=0)

    #setting target and features
    target = df.close
    factors = ['volume', 'high', 'low', 'open']
    markers = df[factors]

    # splitting data into evalset and trainset
    train_markers, eval_markers, train_target, eval_target = train_test_split(markers, target, random_state=0) 

    #creating the model
    model = DecisionTreeRegressor(random_state = 0)
    model.fit(train_markers, train_target)
    predictions = model.predict(eval_markers)
    return mean_absolute_error(eval_target, predictions)

companies = ["google", "microsoft", "apple", "oracle", "facebook", "amazon"]
company_basket_rf = company_basket_dtr = [predict_google(), predict_msft(), predict_apple(), predict_orcl(), predict_fb(), predict_amzn()]
company_basket_dtr = [predict_google_dtr(), predict_msft_dtr(), predict_apple_dtr(), predict_orcl_dtr(), predict_fb_dtr(), predict_amzn_dtr()]
fig = plt.figure(figsize = (10, 5))

X_axis = np.arange(len(companies))

plt.bar(X_axis - 0.2, company_basket_rf, 0.4, label = 'Random Forest')
plt.bar(X_axis + 0.2, company_basket_dtr, 0.4, label = 'Decision Tree Regressor')

plt.xticks(X_axis, companies)

plt.xlabel("Company")
plt.ylabel("Average error")

plt.title("average error in large tech companies")

plt.legend()

plt.show()