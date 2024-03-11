import pickle
import pandas as pd
import numpy as np

# load test file into dataframe


df_test = pd.read_csv("test.csv", header=None)

# load_file("test.csv")
# print(df_test)


df_test.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']

# test_data_features


column_names = ["Max-Min %", "Time", "Gradient"]

df_test_features = pd.DataFrame(columns=column_names)


Test_Max_Min_Percentage = (df_test[['6', '7', '8', '9', '10', '11', '12', '13', '14','15']].max(axis=1) - df_test[['1', '2', '3']].min(axis=1)) / df_test[['1', '2', '3']].min(axis=1)

df_test_features["Max-Min %"] = Test_Max_Min_Percentage

time_max_G = df_test[['6', '7', '8', '9', '10', '11', '12', '13', '14','15', '16', '17', '18']].idxmax(axis=1).astype(int) * 5

df_test_features["Time"] = time_max_G


df_test_gradient = pd.DataFrame()
gradients = np.gradient(df_test)
size = len(df_test)
max_gradients = [0] * size

for array in gradients:
    for i in range(len(array)):
        x = max(array[i])
        max_gradients[i] = x

df_test_gradient = df_test_gradient.append(max_gradients)

df_test_features["Gradient"] = df_test_gradient

picklefilename = 'SVMClassifier.pickle'
import joblib
loaded_clf = joblib.load('SVMClassifier.pickle')
test_pred = loaded_clf.predict(df_test_features)

df_results = pd.DataFrame(test_pred)

df_results.to_csv('Result.csv', header=False, index=False)