import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import pickle
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold, RepeatedKFold
from joblib import dump, load



Insulin_col_list = ["Date", "Time", "BWZ Carb Input (grams)"]
df_insulin = pd.read_csv('InsulinData.csv', usecols=Insulin_col_list)
df_insulin['DateTime'] = pd.to_datetime(df_insulin.pop("Date")) + pd.to_timedelta(df_insulin.pop("Time"))


df_insulin.dropna(how='any', inplace=True)
df_insulin.reset_index(drop=True, inplace=True)


datetime_insulin_series = df_insulin["DateTime"]
meal_input = df_insulin["BWZ Carb Input (grams)"]

CGM_col_list = ["Date", "Time", "Sensor Glucose (mg/dL)"]
df_cgm_data = pd.read_csv('CGMData.csv', usecols=CGM_col_list)
df_cgm_data.dropna()

df_cgm_data['DateTime'] = pd.to_datetime(df_cgm_data.pop("Date")) + pd.to_timedelta(df_cgm_data.pop("Time"))

datetime_glucose_series = df_cgm_data["DateTime"]
glucose_series = df_cgm_data["Sensor Glucose (mg/dL)"]

insulin_length = df_insulin.index

in_size = len(insulin_length)

total_meal_count = 0

meal_start_times = []


for i in range(0, in_size - 1, 1):

    if (datetime_insulin_series[i] - datetime_insulin_series[i + 1]) >= pd.Timedelta(hours=2, minutes=30) and meal_input[i + 1] != 0:

        meal_start_time = (datetime_insulin_series[i + 1] - pd.Timedelta("30 minutes"))
        total_meal_count += 1
        meal_start_times.append(meal_start_time)

index = df_cgm_data.index
cgm_total_rows = len(index)

cgm_meal_start_indices = []

loopstart = 0

for meal_time in meal_start_times:
    # print(meal_time)

    for i in range(loopstart, cgm_total_rows, 1):

        if pd.Timedelta("0 minutes") <= (datetime_glucose_series[i] - meal_time) < pd.Timedelta("5 minutes"):
            # print(datetime_glucose_series[i])
            cgm_meal_start_indices.append(i)
            loopstart = i
            break



meal_array = [0] * 30
meal_data_df = pd.DataFrame()

for index in cgm_meal_start_indices:
    for i in range(0, 30, 1):
        meal_array[i] = glucose_series[index - i]

    meal_series = pd.Series(meal_array)
    meal_data_df = meal_data_df.append(meal_series, ignore_index=True)



# Now same for Patient 2
Insulin_col_list = ["Date", "Time", "BWZ Carb Input (grams)"]
df_insulin_2 = pd.read_csv('Insulin_patient2.csv', usecols=Insulin_col_list)
df_insulin_2['DateTime'] = pd.to_datetime(df_insulin_2.pop("Date")) + pd.to_timedelta(df_insulin_2.pop("Time"))

print("Patient 2 .....................................")
df_insulin_2.dropna(how='any', inplace=True)
df_insulin_2.reset_index(drop=True, inplace=True)


datetime_insulin_series_2 = df_insulin_2["DateTime"]
meal_input_2 = df_insulin_2["BWZ Carb Input (grams)"]

CGM_col_list = ["Date", "Time", "Sensor Glucose (mg/dL)"]
df_cgm_data_2 = pd.read_csv('CGM_patient2.csv', usecols=CGM_col_list)
df_cgm_data_2.dropna()
df_cgm_data_2['DateTime'] = pd.to_datetime(df_cgm_data_2.pop("Date")) + pd.to_timedelta(df_cgm_data_2.pop("Time"))

datetime_glucose_series_2 = df_cgm_data_2["DateTime"]
glucose_series_2 = df_cgm_data_2["Sensor Glucose (mg/dL)"]


insulin_length_2 = df_insulin_2.index

in_2_size = len(insulin_length_2)

total_meal_count_2 = 0

meal_start_times_2 = []

for i in range(0, in_2_size - 1, 1):

    if (datetime_insulin_series_2[i] - datetime_insulin_series_2[i + 1]) >= pd.Timedelta(hours=2, minutes=30) and meal_input_2[i + 1] != 0:
        # print(datetime_insulin_series[i])

        meal_start_time_2 = (datetime_insulin_series_2[i + 1] - pd.Timedelta("30 minutes"))
        # print(meal_start_time_2)
        total_meal_count_2 += 1
        meal_start_times_2.append(meal_start_time_2)

index_2 = df_cgm_data_2.index
cgm_total_rows_2 = len(index_2)

cgm_meal_start_indices_2 = []

loopstart_2 = 0

for meal_time_2 in meal_start_times_2:

    for j in range(loopstart_2, cgm_total_rows_2, 1):
        if pd.Timedelta("0 minutes") <= (datetime_glucose_series_2[j] - meal_time_2) < pd.Timedelta("10 minutes"):
            # print("Success!")
            cgm_meal_start_indices_2.append(j)
            # print(j)
            loopstart_2 = j
            break


meal_array_2 = [0] * 30
# meal_data_df = pd.DataFrame()  - ALREADY EXISTS

for index_2 in cgm_meal_start_indices_2:
    for i in range(0, 30, 1):
        meal_array_2[i] = glucose_series_2[index_2 - i]

    meal_series_2 = pd.Series(meal_array_2)
    # APPEND to meal_data_df
    meal_data_df = meal_data_df.append(meal_series_2, ignore_index=True)


Insulin_col_list = ["Date", "Time", "BWZ Carb Input (grams)"]
df_insulin_NM = pd.read_csv('InsulinData.csv', usecols=Insulin_col_list)
df_insulin_NM['DateTime'] = pd.to_datetime(df_insulin_NM.pop("Date")) + pd.to_timedelta(df_insulin_NM.pop("Time"))

NM_datetime_insulin_series = df_insulin_NM["DateTime"]
NM_meal_input = df_insulin_NM["BWZ Carb Input (grams)"]

NM_insulin_length = df_insulin_NM.index
NM_in_size = len(NM_insulin_length)
# print(NM_in_size)
total_no_meal_count = 0

df_cgm_data = pd.read_csv('CGMData.csv', usecols=CGM_col_list)
df_cgm_data.dropna()

df_cgm_data['DateTime'] = pd.to_datetime(df_cgm_data.pop("Date")) + pd.to_timedelta(df_cgm_data.pop("Time"))

datetime_glucose_series = df_cgm_data["DateTime"]
glucose_series_2 = df_cgm_data["Sensor Glucose (mg/dL)"]

no_meal_start_times = []

prev_meal = NM_datetime_insulin_series[0] + pd.Timedelta("35 minutes")

for meal_time in meal_start_times:

    for i in range(0, NM_in_size, 1):

        if (prev_meal - NM_datetime_insulin_series[i]) > pd.Timedelta("30 minutes") and (NM_datetime_insulin_series[i] - pd.Timedelta("2 hours")) > meal_time:
            no_meal_start_times.append(NM_datetime_insulin_series[i] - pd.Timedelta("2 hours"))
            total_no_meal_count += 1
            prev_meal = meal_time
            break


index = df_cgm_data.index
cgm_total_rows = len(index)

no_meal_start_indices = []


loopstart = 0

for no_meal_time in no_meal_start_times:
    # print(meal_time)

    for i in range(loopstart, cgm_total_rows, 1):

        if pd.Timedelta("0 minutes") <= (datetime_glucose_series[i] - no_meal_time) < pd.Timedelta("5 minutes"):
            # print(datetime_glucose_series[i])
            no_meal_start_indices.append(i)
            loopstart = i
            break


no_meal_array = [0] * 24
no_meal_data_df = pd.DataFrame()

for index in no_meal_start_indices:
    for i in range(0, 24, 1):
        no_meal_array[i] = glucose_series[index - i]
        # print(no_meal_array[i])

    no_meal_series = pd.Series(no_meal_array)
    no_meal_data_df = no_meal_data_df.append(no_meal_series, ignore_index=True)


meal_data_df = meal_data_df[meal_data_df.isnull().sum(axis=1) < 3]

no_meal_data_df = no_meal_data_df[no_meal_data_df.isnull().sum(axis=1) < 3]

meal_data_df.to_csv('Refined_Meal_data_Results.csv', header=False, index=False)
no_meal_data_df.to_csv('Refined_No_Meal_data_Results.csv', header=False, index=False)

meal_data_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30']
print(meal_data_df)

no_meal_data_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24']
print(no_meal_data_df)

total_meal_data_rows = meal_data_df.count
total_no_meal_data_rows = no_meal_data_df.count

column_names = ["Max-Min %", "Time", "Gradient", "Label"]

meal_feature_DF = pd.DataFrame(columns=column_names)
no_meal_feature_DF = pd.DataFrame(columns=column_names)



Meal_Max_Min_Percentage = (meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14', '15']].max(axis=1) - meal_data_df[['1', '2', '3']].min(axis=1)) / meal_data_df[['1', '2']].min(axis=1)

No_Meal_Max_Min_Percentage = (no_meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14', '15']].max(axis=1) - no_meal_data_df[['1', '2', '3']].min(axis=1)) / no_meal_data_df[['1', '2', '3']].min(axis=1)


meal_feature_DF["Max-Min %"] = Meal_Max_Min_Percentage
no_meal_feature_DF["Max-Min %"] = No_Meal_Max_Min_Percentage


time_max_G_meal = meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']].idxmax(axis=1).astype(int) * 5
time_max_G_no_meal = no_meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18']].idxmax(axis=1).astype(int) * 5


meal_feature_DF["Time"] = time_max_G_meal
no_meal_feature_DF["Time"] = time_max_G_no_meal

fft_meal = np.fft.fft(meal_data_df)
fft_nomeal = np.fft.fft(no_meal_data_df)


meal_gradient = np.gradient(meal_data_df)
no_meal_gradient = np.gradient(no_meal_data_df)

meal_gradient_df = pd.DataFrame()
no_meal_gradient_df = pd.DataFrame()


size = len(meal_data_df)

max_gradients = [0] * size

for array in meal_gradient:

    for i in range(len(array)):
        x = max(array[i])
        max_gradients[i] = x

meal_gradient_df = meal_gradient_df.append(max_gradients)
# print (max_gradients)

no_size = len(no_meal_data_df)

nm_max_gradients = [0] * no_size

for array in no_meal_gradient:

    for i in range(len(array)):
        x = max(array[i])

        nm_max_gradients[i] = x

no_meal_gradient_df = no_meal_gradient_df.append(nm_max_gradients)


meal_feature_DF["Gradient"] = meal_gradient_df
no_meal_feature_DF["Gradient"] = no_meal_gradient_df


# print the names of the 3 features
print("Features: Max-Min%, Time, Gradient")

print("Labels: 'Meal' 'No Meal'")

meal_feature_DF["Label"] = 1
no_meal_feature_DF["Label"] = 0

meal_feature_DF.to_csv('Meal_Features_DF.csv', header=False, index=False)

no_meal_feature_DF.to_csv('No_Meal_Features_DF.csv', header=False, index=False)

all_feature_df = pd.DataFrame()
all_feature_df = all_feature_df.append(meal_feature_DF)
all_feature_df = all_feature_df.append(no_meal_feature_DF)

all_feature_df = all_feature_df.dropna()
print("All feature df \n")
print(all_feature_df)
all_feature_df.to_csv('All_Feature_DF.csv', header=False, index=False)


feature_df = pd.DataFrame()
label = pd.DataFrame()
feature_df = all_feature_df.drop(columns='Label')
label = all_feature_df["Label"]
feature_df = feature_df.reset_index(drop=True)
label = label.reset_index(drop=True)
print(feature_df)
print(label)

kfold = KFold(n_splits=10,shuffle=False)
Acc = []
Prec = []
Recall = []
F1 = []

model=SVC(kernel='linear', gamma='auto')
for train_index, test_index in kfold.split(feature_df):
    X_train,X_test,y_train,y_test = feature_df.loc[train_index], feature_df.loc[test_index], label.loc[train_index], label.loc[test_index]
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print("Accuracy: ", acc)
    Acc.append(acc)
    print("Precision: ", precision_score(y_test, y_pred))
    print("Recall: ", recall_score(y_test, y_pred))
    recall = recall_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    Recall.append(recall)
    Prec.append(prec)
    f1 = 2 * recall * prec / (recall + prec)
    print("F1 Score: ", f1)
    F1.append(f1)


print('Best Accuracy Score is',np.max(Acc)*100)
print('Best Recall Score is',np.max(Recall)*100)
print('Best Precision Score is',np.max(Prec)*100)
print(F1)


classifier=SVC(kernel='linear', gamma='auto')
X, y= feature_df, label
classifier.fit(X,y)
dump(classifier, 'SVMClassifier.pickle')
