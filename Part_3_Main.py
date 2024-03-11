import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sklearn as sk
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
# import matplotlib.pyplot as plt
import scipy.fftpack
from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import linear_sum_assignment


# Get Columns from INSULINData CSV file

Insulin_col_list = ["Date", "Time", "BWZ Carb Input (grams)"]
df_insulin = pd.read_csv('InsulinData.csv', usecols=Insulin_col_list)
df_insulin['DateTime'] = pd.to_datetime(df_insulin.pop("Date")) + pd.to_timedelta(df_insulin.pop("Time"))


# removes all rows without carb intake

df_insulin.dropna(how='any', inplace=True)
df_insulin.reset_index(drop=True, inplace=True)


datetime_insulin_series = df_insulin["DateTime"]
cgm_carb_series = df_insulin["BWZ Carb Input (grams)"]

CGM_col_list = ["Date", "Time", "Sensor Glucose (mg/dL)"]
df_cgm_data = pd.read_csv('CGMData.csv', usecols=CGM_col_list)
df_cgm_data.dropna()

df_cgm_data['DateTime'] = pd.to_datetime(df_cgm_data.pop("Date")) + pd.to_timedelta(df_cgm_data.pop("Time"))

datetime_cgm_glucose_series = df_cgm_data["DateTime"]
cgm_glucose_series = df_cgm_data["Sensor Glucose (mg/dL)"]

# if carbs exists at a time between 2 hours, then get the time of the start of the meal


# print(df_insulin.head(100))

insulin_length = df_insulin.index

in_size = len(insulin_length)

total_meal_count = 0

meal_start_times = []
meal_carb_amounts = []

# Find Meal Start times


for i in range(0, in_size - 1, 1):

    if (datetime_insulin_series[i] - datetime_insulin_series[i + 1]) >= pd.Timedelta(hours=2, minutes=30) and \
            cgm_carb_series[i + 1] != 0:
        # print(datetime_insulin_series[i])

        meal_start_time = (datetime_insulin_series[i + 1] - pd.Timedelta("30 minutes"))
        # print(meal_start_time)
        total_meal_count += 1
        meal_start_times.append(meal_start_time)
        meal_carb_amounts.append(cgm_carb_series[i + 1])


# FINDING CORRESPONDING TIME IN OTHER FILE

index = df_cgm_data.index
cgm_total_rows = len(index)

cgm_meal_start_indices = []

final_meal_carb_amounts = []

loopstart = 0
j = 0

for meal_time in meal_start_times:
    # print(meal_time)
    j += 1

    for i in range(loopstart, cgm_total_rows, 1):

        if pd.Timedelta("0 minutes") <= (datetime_cgm_glucose_series[i] - meal_time) < pd.Timedelta("5 minutes"):
            # print(datetime_cgm_glucose_series[i])
            cgm_meal_start_indices.append(i)
            final_meal_carb_amounts.append(meal_carb_amounts[j - 1])
            loopstart = i
            break


# GET MEAL DATA USING INDICES OF MEAL DATA STARTS


meal_array = [0] * 30
meal_data_df = pd.DataFrame()

for index in cgm_meal_start_indices:
    for i in range(0, 30, 1):
        meal_array[i] = cgm_glucose_series[index - i]

    meal_series = pd.Series(meal_array)
    meal_data_df = meal_data_df.append(meal_series, ignore_index=True)


n = (max(meal_carb_amounts) - min(meal_carb_amounts)) / 20

print(meal_array)
meal_data_df['Carb Amount'] = final_meal_carb_amounts

gramcgm_carb_series = meal_data_df['Carb Amount']

bin_number = [0] * len(gramcgm_carb_series)

j = 0

for i in range(len(gramcgm_carb_series)):

    if gramcgm_carb_series[i] < 30:
        bin_number[j] = 0
        j += 1
    elif gramcgm_carb_series[i] < 60:
        bin_number[i] = 1
        j += 1
    else:
        bin_number[i] = 2

meal_data_df['Bin #'] = bin_number

meal_data_df.to_csv('Meal_data_Results.csv', header=False, index=False)

meal_data_df = meal_data_df[meal_data_df.isnull().sum(axis=1) < 2]



meal_data_df.columns = ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14',
                        '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29',
                        '30', 'Carb Amount', 'Bin #']

meal_data_df = meal_data_df.dropna()

meal_data_df.to_csv('Refined_Meal_data_Results.csv', header=False, index=False)

# FEATURE EXTRACTION


# meal_data_features
total_meal_data_rows = meal_data_df.count

column_names = ["Max-Min %", "Time", "Gradient", "FFT", "Bin #", "Carb Amount"]

meal_feature_DF = pd.DataFrame(columns=column_names)

meal_feature_DF["Carb Amount"] = meal_data_df["Carb Amount"]

new_bin_number = meal_data_df['Bin #']

# MIN MAX PERCENTAGE


Meal_Max_Min_Percentage = (meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '24', '25']].max(axis=1) - meal_data_df[['1', '2', '3', '4', '5', '6', '7', '8', '9']].min(axis=1)) / meal_data_df[['1', '2', '3', '4', '5', '6', '7', '8', '9']].min(axis=1)




meal_feature_DF["Max-Min %"] = Meal_Max_Min_Percentage



print(meal_feature_DF["Max-Min %"].mean())

# TIME until MAX glucose

time_max_G_meal = meal_data_df[['6', '7', '8', '9', '10', '11', '12', '13', '14','15', '16', '17', '18', '19', '20', '21', '22']].idxmax(axis=1).astype(int) * 5 - meal_data_df[['1', '2', '3', '4', '5', '6', '7', '8', '9']].idxmax(axis=1).astype(int) * 5

meal_feature_DF["Time"] = time_max_G_meal




# FFT FAST FOURIER TRANSFORMATION


def second_FFT(row):
    fft_meal = np.fft.fft(row)

    fft = fft_meal[1]

    fft = fft.real

    return fft


fft_meal_data = meal_data_df.drop(columns=['Carb Amount', 'Bin #'])

meal_feature_DF['FFT'] = meal_data_df.apply(lambda x: second_FFT(x), axis=1)  # here we are applying apply function which performs the required action on each row

# """For each row we are passing yes or no and storing it in a new column"""


fft_meal = np.fft.fft(meal_data_df)


# Number of samplepoints
yf = scipy.fftpack.fft(fft_meal_data)


print("1st derivative . . . . . . .")
meal_gradient_1 = meal_data_df.diff(axis=1)

print(meal_gradient_1)

maxValues_1 = meal_gradient_1.max(axis=1)

# print(maxValues_1)


meal_feature_DF["Gradient"] = maxValues_1


# Second GRADIENT - d^2CGM/dt^2


print("2nd derivative . . . . . . .")
meal_gradient_2 = meal_gradient_1.diff(axis=1)

maxValues_2 = meal_gradient_2.max(axis=1)

print(maxValues_2)

meal_feature_DF["d^2CGM/dt^2"] = maxValues_2

print(meal_feature_DF)


# CLUSTERING K MEANS

# print the names of the 3 features
print("Features: Max-Min%, Max-Min, Gradient, FFT, 2nd deriv")

meal_feature_DF["Bin #"] = new_bin_number

result_columns = ["SSE Kmeans", "SSE DBScan", "EntropyK", "EntropyDB", "PurityK", "PurityDB"]

Results_DF = pd.DataFrame(columns=result_columns)

meal_feature_DF = meal_feature_DF.dropna()

ground_truth = meal_feature_DF["Bin #"]

ground_truth_DF = pd.DataFrame(ground_truth)

meal_feature_DF.to_csv('Meal_Features_DF_with_Bin.csv', header=False, index=False)

meal_feature_DF.to_csv('Meal_Features_DF.csv', header=False, index=False)

ground_truth_DF.to_csv('Ground_truth_DF.csv', header=False, index=False)

kmeans = KMeans(n_clusters=3, random_state=1).fit(meal_feature_DF)

k_cluster_labels = kmeans.labels_
print(k_cluster_labels)

k_cluster_centers = kmeans.cluster_centers_
print(k_cluster_centers)

k_cluster_SSE = kmeans.inertia_
print(k_cluster_SSE)

SSE_Kmeans = k_cluster_SSE



print(k_cluster_labels)


def purity_score(y_true, y_pred):
    # compute contingency matrix (also called confusion matrix)
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
    # print(contingency_matrix)
    # return purity
    return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


Kmean_purity = purity_score(ground_truth, k_cluster_labels)

print(Kmean_purity)

PurityK = Kmean_purity

EntropyK = entropy([Kmean_purity, 1 - Kmean_purity], base=2)


def cluster_accuracy(y_true, y_pred):
    contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)

    # Find optimal one-to-one mapping between cluster labels and true labels
    row_ind, col_ind = linear_sum_assignment(-contingency_matrix)

    # Return cluster accuracy
    return contingency_matrix[row_ind, col_ind].sum() / np.sum(contingency_matrix)


Kmean_accuracy = cluster_accuracy(ground_truth, k_cluster_labels)

print(Kmean_accuracy)

print(ground_truth)


# CLUSTERING DBSCAN

X = meal_feature_DF

db = DBSCAN(eps=50, min_samples=5).fit(X)

dbs_clustering_labels = db.labels_



neigh = NearestNeighbors(n_neighbors=2)
nbrs = neigh.fit(X)
distances, indices = nbrs.kneighbors(X)

distances = np.sort(distances, axis=0)
distances = distances[:, 1]


core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
core_samples_mask[db.core_sample_indices_] = True
labels = db.labels_

# Number of clusters in labels, ignoring noise if present.
n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
n_noise_ = list(labels).count(-1)

print('Estimated number of clusters: %d' % n_clusters_)
print('Estimated number of noise points: %d' % n_noise_)
print("Homogeneity: %0.3f" % metrics.homogeneity_score(ground_truth, labels))
print("Completeness: %0.3f" % metrics.completeness_score(ground_truth, labels))
print("V-measure: %0.3f" % metrics.v_measure_score(ground_truth, labels))
print("Adjusted Rand Index: %0.3f"% metrics.adjusted_rand_score(ground_truth, labels))
print("Adjusted Mutual Information: %0.3f"% metrics.adjusted_mutual_info_score(ground_truth, labels))

purity_DB = metrics.silhouette_score(X, labels)
print("Silhouette Coefficient: %0.3f"% metrics.silhouette_score(X, labels))

DBScan_purity = purity_score(ground_truth, labels)

print("DBSCAN Purity:")
print(DBScan_purity)

SSE_DBScan = sum(distances)
print(SSE_DBScan)

PurityDB = DBScan_purity

EntropyDB = entropy([purity_DB, 1 - purity_DB], base=2)

Results_DF = pd.DataFrame([SSE_Kmeans, SSE_DBScan, EntropyK, EntropyDB, PurityK, PurityDB])
Results_DF = Results_DF.transpose()

print(Results_DF)

Results_DF.to_csv('Result.csv', header=False, index=False)