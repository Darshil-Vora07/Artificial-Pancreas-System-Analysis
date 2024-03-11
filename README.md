Project Overview
This repository hosts a comprehensive analysis of the Artificial Pancreas system, focusing on the Medtronic 670G system. It encompasses three interconnected projects aimed at extracting time series properties from glucose levels, training a machine model to distinguish between meal and no meal data, and applying cluster validation techniques on meal data. This work leverages datasets from Continuous Glucose Monitoring (CGM) and insulin pump data to derive insights crucial for diabetes management through technology.

Technology Stack
Python 3.6 to 3.8
Libraries: scikit-learn==0.21.2, pandas==0.25.1, Python pickle

Project Descriptions
Project 1: Time Series Analysis
Objective: Analyze CGM data to extract metrics related to glucose levels over different periods.
Key Metrics: Percentage time in hyperglycemia, hyperglycemia critical, range, range secondary, hypoglycemia level 1 and 2, for daytime, overnight, and whole day intervals.


Project 2: Machine Learning Model for Meal Detection
Objective: Train a machine learning model to differentiate between times when a meal was consumed and when no meal was taken.
Approach: SVM classifier trained on features extracted from CGM and insulin pump data.


Project 3: Clustering and Cluster Validation
Objective: Perform clustering on meal data based on carbohydrate intake and validate clusters using SSE, entropy, and purity metrics.
Techniques: K-means and DBSCAN clustering applied to features extracted from meal time series data.
