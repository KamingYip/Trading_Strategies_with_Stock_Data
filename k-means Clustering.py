"""
Kaming Yip
CS677 A1 Data Science with Python
Apr. 14, 2020
Assignment 11.1: k-means
"""

from pandas_datareader import data as web
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def main():
    def get_stock(ticker, start_date, end_date):
        """
        download the historical data from yahoo web
        & manipulate the data to create desirable columns
        """
        try:
            df = web.get_data_yahoo(ticker, start=start_date, end=end_date)
            df['Return'] = df['Adj Close'].pct_change()
            df['Return'].fillna(0, inplace = True)
            df['Return'] = 100.0 * df['Return']
            df['Return'] = df['Return'].round(3)
            df['Date'] = df.index
            df['Date'] = pd.to_datetime(df['Date'])
            df['Month'] = df['Date'].dt.month
            df['Year'] = df['Date'].dt.year
            df['Day'] = df['Date'].dt.day
            for col in ['Open', 'High', 'Low', 'Close', 'Adj Close']:
                df[col] = df[col].round(2)
            df['Weekday'] = df['Date'].dt.weekday_name  
            df['Week_Number'] = df['Date'].dt.strftime('%U')
            df['Year_Week'] = df['Date'].dt.strftime('%Y-%U')
            col_list = ['Date', 'Year', 'Month', 'Day', 'Weekday', 
                        'Week_Number', 'Year_Week', 'Open', 
                        'High', 'Low', 'Close', 'Volume', 'Adj Close',
                        'Return']
            num_lines = len(df)
            df = df[col_list]
            print('read', num_lines, 'lines of data for ticker:' , ticker)
            return df
        except Exception as error:
            print(error)
            return None
    
    # design the selected stock name and time frame
    try:
        ticker='YELP'
        input_dir = os.getcwd()
        output_file = os.path.join(input_dir, ticker + '.csv')
        df = get_stock(ticker, start_date='2016-01-01', end_date='2019-12-31')
        df.to_csv(output_file, index=False)
        print('wrote ' + str(len(df)) + ' lines to file: ' + ticker + '.csv', end = "\n\n" + "-" * 50 + "\n\n")
    except Exception as e:
        print(e)
        print('failed to get Yahoo stock data for ticker: ', ticker, end = "\n\n" + "-" * 50 + "\n\n")
       
    def weekly_return_volatility(data, start_date, end_date):
        """
        calculate the weekly mean return and volatility
        & create a new file to contain these infor
        """
        try:
            df_2 = data[data['Date'] >= start_date]
            df_2 = df_2[df_2['Date'] <= end_date]
            df_2 = df_2[['Year', 'Week_Number', 'Open', 'Adj Close', 'Return']]
            df_2.index = range(len(df_2))
            df_grouped = df_2.groupby(['Year', 'Week_Number'])['Return'].agg([np.mean, np.std])
            df_grouped.reset_index(['Year', 'Week_Number'], inplace=True)
            df_grouped.rename(columns={'mean': 'mean_return', 'std':'volatility'}, inplace=True)
            df_grouped.fillna(0, inplace=True)
            df_grouped["Open"] = df_2.groupby(["Year", "Week_Number"])["Open"].head(1).\
                                 reset_index(drop = True).copy()
            df_grouped["Adj Close"] = df_2.groupby(["Year", "Week_Number"])["Adj Close"].tail(1).\
                                      reset_index(drop = True).copy()
            return df_grouped
        except Exception as error:
            print(error)
            return None
    
    # create the weekly dataframe with mean return and volatility values
    try:
        df_weekly = weekly_return_volatility(df, start_date='2018-01-01', end_date='2019-12-31')
    except Exception as e:
        print("Error in weekly_return_volatility: ", end = " ")
        print(e)
    
    def weekly_label(data, year):
        """
        to create labels
        """
        try:
            df_label = data[data["Year"] == year].copy()
            mean_return_percent50 = np.percentile(df_label["mean_return"], 50)
            volatility_percent50 = np.percentile(df_label["volatility"], 50)      
            df_label["True Label"] = np.where((df_label["mean_return"] >= mean_return_percent50) & \
                                              (df_label["volatility"] <= volatility_percent50), "Green", "Red")
            return df_label
        except Exception as error:
            print(error)
            return None
        
    try:
        df_labeling = pd.DataFrame()
        for year in [2018, 2019]:
            df_year_label = weekly_label(df_weekly, year)
            label_count = df_year_label.groupby("True Label")["True Label"].size().to_frame(name = "Freq")
            print("Label Count for Year {0}".format(year))
            print(tabulate(label_count, headers = "keys", numalign = "right"), end = "\n\n")         
            df_labeling = df_labeling.append(df_year_label, ignore_index = True)
        df_labeling["Week_Number"] = df_labeling["Week_Number"].astype(int)
    except Exception as e:
        print("Error in weekly_label:", end = " ")
        print(e)
    
    def k_means(data, feature, n_clusters):
        feature_data = data[feature].values
        scaler = StandardScaler().fit(feature_data)
        feature_data = scaler.transform(feature_data)
        kmeans_classifier = KMeans(n_clusters = n_clusters)
        
        y_means = kmeans_classifier.fit_predict(feature_data)
        centroids = scaler.inverse_transform(kmeans_classifier.cluster_centers_)
        return y_means, centroids
        
    def k_means_plot(data, feature, centroids, n_clusters, year):
        clusters_col = ["blue", "red", "green", "orange", "brown", "pink", "grey", "tan"]
        plt.figure(figsize = (8, 5))
        
        for i in range(n_clusters):
            plt.scatter(data.loc[data["Cluster"] == i, feature[0]],
                        data.loc[data["Cluster"] == i, feature[1]],
                        color = clusters_col[i], s = 15,
                        label = "Cluster {0}".format(i))
            plt.scatter(centroids[i][0], centroids[i][1], color = clusters_col[i],
                        marker = "X", s = 200, alpha = 0.5, label = "Centroid {0}".format(i))
        
        plt.legend()
        plt.title("* The Distribution of Each Cluster in K-Means Clustering when k = {0} for Year {1} *".\
                  format(n_clusters, year))
        plt.xlabel(feature[0])
        plt.ylabel(feature[1])
        plt.tight_layout()
        plt.show()
    
    ########## Q1 ##########
    print("\n" + "#" * 35 + " Q1 " + "#" * 35 + "\n")
    try:
        # part a: k = 3
        print("*" * 20 + " Part a: k = 3 " + "*" * 20, end = "\n\n")
        df_2018 = df_labeling.loc[df_labeling["Year"] == 2018].copy().reset_index(drop = True)
        k = 3
        y_means, centroids = k_means(df_2018, ["mean_return", "volatility"], k)
        df_2018["Cluster"] = y_means
        data_count = df_2018.groupby("Cluster")["Cluster"].size().to_frame(name = "Number").reset_index()
        kmeans_result = pd.DataFrame({"cluster": data_count["Cluster"],
                                      "centroid": list(centroids.round(3)),
                                      "number of data": data_count["Number"]})
        print(" * The Result of K-Means Clustering when k = 3 * ",
              tabulate(kmeans_result, headers = "keys", numalign = "right"),
              sep = "\n\n", end = "\n\n")        
        k_means_plot(df_2018, ["mean_return", "volatility"], centroids, k, 2018)
        
        # part b: k = 1, 2, ..., 7, 8 and find the best k
        print("\n\n" + "*" * 15 + " Part b: k = 1, 2, ..., 7, 8 and find the best k " + "*" * 15)
        df_2018 = df_labeling.loc[df_labeling["Year"] == 2018].copy().reset_index(drop = True)
        feature_data_2018 = df_2018[["mean_return", "volatility"]].values
        scaler_2018 = StandardScaler().fit(feature_data_2018)
        feature_data_scaled = scaler_2018.transform(feature_data_2018)
        
        # Professor said we can treat inertia and distortion as the same and
        # do not average the sum of distances
        inertias = []
        k_range = range(1, 9)
        
        for k in k_range:
            kmeans_classifier = KMeans(n_clusters = k, random_state = 677)
            kmeans_classifier.fit(feature_data_scaled)
            centroids = scaler_2018.inverse_transform(kmeans_classifier.cluster_centers_)
            inertias.append(kmeans_classifier.inertia_)
            
        fig, ax = plt.subplots(figsize = (8, 4))
        plt.title("* Elbow Method of K-means Clustering *")
        plt.plot(k_range, inertias, marker = "o", color = "green")
        plt.annotate("Knee", xy = (4, inertias[3]), xycoords = "data",
                     xytext = (4.5, inertias[3]+10), color = "black",
                     arrowprops = dict(shrink = 0.005,
                                       color = "black"),
                     ha = "left", va = "bottom")
        plt.xlabel("number of clusters: k")
        plt.ylabel("Inertia")
        plt.tight_layout()
        plt.show()
        
        print("\nAs displayed from the inertia results against different k values, the best k = 4.", end = "\n\n")
    except Exception as e:
        print("Error in Question 1:", end = " ")
        print(e)
        
    ########## Q2 ##########
    print("\n" + "#" * 35 + " Q2 " + "#" * 35 + "\n")
    try:
        k = 4
        df_2018 = df_labeling.loc[df_labeling["Year"] == 2018].copy().reset_index(drop = True)
        df_2019 = df_labeling.loc[df_labeling["Year"] == 2019].copy().reset_index(drop = True)
        feature_data_2018 = df_2018[["mean_return", "volatility"]].values
        
        # train the k-means classifier with Year 2018 data
        scaler_2018 = StandardScaler().fit(feature_data_2018)
        feature_data_2018 = scaler_2018.transform(feature_data_2018)
        kmeans_classifier = KMeans(n_clusters = k, random_state = 677)
        kmeans_classifier.fit(feature_data_2018)
        centroids_2018 = scaler_2018.inverse_transform(kmeans_classifier.cluster_centers_)
        y_means_2018 = kmeans_classifier.predict(feature_data_2018)
        df_2018["Cluster"] = y_means_2018
        k_means_plot(df_2018, ["mean_return", "volatility"], centroids_2018, k, 2018)
                
        # predict the cluster for data in Year 2019
        feature_data_2019 = df_2019[["mean_return", "volatility"]].values
        feature_data_2019 = scaler_2018.transform(feature_data_2019)
        y_means_pred = kmeans_classifier.predict(feature_data_2019)
        
        # compute the percentage of each color of label in each cluster
        df_2019["Cluster"] = y_means_pred
        k_means_plot(df_2019, ["mean_return", "volatility"], centroids_2018, k, 2019)
        kmeans_result_2019 = pd.DataFrame({"Cluster": range(k),
                                           "Centroid": list(centroids_2018.round(3)),
                                           "Number of Data": "",
                                           "Green Label\nperc.(%)": "",
                                           "Red Label\nperc.(%)": ""})
        for i in range(k):
            df_clusters = df_2019[df_2019["Cluster"] == i].copy().reset_index(drop = True)
            kmeans_result_2019.loc[kmeans_result_2019["Cluster"] == i, "Number of Data"] = df_clusters.shape[0]
            kmeans_result_2019.loc[kmeans_result_2019["Cluster"] == i, "Green Label\nperc.(%)"] =\
                (0.00 if df_clusters.shape[0] == 0 else\
                 round(df_clusters[df_clusters["True Label"] == "Green"].shape[0] / df_clusters.shape[0] * 100, 2))
            kmeans_result_2019.loc[kmeans_result_2019["Cluster"] == i, "Red Label\nperc.(%)"] =\
                (0.00 if df_clusters.shape[0] == 0 else\
                 round(df_clusters[df_clusters["True Label"] == "Red"].shape[0] / df_clusters.shape[0] * 100, 2))
        print(tabulate(kmeans_result_2019, headers = "keys", numalign = "right"), end = "\n\n")    
        
    except Exception as e:
        print("Error in Question 2:", end = " ")
        print(e)    
        
    ########## Q3 ##########
    print("\n" + "#" * 35 + " Q3 " + "#" * 35 + "\n")
    try:
        print("As shown from the result, Cluster 1 with 16 data points is a \"pure\" cluster,",
              "which has 100% of labels in that cluster as red labels.", sep = "\n")
    except Exception as e:
        print("Error in Question 3:", end = " ")
        print(e)  

main()
