"""
Kaming Yip
CS677 A1 Data Science with Python
Apr. 19, 2020
Assignment 11.2: Custom K-means
"""

from pandas_datareader import data as web
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

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

    class Custom_kmeans:
        def __init__(self, n_clusters, tolerance = 0.0001, max_iterations = 500, distance_parameter_p = 2):
            self.k = n_clusters
            self.tolerance = tolerance
            self.max_iterations = max_iterations
            self.p = distance_parameter_p
            self.inertia = 0
        
        def fit(self, data):
            self.centroids = {}
            
            # initialize the first k data point in the dataset as the origianl
            # k number of random centroids, which can make sure the same initial
            # choice of centroids would not change for different distance metrics
            for i in range(self.k):
                self.centroids[i] = data[i]
            
            # find the optimal centroids to cluster
            for i in range(self.max_iterations):
                self.clusters = {}
                for i in range(self.k):
                    self.clusters[i] = []
                
                # calculate the distances for each data point and each centroid,
                # and assign the data point to cluster with the nearest centroid
                for features in data:
                    distances = [np.linalg.norm(features - self.centroids[centroid], ord = self.p)\
                                 for centroid in self.centroids]
                    clustering = distances.index(min(distances))
                    self.clusters[clustering].append(features)
                
                prev_centroids = dict(self.centroids)
                
                # recalculate the centoirds
                for clustering in self.clusters:
                    self.centroids[clustering] = np.average(self.clusters[clustering], axis = 0)
                
                # set a flag
                isOptimal = True
                
                # evaluate whether the current centroids are optimal enough (by tolerance)
                for centroid in self.centroids:
                    original_centroid = prev_centroids[centroid]
                    current_centroid = self.centroids[centroid]
                    if np.sum((current_centroid - original_centroid)/original_centroid * 100.0) > self.tolerance:
                        isOptimal = False
                
                # break out of the loop if the centroids are optimal, i.e. the centroids
                # don't change their positions much comparing to the previous set
                if isOptimal:
                    break
            
            # calculate the inertia of the final result
            for i in range(self.k):
                centroid = self.centroids[i]
                cluster_points = self.clusters[i]
                for cluster_point in cluster_points:
                    self.inertia += (np.linalg.norm(cluster_point - centroid, ord = self.p))**2
            
        def predict(self, data):
            pred = []
            for point in data:
                 distances = [np.linalg.norm(point - self.centroids[centroid], ord = self.p)\
                              for centroid in self.centroids]
                 clustering = distances.index(min(distances))
                 pred.append(clustering)
            return pred
    
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
    
    def k_means_apply(train_data, test_data, features, n_clusters, p, year):
        train = train_data.copy()
        test = test_data.copy()
        train_feature_data = train[features].values
        test_feature_data = test[features].values
        
        # train the k-means classifier with year 1 data
        scaler = StandardScaler().fit(train_feature_data)
        train_feature_data = scaler.transform(train_feature_data)
        test_feature_data = scaler.transform(test_feature_data)
        
        # apply custom k-means clustering with assigned distance metric
        kmeans_classifier = Custom_kmeans(n_clusters = n_clusters, distance_parameter_p = p)
        kmeans_classifier.fit(train_feature_data)
        centroids = scaler.inverse_transform(list(kmeans_classifier.centroids.values()))
        y_means_pred = kmeans_classifier.predict(test_feature_data)
        test["Cluster"] = y_means_pred
        print("*" * 35 + " p = {0} ".format(p) + "*" * 35)
        k_means_plot(test, ["mean_return", "volatility"], centroids, n_clusters, year)
        kmeans_result = pd.DataFrame({"Cluster": range(n_clusters),
                                      "Centroid": list(centroids.round(3)),
                                      "Number of Data": "",
                                      "Green Label\nperc.(%)": "",
                                      "Red Label\nperc.(%)": ""})
        for i in range(n_clusters):
            df_clusters = test[test["Cluster"] == i].copy().reset_index(drop = True)
            kmeans_result.loc[kmeans_result["Cluster"] == i, "Number of Data"] = df_clusters.shape[0]
            kmeans_result.loc[kmeans_result["Cluster"] == i, "Green Label\nperc.(%)"] =\
                (0.00 if df_clusters.shape[0] == 0 else\
                 round(df_clusters[df_clusters["True Label"] == "Green"].shape[0] / df_clusters.shape[0] * 100, 2))
            kmeans_result.loc[kmeans_result["Cluster"] == i, "Red Label\nperc.(%)"] =\
                (0.00 if df_clusters.shape[0] == 0 else\
                 round(df_clusters[df_clusters["True Label"] == "Red"].shape[0] / df_clusters.shape[0] * 100, 2))
        print(tabulate(kmeans_result, headers = "keys", numalign = "right"), end = "\n" * 4)
    
    ########## Custom K-means Clustering ##########
    print("\n" + "#" * 25 + " Custom K-means Clustering " + "#" * 25 + "\n")
    try:
        k = 4
        df_2018 = df_labeling.loc[df_labeling["Year"] == 2018].copy().reset_index(drop = True)
        df_2019 = df_labeling.loc[df_labeling["Year"] == 2019].copy().reset_index(drop = True)
        
        # Apply different distance metrics: p = 1, 1.5, 2
        p_list = [1, 1.5, 2]
        for p in p_list:
            k_means_apply(df_2018, df_2019, ["mean_return", "volatility"], k, p, 2019)
        
        print("*" * 35 + " Summary " + "*" * 35 + "\n",
              "As shown in the results of custom k-means clustering against different distance",
              "metrics (i.e. p = 1, 1.5, 2), the cluster set remain exactly the same when applying",
              "Euclidean distance metric (p = 2) and Minkowski p = 1.5 metric, while the cluster",
              "set for Manhattan distance metric (p = 1) is different from the other two metrics.",
              "In Euclidean distance metric and Minkowski p = 1.5 metric, there are two clusters",
              "among the total 4 clusters that are \"pure\", both as 100%, for Year 2019 data,",
              "while there are 3 clusters out of 4 clusters as 100% pure in Manhattan distance",
              "metric. Therefore, the Manhattan distance metric gives the most \"pure\" clusters.",
              sep = "\n")
        
    except Exception as e:
        print("Error:", end = " ")
        print(e)

main()
