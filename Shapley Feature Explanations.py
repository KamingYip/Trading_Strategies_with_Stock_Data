"""
Kaming Yip
CS677 A1 Data Science with Python
Mar 20, 2020
Assignment 7.2: Shapley Feature Explanations
"""

from pandas_datareader import data as web
import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


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
        
    def Logistic_Reg(train_data, test_data, predictor):
        # train the logistic regression model by stock data in year 1
        train_X = train_data[predictor].values
        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        train_Y = train_data["True Label"].values
        classifier = LogisticRegression(solver = "lbfgs")
        classifier.fit(train_X, train_Y)
        
        # predict the labels in year 2
        test_X = test_data[predictor].values
        test_X = scaler.transform(test_X)
        pred_Y = classifier.predict(test_X)
        return pred_Y
    
    def kNN(train_data, test_data, predictor, num_neighbors):
        # train the kNN model by stock data in year 1
        train_X = train_data[predictor].values  
        scaler = StandardScaler().fit(train_X)
        train_X = scaler.transform(train_X)
        train_Y = train_data["True Label"].values
        classifier = KNeighborsClassifier(n_neighbors = num_neighbors, p = 2)
        classifier.fit(train_X, train_Y)
        
        # predict the labels in year 2
        test_X = test_data[predictor].values
        test_X = scaler.transform(test_X)
        pred_Y = classifier.predict(test_X)
        return pred_Y
    
    def Linear_model(train_data, test_data, predictor):
        # train the linear model by stock data in year 1
        train_X = train_data[predictor].values
        train_Y = train_data["Adj Close"].values
        lin_reg = LinearRegression(fit_intercept = True)
        lin_reg.fit(train_X, train_Y)
        
        # predict the labels in year 2
        predicted_labels = []
        prev_label = "None"
        for i in range(len(test_data)):
            test_X = np.array(test_data.iloc[i][predictor]).reshape(1, -1)
            pred_Y = lin_reg.predict(test_X)
            if i == 0:
                prev_price = train_data.iloc[-1]["Adj Close"]
            else:
                prev_price = test_data.iloc[i - 1]["Adj Close"]
            if pred_Y > prev_price:
                prev_label = "Green"
                predicted_labels.append(prev_label)
            elif pred_Y < prev_price:
                prev_label = "Red"
                predicted_labels.append(prev_label)
            else:
                if prev_label == "None":
                    predicted_labels.append(train_data.iloc[-1]["True Label"])
                else:
                    predicted_labels.append(prev_label)
        return np.asarray(predicted_labels)
    
    def AccuracyCal(actual, pred):
        cm = confusion_matrix(actual, pred)
        diagonal_sum = cm.trace()
        sum_of_all_elements = cm.sum()
        accuracy = diagonal_sum / sum_of_all_elements
        return accuracy
    
    def Iris_Accuracy(data, feature_list):
        X = data[feature_list].values
        le = LabelEncoder()
        Y = le.fit_transform(data["Bi-Class"].values)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.5, 
                                                            random_state = 0)
        log_reg_classifier = LogisticRegression(solver = "lbfgs")
        log_reg_classifier.fit(X_train, Y_train)
        prediction = log_reg_classifier.predict(X_test)
        accuracy = np.mean(prediction == Y_test)
        return accuracy
        
    
    ########## Q1 ##########
    print("\n" + "#" * 35 + " Q1 " + "#" * 35 + "\n")
    try:
        df_2018 = df_labeling.loc[df_labeling["Year"] == 2018].copy().reset_index(drop = True)
        df_2019 = df_labeling.loc[df_labeling["Year"] == 2019].copy().reset_index(drop = True)
        accuracy_table = pd.DataFrame(columns = ["Accuracy", "Accuracy\n(w/o μ)", "Accuracy\n(w/o σ)",
                                                 "Delta_1\n(w/o μ)", "Delta_2\n(w/o σ)"],
                                      index = ["Log. Reg", "k-NN", "Linear Model"])
        actual_labels = df_2019["True Label"].values
        
        # different models' accuracy with different predictors
        predictor_list = [["mean_return", "volatility"], ["mean_return"], ["volatility"]]
        num_neighbors = 3
        for i in range(len(predictor_list)):
            features = predictor_list[i]
            log_pred = Logistic_Reg(df_2018, df_2019, features)
            kNN_pred = kNN(df_2018, df_2019, features, num_neighbors)
            lin_pred = Linear_model(df_2018, df_2019, features)
            pred_table = [log_pred, kNN_pred, lin_pred]
            accuracy_list = []
            for ele in pred_table:
                accuracy_list.append(AccuracyCal(actual_labels, ele))
            accuracy_table.iloc[:, i] = accuracy_list
        accuracy_table["Delta_1\n(w/o μ)"] = accuracy_table["Accuracy"] - accuracy_table["Accuracy\n(w/o μ)"]
        accuracy_table["Delta_2\n(w/o σ)"] = accuracy_table["Accuracy"] - accuracy_table["Accuracy\n(w/o σ)"]
        
        print("\n" + " " * 10 + "Different Models\' Accuracy with Different Predictors\n")
        print(tabulate(accuracy_table.round(3), headers = "keys", numalign = "right"), end = "\n\n") 
        
        print("As displayed in the table above, the marginal contributions of μ and σ",
              "are similar in both logistic regression model and linear model. Specially,",
              "those two features have very limited contributions to accuracy in linear",
              "model, which I think the adjusted closing prices in 2019 are not in good",
              "fit with the linear model generated by 2018 stock data. In other words,",
              "it is not satisfied to predict 2019 labels by linear model generated by",
              "features μ and σ in 2018.\n",
              "However, in k-NN model, it is obviously shown that σ has a higher contribution",
              "than μ in predicting labels in 2019, which it is fair to say that, in this",
              "scenario, σ is more important than μ in predicting labels.",
              sep = "\n")
    
    except Exception as e:
        print("Error in Question 1:", end = " ")
        print(e)
    
    ########## Q2 ##########
    print("\n" + "#" * 35 + " Q2 " + "#" * 35 + "\n")
    try:
        url = r"https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
        iris_data = pd.read_csv(url, names=["sepal-length", "sepal-width", 
                                            "petal-length", "petal-width", "Class"])
        
        features = ["sepal-length", "sepal-width", "petal-length", "petal-width"]
        class_labels = ["Iris-versicolor", "Iris-setosa", "Iris-virginica"]
        contribution_table = pd.DataFrame(columns = ["Versicolor", "Setosa", "Virginica"],
                                          index = [i + " △" for i in features])
        
        for i in range(len(class_labels)):
            one_class = class_labels[i]
            data = iris_data.copy()
            data["Bi-Class"] = data["Class"].apply(lambda x: 1 if x == one_class else -1)
            
            # accuracy with all features
            A = Iris_Accuracy(data, features)
            
            marginal_contributions = []
            for j in range(len(features)):
                feature_rm = features[ : j] + features[j+1 : ]
                # accuracy without one of the features
                A_j = Iris_Accuracy(data, feature_rm)
                marginal_contributions.append(A - A_j)
            contribution_table.iloc[:, i] = marginal_contributions
        
        print("\nMarginal Contributions of Different Features in Iris Data\n")
        print(tabulate(contribution_table.round(3), headers = "keys", numalign = "right"), end = "\n\n")        
        
        print("As shown in the table above, it is interesting to see that",
              "when taking Setosa as one class and the other two species as",
              "another class, all of those 4 features have no contributions",
              "to the accuracy in predicting the flower class, which indicates",
              "that Setosa can be easily seperated from the other two species.\n",
              "On the other hand, when taking Versicolor as one single class",
              "and the other two species as another class, it seems that all",
              "features contribute in classifying the Versicolor from the other",
              "class. The result indicates that all 4 features of Versicolor can,",
              "in some degree, help in distinguishing itself from the other species.",
              sep = "\n")
        
    except Exception as e:
        print("Error in Question 2:", end = " ")
        print(e)
   
main()
