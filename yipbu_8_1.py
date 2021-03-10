"""
Kaming Yip
CS677 A1 Data Science with Python
Mar 28, 2020
Assignment 8.1: tips
"""

import os
import pandas as pd
import numpy as np
from tabulate import tabulate
from scipy import stats
import warnings
warnings.simplefilter("ignore")

def main():
    try:
        ticker = "tips"
        input_dir = os.getcwd()
        input_file = os.path.join(input_dir, ticker + ".csv")
        tips = pd.read_csv(input_file)
    except Exception as e:
        print(e)
        print("failed to get data from file", ticker + ".csv", end = "\n\n" + "-" * 50 + "\n\n")  

    ########## Q1 ##########
    print("\n" + "#" * 35 + " Q1 " + "#" * 35 + "\n")
    try:
        tips["tip_percent"] = 100 * tips["tip"] / tips["total_bill"]
        avg_lunch_tip_percent = np.mean(tips.loc[tips["time"] == "Lunch", "tip_percent"])
        avg_dinner_tip_percent = np.mean(tips.loc[tips["time"] == "Dinner", "tip_percent"])
        print("The average tip (as a percentage of meal cost) for lunch is {0}%.".\
              format(round(avg_lunch_tip_percent, 2)),
              "The average tip (as a percentage of meal cost) for dinner is {0}%.".\
              format(round(avg_dinner_tip_percent, 2)), sep = "\n")
    except Exception as e:
        print("Error in Question 1:", end = " ")
        print(e)
        
    ########## Q2 ##########
    print("\n" + "#" * 35 + " Q2 " + "#" * 35 + "\n")
    try:
        avg_tip_day = tips.groupby("day")["tip_percent"].agg(np.mean).\
                      reindex(["Thur", "Fri", "Sat", "Sun"]).\
                      to_frame(name = "Avg Tip Percent(%)")
        print(tabulate(avg_tip_day.round(2), headers = "keys", numalign = "right"))
    except Exception as e:
        print("Error in Question 2:", end = " ")
        print(e)    
    
    ########## Q3 ##########
    print("\n" + "#" * 35 + " Q3 " + "#" * 35 + "\n")
    try:
        avg_tip_day_time = tips.groupby(["day", "time"])["tip_percent"].\
                           agg([("Count", np.size), ("Avg Tip Percent(%)", np.mean),
                                ("Max Tip Percent(%)", np.max), ("Min Tip Percent(%)", np.min)]).\
                           sort_values(by = "Avg Tip Percent(%)", ascending = False).\
                           reset_index()
        print(tabulate(avg_tip_day_time.round(2), headers = "keys", numalign = "right"), end = "\n\n")
        print("As shown, {0} {1} has the highest average tip percent as {2}%.".\
              format(avg_tip_day_time.loc[avg_tip_day_time["Avg Tip Percent(%)"].idxmax(), "day"],
                     avg_tip_day_time.loc[avg_tip_day_time["Avg Tip Percent(%)"].idxmax(), "time"],
                     round(avg_tip_day_time["Avg Tip Percent(%)"].max(), 2)))
    except Exception as e:
        print("Error in Question 3:", end = " ")
        print(e)   

    ########## Q4 ##########
    print("\n" + "#" * 35 + " Q4 " + "#" * 35 + "\n")
    try:
        cor_prices_tips = tips["total_bill"].corr(tips["tip_percent"], method = "pearson")
        print("The correlation between meal prices and tip percentages is {0:.2f}.".format(cor_prices_tips))
    except Exception as e:
        print("Error in Question 4:", end = " ")
        print(e)   

    ########## Q5 ##########
    print("\n" + "#" * 35 + " Q5 " + "#" * 35 + "\n")
    try:
        cor_size_tips = tips["size"].corr(tips["tip_percent"], method = "pearson")
        print("The correlation between size of the group and tip percentages is {0:.2f}.".format(cor_size_tips))
    except Exception as e:
        print("Error in Question 5:", end = " ")
        print(e)   

    ########## Q6 ##########
    print("\n" + "#" * 35 + " Q6 " + "#" * 35 + "\n")
    try:
        smokers = tips.groupby("smoker")["size"].agg([("Total", np.sum)]).reset_index()
        smokers["Percent(%)"] = 100 * smokers["Total"] / np.sum(smokers["Total"])
        print(tabulate(smokers.round(2), headers = "keys", numalign = "right"))
    except Exception as e:
        print("Error in Question 6:", end = " ")
        print(e)

    ########## Q7 ##########
    print("\n" + "#" * 35 + " Q7 " + "#" * 35 + "\n")
    try:
        day_count = 0
        for i in range(len(tips)):
            if i == 0:
                weekday = tips.loc[i, "day"]
                tips.loc[i, "day_number"] = day_count
            else:
                if tips.loc[i, "day"] == weekday:
                    tips.loc[i, "day_number"] = day_count
                else:
                    day_count += 1
                    tips.loc[i, "day_number"] = day_count
                    weekday = tips.loc[i, "day"]
        
        day_tip = pd.DataFrame(columns = ["Day Count", "day", "Correlation"])
        for i in range(day_count + 1):
            day_meal = tips.loc[tips["day_number"] == i].reset_index()
            meal_num = pd.Series(range(len(day_meal)))
            #cor_day_tip = stats.pearsonr(day_meal["tip_percent"], meal_num)[0]
            cor_day_tip = day_meal["tip_percent"].corr(meal_num, method = "pearson")
            day_tip = day_tip.append({"Day Count": i, "day": day_meal["day"].unique()[0],
                                      "Correlation": cor_day_tip},
                                     ignore_index = True)
        day_tip["Tips Increasing with Time"] = day_tip["Correlation"].apply(lambda x: "True" if x > 0.5 else "False")
        print(tabulate(day_tip.round(3), headers = "keys", numalign = "right"))        
    except Exception as e:
        print("Error in Question 7:", end = " ")
        print(e)

    ########## Q8 ##########
    print("\n" + "#" * 35 + " Q8 " + "#" * 35 + "\n")
    try:
        smoker_tip = tips.loc[tips["smoker"] == "Yes", "tip"]
        non_smoker_tip = tips.loc[tips["smoker"] == "No", "tip"]
        p_value = stats.ttest_ind(smoker_tip, non_smoker_tip)[1]
        print("With T-test for the two groups of customers (smoker vs. non-smoker), the",
              "p-value for this test is {0:.5f}. Therefore, at the α = 0.05 level of".format(p_value),
              "significance, there is no sufficient evidence to conclude that there is",
              "a difference in the average number of tip amounts from smokers and non-smokers.", sep = "\n")
    except Exception as e:
        print("Error in Question 8:", end = " ")
        print(e)

main()
