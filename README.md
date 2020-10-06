# Data Science with Stock Data
In this project, a selected stock's data for two years 2018 and 2019 will be scraped and "green" or "red" label will be assigned by predetermined rules to each week. We will implement a number of machine learning classifiers to predict labels and assess the model performance. For these classifiers, we will also compute the performance of a trading strategy based on the labels.

Our main objects for this project are to explore the characters of each machine learning algorithm and test how they perform when applying in stock data. To better understand the project, we will go through several concepts and explain how we will define them in this instance.

For the stock data, we will label each week as "green" or "red".
- a "green" week means that it was a good week to be invested for that week (from Friday to Friday)
- a "red" week means that it was <b>NOT</b> a good week (e.g. prices fell or there was too much volatility in the price)

In addition to colors, we will define a set of numeric features for each week (e.g. average of daily returns and average volatility). Typically, we will train the classifier using these features and labels from year 2018 and then, using features for year 2019, we will predict labels for year 2019. The accuracy of the classifier is the percentage of weeks in year 2019 predicted correctly by the classifier. A correct color is the one that we manually assigned for that week.

We assume that for each week, we know the opening price of the first trading day and the (adjusted) close price of the last trading day in a week. In the strategy based on labels, we will invest in the stock during "green" weeks and keep the money in cash during "red" weeks. There are no short positions (which refers to a trading technique in which an investor sells a security with plans to buy it later, or in this case, to sale a stock you do not own). We start with $100 prior to week 1. The strategy is summarized as follows:
1. for the very first "green" week, we invest $100 by buying (possibly fractional number of) shares of the selected stock at the opening price of the first trading day of that week
2. if the next week is "red" (we want to be out of market next week):
   * if we have a position this week (i.e. this week is "green"), we sell our shares at the adjusted closing price of last trading day of this week
   * if we have no position this week (i.e. this week is "red"), we do nothing and remain in cash for next week
3. if the next week is "green" (we want to be invested next week):
   * if we have a position this week (i.e. this week is "green"), we do nothing and continue to be invested in the stock for next week
   * if we have no position this week (i.e. this week is "red"), we buy (possibly fractional number of) shares of the stock at the opening price of next week - we invest all the money we have accumulated so far
4. ignore trading costs and assume that we are able to buy or sell at open or adjusted closing prices
