# Data-Science-with-Stock-Data
In this project, a selected stock's data for two years 2018 and 2019 will be scraped and "green" or "red" label will be assigned by predetermined rules to each week. We will implement a number of machine learning classifiers to predict labels and assess the model performance. For these classifiers, we will also compute the performance of a trading strategy based on the labels.

Our main objects for this project are to explore the characters of each machine learning algorithm and test how they perform when applying in stock data. To better understand the project, we will go through several concepts and explain how we will define them in this instance.

For the stock data, we will label each week as "green" or "red".
- a "green" week means that it was a good week to be invested for that week (from Friday to Friday)
- a "red" week means that it was <b>NOT</b> a good week (e.g. prices fell or there was too much volatility in the price)

