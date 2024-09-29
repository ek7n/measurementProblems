###################################################
# PROJECT: Rating Product & Sorting Reviews on Amazon
###################################################

###################################################
# Business Problem
###################################################

# One of the most critical problems in e-commerce is accurately calculating the post-sales ratings given to products.
# Solving this problem means increased customer satisfaction for the e-commerce site, increased product visibility for sellers,
# and a seamless shopping experience for buyers. Another issue is accurately sorting the reviews given to products.
# Since misleading reviews can directly affect product sales, they can lead to financial loss and customer loss.
# By addressing these two main problems, the e-commerce site and sellers will increase sales, while customers will have a smooth purchase journey.

###################################################
# Dataset Information
###################################################

# This dataset, which contains Amazon product data, includes various metadata with product categories.
# It has user ratings and reviews for the most reviewed product in the electronics category.

# Variables:
# reviewerID: User ID
# asin: Product ID
# reviewerName: Username
# helpful: Helpfulness rating of the review
# reviewText: Review text
# overall: Product rating
# summary: Summary of the review
# unixReviewTime: Review time (Unix format)
# reviewTime: Raw review time
# day_diff: Number of days since the review
# helpful_yes: Number of helpful votes for the review
# total_vote: Total votes given to the review

import pandas as pd
from sklearn.preprocessing import MinMaxScaler
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 500)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)
import scipy.stats as st
import math

###################################################
# TASK 1: Calculate the Average Rating Based on Current Reviews and Compare with Existing Average Rating
###################################################

# Users have rated and reviewed a product in the shared dataset.
# The goal here is to evaluate the ratings by weighting them according to date.
# The first average rating should be compared with the time-based weighted rating.

###################################################
# Step 1: Read the Dataset and Calculate the Average Rating of the Product
###################################################
df = pd.read_csv("amazon_review.csv")
df.info()
df["overall"].mean()

###################################################
# Step 2: Calculate the Time-Based Weighted Average Rating
###################################################

df.describe().T
df["overall"].value_counts()
df["unixReviewTime"].value_counts()
df["reviewTime1"] = pd.to_datetime(df["unixReviewTime"], unit="s")

df["reviewTime"].describe().T
df["reviewTime"].max()
current_date = pd.to_datetime("2014-07-25 00:00:00")
df["days"] = (current_date - df["reviewTime"]).dt.days
df["days"].describe([0, 0.25, 0.50, 0.75, 0.90, 0.95]).T

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

# while the weights can be determined by business logic, using quartiles are also an option
def quartiles_time_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[df["days"] <= df["days"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > df["days"].quantile(0.25)) & (dataframe["days"] <= df["days"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > df["days"].quantile(0.50)) & (dataframe["days"] <= df["days"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > df["days"].quantile(0.75)), "overall"].mean() * w4 / 100

quartiles_time_based_weighted_average(df)

###################################################
# TASK 2: Determine the Top 20 Reviews to Display on the Product Detail Page
###################################################

###################################################
# Step 1: Create the helpful_no Variable
###################################################

# Note:
# total_vote represents the total up-down vote count for a review.
# up means helpful.
# The variable helpful_no does not exist in the dataset and needs to be created using existing variables.

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df.info()
df["helpful_yes"].value_counts()
df["total_vote"].value_counts()

df[["helpful_yes", "total_vote"]].head(100)

df.loc[df["total_vote"] > 100]

###################################################
# Step 2: Calculate and Add score_pos_neg_diff, score_average_rating, and wilson_lower_bound Scores to the Data
###################################################

df["score_pos_neg_diff"] = df["helpful_no"] - df["helpful_yes"]

def score_average_rating(df, up, down):
    df["score_average_rating"] = df[up] / (df[up] + df[down])
    df.loc[df[up] + df[down] == 0, "score_average_rating"] = 0
    return df

score_average_rating(df, "helpful_yes", "helpful_no")

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Calculate Wilson Lower Bound Score

    - The lower bound of the confidence interval for the Bernoulli parameter p is considered as the WLB score.
    - The calculated score is used for product ranking.
    - Note:
    If scores range from 1 to 5, ratings of 1-3 can be marked as negative and 4-5 as positive to fit into a Bernoulli distribution.
    This might bring certain challenges; hence, using a Bayesian average rating is suggested.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence level

    Returns
    -------
    wilson score: float
    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)

df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df["wilson_lower_bound"].describe().T

##################################################
# Step 3: Identify the Top 20 Reviews and Interpret the Results
###################################################

df.sort_values(by="wilson_lower_bound", ascending=False).head(20)[["reviewerID", "reviewerName", "wilson_lower_bound"]]

id = "A12B7ZMXFI6IXY"

df[df["reviewerID"] == id]
