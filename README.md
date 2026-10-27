# Smart Segment

A customer segmentation app built with Streamlit and K-Means clustering, created for a Supply Chain Management project (Semester 3).

# What it does

Smart Segment reads retail transaction data from Online_Retail.csv and groups customers into segments based on their buying behavior. It uses four measures: how recently they bought (Recency), how often they order (Frequency), how much they spend (Monetary), and how many items they buy (Total Quantity). The model trains once when the app starts and is then cached for speed.

# Customer segments
Regular Customers: repeat buyers with moderate spend who purchased recently
High-Spending Customers: top accounts with frequent, large orders and the highest revenue
Loyal Customers: frequent bulk buyers who order often and recently
How to use

Keep app.py and Online_Retail.csv together in the same folder, then run the app with Streamlit and open it in your browser at localhost:8501.

# Requirements

Python 3.10, numpy, pandas, streamlit and scikit-learn installed.
