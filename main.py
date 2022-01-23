import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as mp
data=pd.read_csv("https://raw.githubusercontent.com/ProgrammingHero1/predict-iphone-price/main/iphone_price.csv")

# Displaying the graph of Iphone Price upto Model Iphone-12
print(data.head())
mp.scatter(data["version"],data["price"])

mp.show()

# Displaying the Iphone Price of Upcoming Model
n=int(input("Enter Model"))
demo=LinearRegression()
demo.fit(data[["version"]],data[["price"]])
print("The Iphone Price of Model ",n,"is",demo.predict([[n]]))