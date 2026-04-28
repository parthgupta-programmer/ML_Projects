import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error

# Reading CSV Data(Data of different months)
df=pd.read_csv("august_data.csv")
df2=pd.read_csv("july_data.csv")
df3=pd.read_csv("june_data.csv")
df4=pd.read_csv("june_data.csv")

# Concatenating into a single dataframe (df)
df=pd.concat([df2, df])
df=pd.concat([df3,df])
df=pd.concat([df4,df])

# Converting Date into datetime and then making a Day column containing the day of that particular date
df["Date"]=pd.to_datetime(df["Date"])
df["Day"]=df["Date"].dt.day_name()
# Since ,saturday and sunday ,college remains off,assuming no student goes on these days.
df=df[(df["Day"]!="Sunday") & (df["Day"]!="Saturday")]

# Now,storing data into X(input) and y(output)
X=df[["Day","Weather","Exam"]]
y=df[["Strength"]]


# Now,Scaling each category of column in terms of numbers so that the computer can learn this way.
preprocessor=ColumnTransformer([("cat",OneHotEncoder(handle_unknown='ignore'),[0,1,2])])
X_scaled_data=preprocessor.fit_transform(X)

# Training the model as per the full scaled data
model=LinearRegression()
model.fit(X_scaled_data,y)

# Testing the data set
X_train, X_test, y_train, y_test=train_test_split(X_scaled_data,y,test_size=0.2,random_state=42)
y_pred_test=model.predict(X_test)

# Calculating for Error Metric Displaying MAE and RMSE
mae=mean_absolute_error(y_test, y_pred_test)
rmse=math.sqrt(mean_squared_error(y_test, y_pred_test))

# Printing the error metrics as per the testing done
print("Error Metrics:\n")
print("Mean Absolute Error (MAE):",mae)
print("Root Mean Square Error (RMSE):",rmse)

# Predicting the whole data after the training has been done.
y_pred=model.predict(X_scaled_data)

# Extracting actual and predicted data for the last week
lastweek_data=df["Date"].between("2025-08-01", "2025-08-07")
y_lastweek=y[lastweek_data]
# x_indx=y_lastweek.index
x_labels = df.loc[lastweek_data, 'Day']
y_pred_lastweek=y_pred[lastweek_data]

# Taking input from the user to enter any date,weather and exam season flag to get the extimated number of the students who may use library on that particular day.
print("Enter Data to get the estimated number of students using library on that particular day:")

print("Date(In YYYY-MM-DD Format):",end="")
date_input=input()
print("Weather:",end="")
weather_input=input()
print("Exam Season(ST-1/ST-2/No):",end="")
exam_input=input()

# Converting date into datetime and getting the day of that particular date
input_date=pd.to_datetime(date_input)
input_day=input_date.day_name()
input_df=pd.DataFrame([[input_day,weather_input,exam_input]],columns=["Day","Weather","Exam"])
input_scaled=preprocessor.transform(input_df)

# Printing the predicted strength on the console.
predicted_strength=int((model.predict(input_scaled))[0][0])
print("Estimated Number of Students:",(predicted_strength))

# Graph for actual v/s predicted data for the last week
print("Graph showing the comparison of last week predicted and actual data:")

# Inputting user command to allow it to show the graph.
print("Enter to see:",end="")
respn=input() # storing user's response

# Plotting the graph
plt.figure(figsize=(7,4))
plt.title("Actual VS Predicted Data")
plt.xlabel("Days -------->")
plt.ylabel("Number of Students --------->")
plt.grid()
plt.plot(x_labels,y_lastweek.values,label="Actual Data",marker="o")
plt.plot(x_labels,y_pred_lastweek,label="Predicted Data",marker="*")
plt.xticks()
plt.yticks()
plt.legend()

# Showing the graph if user press the enter key
if(respn==""):
    plt.show()
else:
    print("Enter Again after running the code again!!")

