import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.simplefilter(action='ignore', category=SettingWithCopyWarning)


#STEP 1
st.write("## Step 1: Load dataset")
st.sidebar.title("Upload Dataset(s)")
uploaded_files = st.sidebar.file_uploader("Choose one or more CSV files", type="csv", accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        df = pd.read_csv(uploaded_file)

st.write(df)

#preprocessing data
del df['url']

encoder=LabelEncoder()

df['sale']=df['sale'].str.replace('%','')
df['sale']=df['sale'].astype(float)
for i in range(len(df['sale'])):
    df['sale'][i] = df['sale'][i]/100

log_price = np.log1p(df['price'])
df['log_price'] = log_price

continuous_df = df.select_dtypes(include=['float64', 'int64'])
categorical_df = df.select_dtypes(include=['object'])


#STEP 2
st.write("## Step 2: Problem statements")

target = df['price']
features = df.drop('price', axis=1)
st.write("The target value to be used for the prediction is 'Price'.")
st.write("The features used for the prediction are 'type', 'sale', 'furniture', 'rate', and 'delivery'")


#STEP 3
st.write("## Step 3: Visualising the distribution of Target variable")
def hist_visualise(df, column_name):
    fig, ax = plt.subplots(figsize=(10,6))
    #plt.figure(figsize=(10,6))
    sns.histplot(df[column_name], bins=50)
    ax.set_title(f'Distribution of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

hist_visualise(df, 'price')
st.write("The graph has a right-skewed / positive distribution. This means that items less than $10,000 has a more higher demand.")


#STEP 4
st.write("## Step 4: Data exploration at basic level")
def data_exploration(df):
    st.write("Missing values:")
    st.write(df.isnull().sum(), end="\n")
    st.write("Duplicate values:")
    st.write(df.duplicated().sum(), end="\n")
    st.write("Unique values:")
    st.write(df.nunique(), end="\n")
    st.write("Summary statistics of the data:")
    st.write(df.describe().T, end="\n")
    st.write("First 10 rows of the data:")
    st.write(df.head(10), end="\n")
    #st.write(df.shape)

dataExploration = data_exploration(df)
#st.write(dataExploration)

# Create a dummy 'date' column
df['date'] = pd.date_range(start='2023-01-01', periods=len(df), freq='D')
df.set_index('date', inplace=True)

# Moving average calculation for price
st.write("Analysis Time Series")
st.write("Trend analysis using moving averages or statistics (7 days window")
df['moving_avg'] = df['price'].rolling(window=7).mean()
df[['price', 'moving_avg']].plot(figsize=(10, 6), title='Price Moving Average (7-day window)')
st.pyplot(plt.gcf())

# Calculate the confidence interval
import scipy.stats
price = df['price']

def mean_confidence_interval(price, confidence=0.95):
    a = 1.0 * np.array(price)
    n = len(a)
    m, se = np.nanmean(a), scipy.stats.sem(a, nan_policy='omit')
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h

mean, lower, upper = mean_confidence_interval(price)

# Create the plot
st.write("Confidence Intervals, data variability and prediction accuracy:")
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar('Price', mean, yerr=[[mean - lower], [upper - mean]], capsize=10, color='skyblue', edgecolor='black')
ax.axhline(mean, color='black', linestyle='--', label=f'Mean: {mean:.2f}')
ax.axhline(lower, color='red', linestyle='--', label=f'Lower: {lower:.2f}')
ax.axhline(upper, color='green', linestyle='--', label=f'Upper: {upper:.2f}')
ax.set_title('95% Confidence Interval for Price')
ax.set_ylabel('Price')
ax.legend()
st.pyplot(fig)


#STEP 5
st.write("## Step 5: Visualising the distribution of Target variable")
def EDA_Visualise(df, column_name):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.histplot(df[column_name], bins=50)
    ax.set_title(f'Distribution of {column_name}')
    ax.set_xlabel(column_name)
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

def bar_chart(df, column_name):
    top_10 = df.nlargest(10, column_name)
    fig, ax = plt.subplots(figsize=(10,6))
    sns.barplot(x=column_name, y=df.columns[0], data=top_10)
    ax.set_title(f'Top 10 of {column_name}:')
    ax.set_xlabel(column_name)
    ax.set_ylabel(df.columns[0])
    st.pyplot(fig)

EDA_Visualise(df, 'price')
EDA_Visualise(df, 'delivery')
EDA_Visualise(df, 'sale')
EDA_Visualise(df, 'rate')

bar_chart(df, 'price')
bar_chart(df, 'delivery')
bar_chart(df,'sale')
bar_chart(df, 'rate')


#STEP 6
st.write("## Step 6: Checking and handling outliers")

#identify outliers with continuous variables
Q1 = continuous_df.quantile(0.25)
Q3 = continuous_df.quantile(0.75)
IQR = Q3 - Q1
outliers = ((continuous_df < (Q1 - 1.5 * IQR)) | (continuous_df > (Q3 + 1.5 * IQR)))
st.write("Outliers:")
st.write(outliers.sum())


#STEP 7
st.write("## Step 7: Missing values analysis")

st.write("Missing values:")
st.write(df.isnull().sum())
#Dropping missing values
df.dropna(inplace=True)


#STEP 8
st.write("## Step 8: Feature selection - Visual and statistic correlation analysis for selection of best features")
st.write("Box Plot")
def boxplot(df, column_name):
    fig, ax = plt.subplots(figsize=(10,6))
    sns.boxplot(x=df[column_name])
    ax.set_title(f'{column_name} Boxplot')
    st.pyplot(fig)

boxplot(df, 'price')
boxplot(df, 'rate')
boxplot(df, 'sale')
boxplot(df, 'delivery')

st.write("Pair Plot")
sns.pairplot(continuous_df)
st.pyplot(plt.gcf())

st.write("Correlation Matrix")
corr_df = continuous_df.drop('price', axis=1)
corr_matrix = corr_df.corr()
st.write(corr_matrix)

fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(corr_matrix, annot=True)
st.pyplot(fig)

#Identify strong, moderate, and weak relationships
target_correlations = corr_matrix['log_price'].drop('log_price')
st.write(target_correlations)
strong_corr = target_correlations[target_correlations.abs() >= 0.7]
moderate_corr = target_correlations[(target_correlations.abs() >= 0.3) & (target_correlations.abs() < 0.7)]
weak_corr = target_correlations[target_correlations.abs() < 0.3]
#st.write("Strong Correlations:")
#st.write(strong_corr)

#st.write("Moderate Correlations:")
#st.write(moderate_corr)

st.write("Weak Correlations:")
st.write(weak_corr)


#STEP 9
st.write("## Step 9: ANOVA test")

#ANOVA test for check if there is any relationship between price and other variables.
from scipy.stats import f_oneway
f_val, p_val = f_oneway(df['price'], df['delivery'], df['sale'], df['rate'], df['log_price'])
st.write("ANOVA test results: F-value =", f_val, ", p-value =", p_val, ", Level of Significance = 0.05")
#If p-value is less than 0.05, we reject the null hypothesis.
if p_val < 0.05:
    st.write("There is a relationship between price and other variables.")
else:
    st.write("There is no relationship between price and other variables.")

#STEP10
st.write("## Step 10: Selecting final predictors/features for building machine learning/AI model")
st.write("In this case, we selected 5 models to train which are: Linear Regression, Decision Tree, Random Forest, K-Nearest Neighbour, and Support Vector Regression.")


#STEP 11
st.write("## Step 11: Data conversion to numeric values for machine learning/predictive analysis")
st.write("As the get_dummies() method was not effective in this case with unexpected characters, such as Arabic characters, we decided to use LabelEncoder from sci-kit learn instead.")

df['type']=encoder.fit_transform(df['type'])
df['furniture'] = encoder.fit_transform(df['furniture'])


#STEP 12 and 13
st.write("## Step 12 and 13: Train/test data split and standardisation/normalisation of data and Investigating multiple regression algorithms")

del df['log_price']
#Splitting the data to predict the price
X = df.drop('price', axis=1)
y = df['price']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Linear Regression
from sklearn.linear_model import LinearRegression
model1 = LinearRegression()
model1.fit(X_train, y_train)

#Decision Tree
from sklearn.tree import DecisionTreeRegressor
model2 = DecisionTreeRegressor()
model2.fit(X_train, y_train)

#Random Forest
from sklearn.ensemble import RandomForestRegressor
model3 = RandomForestRegressor()
model3.fit(X_train, y_train)

#K-Nearest Neighbors
from sklearn.neighbors import KNeighborsRegressor
model4 = KNeighborsRegressor()
model4.fit(X_train, y_train)

#Support Vector Regression
from sklearn.svm import SVR
model5 = SVR()
model5.fit(X_train, y_train)


#Cross-validation
from sklearn.model_selection import cross_val_score
scores1 = cross_val_score(model1, X_train, y_train, cv=5)
scores2 = cross_val_score(model2, X_train, y_train, cv=5)
scores3 = cross_val_score(model3, X_train, y_train, cv=5)
scores4 = cross_val_score(model4, X_train, y_train, cv=5)
scores5 = cross_val_score(model5, X_train, y_train, cv=5)

st.write('Linear Regression:', scores1.mean())
st.write('Decision Tree:', scores2.mean())
st.write('Random Forest:', scores3.mean())
st.write('K-Nearest Neighbors:', scores4.mean())
st.write('Support Vector Regression:', scores5.mean())

#Predicting the price
y_pred1 = model1.predict(X_test)
y_pred2 = model2.predict(X_test)
y_pred3 = model3.predict(X_test)
y_pred4 = model4.predict(X_test)
y_pred5 = model5.predict(X_test)

#Evaluate the models
from sklearn.metrics import mean_squared_error, r2_score
mse1 = mean_squared_error(y_test, y_pred1)
mse2 = mean_squared_error(y_test, y_pred2)
mse3 = mean_squared_error(y_test, y_pred3)
mse4 = mean_squared_error(y_test, y_pred4)
mse5 = mean_squared_error(y_test, y_pred5)

r2_score1 = r2_score(y_test, y_pred1)
r2_score2 = r2_score(y_test, y_pred2)
r2_score3 = r2_score(y_test, y_pred3)
r2_score4 = r2_score(y_test, y_pred4)
r2_score5 = r2_score(y_test, y_pred5)

"""Mean Squared Errors"""
st.write('Mean Squared Errors of Models')
st.write('Linear Regression MSE:', mse1)
st.write('Tree Decision MSE:', mse2)
st.write('Random Forest Decision MSE:', mse3)
st.write('K-Nearest Neighbors MSE:', mse4)
st.write('Support Vector Regression MSE:', mse5)

"""R-squared Scores"""
st.write("R-squared Scores of models")
st.write('Linear Regression R-squared:', r2_score1)
st.write('Tree Decision R-squared:', r2_score2)
st.write('Random Forest Decision R-squared:', r2_score3)
st.write('K-Nearest Neighbors R-squared:', r2_score4)
st.write('Support Vector Regression R-squared:', r2_score5)

#Evaluate the models using cross-validation
from sklearn.model_selection import cross_val_score

cross_val_mse1 = cross_val_score(model1, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_mse2 = cross_val_score(model2, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_mse3 = cross_val_score(model3, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_mse4 = cross_val_score(model4, X, y, cv=5, scoring='neg_mean_squared_error')
cross_val_mse5 = cross_val_score(model5, X, y, cv=5, scoring='neg_mean_squared_error')

cross_val_mse1 = cross_val_mse1 * -1
cross_val_mse2 = cross_val_mse2 * -1
cross_val_mse3 = cross_val_mse3 * -1
cross_val_mse4 = cross_val_mse4 * -1
cross_val_mse5 = cross_val_mse5 * -1

st.write('Cross-Validation Mean Squared Errors of Models')

st.write('Linear Regression Cross-Validation MSE:', cross_val_mse1)
st.write('Decision Tree Cross-Validation MSE:', cross_val_mse2)
st.write('Random Forest Decision Cross-Validation MSE:', cross_val_mse3)
st.write('K-Nearest Neighbors Cross-Validation MSE:', cross_val_mse4)
st.write('Support Vector Regression Cross-Validation MSE:', cross_val_mse5)

#Save the models
import joblib
joblib.dump(model1, 'linear_regression_model.pkl')
joblib.dump(model2, 'decision_tree_model.pkl')
joblib.dump(model3, 'random_forest_model.pkl')
joblib.dump(model4, 'knn_model.pkl')
joblib.dump(model5, 'svr_model.pkl')

#Coefficients
coefficients = pd.DataFrame({'Feature': X.columns,
                             'Linear Regression Coefficient': model1.coef_[0],
                             'Decision Tree Coefficient': model2.feature_importances_,
                             'Random Forest Coefficient': model3.feature_importances_})
coefficients.sort_values(by='Linear Regression Coefficient', ascending=False, inplace=True)
st.write("Coefficients:")
st.write(coefficients)

#Regression metrics
models = [model1, model2, model3, model4, model5]
"""Regression metrics"""
regression_metrics = pd.DataFrame({'Model': models,
                                 'Mean Squared Error': [mean_squared_error(y_test, y_pred1),
                                                        mean_squared_error(y_test, y_pred2),
                                                        mean_squared_error(y_test, y_pred3),
                                                        mean_squared_error(y_test, y_pred4),
                                                        mean_squared_error(y_test, y_pred5)]})

regression_metrics.sort_values(by='Mean Squared Error', ascending=False, inplace=True)
st.write("Regression Metrics")
st.write(regression_metrics)


#STEP 14
st.write("## Step 14: Selection of the best model")

#Selection of best models based on the lowest mean square error
best_models = ['Linear Regression',
               'Decision Tree',
               'Random Forest',
               'K-Nearest Neighbors',
               'Support Vector Regression']
best_mse = [mse1, mse2, mse3, mse4, mse5]
best_r2_score = [r2_score1, r2_score2, r2_score3, r2_score4, r2_score5]

best_comparison_df = pd.DataFrame({'Model': best_models,
                                   'MSE': best_mse,
                                   'R-squared': best_r2_score})
best_comparison_df.sort_values(by=['MSE'], ascending=True, inplace=True)

#Conclusion
st.write('Best models based on Lowest MSE:')
st.write(best_comparison_df)


#STEP 15
st.write("## Step 15: Deployment of the best model in production")
st.write("Load the save Random Forest model, repredict, and evaluate with 100% of the data")

#Load the models
loaded_model3 = joblib.load('random_forest_model.pkl')

#Make predictions with 100% of data
X = df.drop(columns=['price'])
y = df['price']
true_prediction = loaded_model3.predict(X)
st.write("Prediction:")
st.write(f"{np.mean(true_prediction):.4f}")

#Evaluate the loaded models
loaded_mse3 = mean_squared_error(y, true_prediction)

loaded_r2_score3 = r2_score(y, true_prediction)

st.write("MSE:", loaded_mse3)
st.write("R-squared:", loaded_r2_score3)

#Visualise the prediction
fig, ax = plt.subplots()
ax.scatter(y, true_prediction)
ax.set_title('Actual vs Predicted Prices')
ax.set_xlabel('Actual Price')
ax.set_ylabel('Predicted Price')
st.pyplot(fig)

#Make the residual plot for the prediction and the actual price
residuals = y - true_prediction
fig, ax = plt.subplots()
ax.scatter(true_prediction, residuals)
ax.axhline(y=0, color='r', linestyle='--')
ax.set_title('Residual Plot')
ax.set_xlabel('Predicted Price')
ax.set_ylabel('Residuals')
st.pyplot(fig)

#Histogram of the residuals
fig, ax = plt.subplots()
ax.hist(residuals, bins=50)
ax.set_title('Histogram of Residuals')
ax.set_xlabel('Residuals')
ax.set_ylabel('Frequency')
st.pyplot(fig)


#Save Random Forest model as the best model
best_model = loaded_model3
joblib.dump(best_model, 'best_model.pkl')