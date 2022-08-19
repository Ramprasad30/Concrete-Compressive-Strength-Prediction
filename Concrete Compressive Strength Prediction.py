#!/usr/bin/env python
# coding: utf-8

# ## Predicting Compressive Strength of Concrete given its age and quantitative measurements of ingredients.

# In[1]:


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression,  Lasso, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor


# In[2]:


pd.set_option('display.max_rows',100000)
pd.set_option('display.max_columns',1000)


# In[3]:


data = pd.read_excel('Concrete_Data.xls')


# In[4]:


data.head()


# In[5]:


req_col_names = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer","CoarseAggregate", "FineAggregate", "Age", "CC_Strength"]

curr_col_names = list(data.columns)

mapper = {}
for i,name in enumerate(curr_col_names):
  mapper[name] = req_col_names[i]

data = data.rename(columns=mapper)


# In[6]:


data.head()


# In[7]:


data.shape


# ###### DATA CLEANING
# ###### CHECKING FOR NULL VALUES

# In[8]:


data.isnull().sum()


# In[9]:


data.info()


# ###### exploratory data analysis
# ###### exploring the data

# In[11]:


data.describe()


# In[12]:


data.describe().T


# #### checking the pairwise relations
# #### multi variate analysis

# In[13]:


sns.pairplot(data)
plt.show()


# In[14]:


corr = data.corr()

plt.figure(figsize= (10,9))
sns.heatmap(corr, annot = True, cmap = 'Blues')
b, t = plt.ylim()
plt.ylim(b+0.5, t-0.5)
plt.title("Feature Correlation Heatmap")
plt.show()


# #### features in the data

# In[15]:


data.columns


# In[16]:


ax = sns.distplot(data['CC_Strength'])
ax.set_title('Compressive Strength Distribution')


# #### Univariate analysis(PDF, CDF, Boxplot, Voilin plots,Distribution plots)

# In[17]:


counts,bin_edges = np.histogram(data['CC_Strength'],bins = 10, density = True)
print(counts)
print('')
plt.xlabel('CC_Strength')
pdf = counts/sum(counts)
print(pdf)
print('')
print(bin_edges)
plt.plot(bin_edges[1:],pdf)


# In[18]:


cdf = np.cumsum(pdf)
print(cdf)
plt.plot(bin_edges[1:],cdf)


# In[19]:


counts,bin_edges = np.histogram(data['CC_Strength'],bins = 10, density = True)
print(counts)
print('pdf')
plt.xlabel('CC_Strength')
pdf = counts/sum(counts)
print(pdf)
print('bin_edges')
print(bin_edges)
cdf = np.cumsum(pdf)
print('cdf')
print(cdf)
plt.plot(bin_edges[1:],pdf)
plt.plot(bin_edges[1:],cdf)
plt.legend('cumulative distribution function')


# In[20]:


fig,ax = plt.subplots(figsize = (10,7))
sns.scatterplot(y='CC_Strength',x='Cement',hue="Water", size="Age",data=data,ax=ax,sizes= (50,300))
ax.set_title("CC Strength vs (Cement, Age, Water)")
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show()


# In[21]:


fig, ax = plt.subplots(figsize = (10,7))
sns.scatterplot(y="CC_Strength", x="FineAggregate", hue="FlyAsh", size="Superplasticizer", data=data, ax=ax, sizes=(50, 300))
ax.set_title("CC Strength vs (Fine aggregate, Super Plasticizer, FlyAsh)")
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show()


# In[22]:


fig, ax = plt.subplots(figsize = (10,7))
sns.scatterplot(y="CC_Strength", x="FineAggregate", hue="Water", size="Superplasticizer", data=data, ax=ax, sizes=(50, 300))
ax.set_title("CC Strength vs (Fine aggregate, Super Plasticizer, Water)")
ax.legend(loc="upper left", bbox_to_anchor=(1,1))
plt.show()


# #### Data Preprocessing
# #### Separating Input Features and Target Variable.

# In[23]:


X = data.iloc[:,:-1]   
# Features - All columns but last
y = data.iloc[:,-1]          
# Target - Last Column


# #### Splitting data into Training and Test splits

# In[39]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)


# #### Scaling
# #### Standardizing the data i.e. to rescale the features to have a mean of zero and standard deviation of 1.

# In[25]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)


# ## Model Building

# In[26]:


lr = LinearRegression()
# Linear Regression

lasso = Lasso()
# Lasso Regression

ridge = Ridge()
# Ridge Regression


# ### fitting models on Training data

# In[27]:


lr.fit(x_train, y_train)
# fitting the linear regression model


# In[28]:


lasso.fit(x_train, y_train)
# fitting lasso regression model


# In[29]:


ridge.fit(x_train, y_train)
# fitting the ridge regression model


# #### Making predictions on Test data

# In[30]:


y_pred_lr = lr.predict(x_test)
# predicting the test with linear regression model


# In[31]:


y_pred_lasso = lasso.predict(x_test)
# predicting the test with lasso regression model


# In[32]:


y_pred_ridge = ridge.predict(x_test)
# predicting the test with ridge regression model


# #### Evaluation
# Comparing the Root Mean Squared Error (RMSE), Mean Squared Error (MSE), Mean Absolute Error(MAE) and R2 Score.
# linear regression results

# In[33]:


print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),mean_squared_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)))


# #### lasso regression results

# In[34]:


print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lasso)),mean_squared_error(y_test, y_pred_lasso),
            mean_absolute_error(y_test, y_pred_lasso), r2_score(y_test, y_pred_lasso)))


# #### Ridge Regression results

# In[35]:


print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")
print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_ridge)),mean_squared_error(y_test, y_pred_ridge),
            mean_absolute_error(y_test, y_pred_ridge), r2_score(y_test, y_pred_ridge)))


# In[36]:


print("Model\t\t\t RMSE \t\t MSE \t\t MAE \t\t R2")

print("""LinearRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lr)),mean_squared_error(y_test, y_pred_lr),
            mean_absolute_error(y_test, y_pred_lr), r2_score(y_test, y_pred_lr)))

print("""LassoRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_lasso)),mean_squared_error(y_test, y_pred_lasso),
            mean_absolute_error(y_test, y_pred_lasso), r2_score(y_test, y_pred_lasso)))

print("""RidgeRegression \t {:.2f} \t\t {:.2f} \t{:.2f} \t\t{:.2f}""".format(
            np.sqrt(mean_squared_error(y_test, y_pred_ridge)),mean_squared_error(y_test, y_pred_ridge),
            mean_absolute_error(y_test, y_pred_ridge), r2_score(y_test, y_pred_ridge)))


# #### The performance seem to be similar with all the three methods.
# Plotting the coefficients

# In[37]:


coeff_lr = lr.coef_
#linear regression coefficients
coeff_lasso = lasso.coef_
#lasso regression coefficients
coeff_ridge = ridge.coef_
#ridge regression coefficients

labels = req_col_names[:-1]

x = np.arange(len(labels))
width = 0.3

fig,ax = plt.subplots(figsize = (10,7))
rects1 = ax.bar(x - 2*(width/2), coeff_lr, width, label = 'lr')
rects2 = ax.bar(x, coeff_lasso, width, label = 'lasso')
rects3 = ax.bar(x + 2*(width/2), coeff_ridge, width, label = 'ridge')

ax.set_ylabel('coefficient')
ax.set_xlabel('features')
ax.set_title('feature coefficients')
ax.set_xticks(x)
ax.set_xticklabels(labels,rotation = 45)
ax.legend()

def autolabel(rects):
  """Attach a text label above each bar in *rects*, displaying its height."""
  for rect in rects:
    height = rect.get_height()
    ax.annotate('{:.2f}'.format(height), xy = (rect.get_x() + rect.get_width() / 2, height), 
                 xytext = (0, 3), textcoords = 'offset points', ha = 'center', va = 'bottom')

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

fig.tight_layout()
plt.show()


# #### Lasso Regression, reduces the complexity of the model by keeping the coefficients as low as possible.
# Coefficients with Linear and Ridge are almost same.
# Plotting Predictions

# In[38]:


fig, (ax1, ax2, ax3) = plt.subplots(1,3, figsize = (12, 4))

ax1.scatter(y_pred_lr, y_test, s=20)
ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw = 3)
ax1.set_xlabel('True')
ax1.set_ylabel('predicted')
ax1.set_title('Linear regression')

ax2.scatter(y_pred_lasso, y_test, s=20)
ax2.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw = 3)
ax2.set_xlabel('True')
ax2.set_ylabel('predicted')
ax2.set_title('lasso regression')


ax3.scatter(y_pred_ridge, y_test, s=20)
ax3.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw = 3)
ax3.set_xlabel('True')
ax3.set_ylabel('predicted')
ax3.set_title('Ridge regression')

fig.suptitle('True vs predicted')

fig.tight_layout(rect=[0, 0.03, 1, 0.95])


# In[ ]:




