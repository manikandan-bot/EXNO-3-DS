## EXNO-3-DS

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
Developed by:T.Manikandan
Register number:212224110037
```
```
import pandas as pd
df=pd.read_csv('drive/MyDrive/Encoding Data.csv')
df
```
![image](https://github.com/user-attachments/assets/50d9bd59-37fa-48ec-a528-2e41a216e1a2)
```
from sklearn.preprocessing import LabelEncoder,OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[["ord_2"]])
```
![image](https://github.com/user-attachments/assets/2a94df8d-ff1f-4144-afb1-815d687ae873)
```
df['bo2']=e1.fit_transform(df[["ord_2"]])
df
```
![image](https://github.com/user-attachments/assets/c61c09d5-bcac-489b-a13d-78e96a5e7a46)
```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc
```
![image](https://github.com/user-attachments/assets/c23d5b83-2a82-4b88-bc17-444495db882a)
```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False) # Change 'sparse' to 'sparse_output'
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[["nom_0"]]))
enc
```
![image](https://github.com/user-attachments/assets/92220f77-ae30-4cf4-ba77-f453b3d789eb)
```
df2=pd.concat([df2,enc],axis=1)
df2
```
![image](https://github.com/user-attachments/assets/4176faf1-8fe0-4d08-9d3c-ba9061913fa9)
```
pd.get_dummies(df2,columns=["nom_0"])
```
![image](https://github.com/user-attachments/assets/b53786a0-4ed3-4ef6-a6db-d290e141ad97)
```
pip install --upgrade category_encoders
```
![image](https://github.com/user-attachments/assets/a82e05b3-ad40-4145-bb2a-0d86e82e4eb9)
```
import pandas as pd
df=pd.read_csv('drive/MyDrive/data.csv')
df
```
![image](https://github.com/user-attachments/assets/b6f28783-ff6e-40b2-ae41-920bdd71499c)
```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```
![image](https://github.com/user-attachments/assets/88734f9e-0035-490d-88ec-df7f289671a7)
```
from category_encoders import TargetEncoder
te=TargetEncoder()
CC=df.copy()
new=te.fit_transform(X=CC["City"],y=CC["Target"])
CC=pd.concat([CC,new],axis=1)
CC
```
![image](https://github.com/user-attachments/assets/cd2d2387-4cb6-44de-bd13-a6a2d15b1ca7)
```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("drive/MyDrive/Data_to_Transform.csv")
df
```
![image](https://github.com/user-attachments/assets/231112ae-0dc4-4e1d-81ed-8781776eccca)
```
df.skew()
```
![image](https://github.com/user-attachments/assets/27e98ee0-e654-4333-925b-92e7bef675f6)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/91049c9a-3806-4634-93a7-11daef621c37)
```
np.log(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/1cf75d4c-2461-4eef-8f9e-0089da2eb896)
```
np.reciprocal(df["Moderate Positive Skew"])
```
![image](https://github.com/user-attachments/assets/2b32d25f-a00e-4df7-944c-1af30f68d975)
```
np.sqrt(df["Highly Positive Skew"])
```
![image](https://github.com/user-attachments/assets/f4ad1eee-b9cf-4f23-9527-c9da68485df1)
```
df["Highly Positive Skew_boxcox"], parameters=stats.boxcox(df["Highly Positive Skew"])
df
```
![image](https://github.com/user-attachments/assets/dabd53b3-4636-4340-ab58-303442f98842)
```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```
![image](https://github.com/user-attachments/assets/fab1f7e9-c5bb-48be-9bcd-92929f207d7a)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/1ff7c957-2a07-46e4-8972-5ff3f489801e)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/11c81338-b956-4710-af79-55a16d2e6be9)
```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)

df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])

sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/67ec1de6-4f3a-4e37-9284-4254123711c8)
```
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/22951437-b056-46e4-9f0d-c68e441fd156)
```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df["Highly Negative Skew"],line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/e1d8569f-39c4-44c2-ac6a-7c4047761d2f)
```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```
![image](https://github.com/user-attachments/assets/b5951269-d9b1-4d16-bfbc-87c7a46acce1)




# RESULT:
       Thus the given data, Feature Encoding, Transformation process and save the data to a file was performed successfully.

       
