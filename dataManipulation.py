#list column names and their data types
loanData.dtypes

#remove commas and change to Float
google_state["cost"] = google_state["Cost"].str.replace(',','').astype('float')

#%%return unique values
iris_data.unique()

#%%updateData
iris_data.loc[iris_data['class'] == 'versicolor', 'class'] = 'Iris-versicolor'

#%%where condition
borrEmail.loc[borrEmail['Borremail'] =="mmnfreemason05@verizon.net"]

#so that pandas knows to treat rows with 'NA' as missing values.
iris_data = pd.read_csv('iris.csv', na_values=['NA'])

#change data type
leadsData_clean["Income"] = leadsData_clean["Income"].astype(float)

#summary 
iris_data.describe()

#drop rows by condition
iris_data = iris_data.loc[(iris_data['class'] != 'Iris-setosa') | (iris_data['sepal_width_cm'] >= 2.5)]

#not start with, contains
google_state.loc[ny.logical_not(google_state["Location"].str.startswith("Total",na=False)),]
google_state.loc[ny.logical_not(google_state["Location"].str.contains("Total",na=False)),]

#hist graph
iris_data.loc[iris_data['class'] == 'Iris-setosa', 'sepal_width_cm'].hist()

#new way to assign value
iris_data.loc[(iris_data['class'] == 'Iris-versicolor') &
              (iris_data['sepal_length_cm'] < 1.0),
              'sepal_length_cm'] *= 100.0

#to and from csv
iris_data.to_csv('iris-data-clean.csv', index=False)
iris_data_clean = pd.read_csv('iris-data-clean.csv')               

#assert 
#We can quickly test our data using assert statements: 
#We assert that something must be true, 
#if it is, then nothing happens and the notebook continues running. 
#if our assertion is wrong, then the notebook stops running and brings it to our attention. For example:
assert 1 == 2



#As it stands, sklearn decision trees do not handle categorical data - see issue #5442. 
#This approach of using Label Encoding converts to integers which the DecisionTreeClassifier() will treat as numeric. 
#If your categorical data is not ordinal, this is not good - you'll end up with splits that do not make sense. 
#Using a OneHotEncoder is the only current valid way, but is computationally expensive
###If you're using other languages (R/SAS), then categorical data is not a problem. 

#distribution chart
import seaborn as sb
sb.distplot(model_accuracies)


df['A'].apply(str)
df.applymap(str)
