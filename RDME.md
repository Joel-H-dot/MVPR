# Overview

MVPR is [available on PyPI][pypi], and can be installed via
```none
pip install MVPR
```
This package fits a multi-variable polynomial equation to a set of data using cross validation. The solution is regularised using truncated singular value decomposition, where the truncation point is found using a golden section search.



[pypi]:  https://pypi.org/project/MVPR/

# Example

consider a 3-D set of data, plotted as follows:

![image](https://user-images.githubusercontent.com/60707891/114895020-09b8d580-9e07-11eb-83af-83843a049463.png)



First import the data:
```
df= pd.read_excel(r'C:\Users\filepath\data.xlsx')
data=df.to_numpy()
df= pd.read_excel(r'C:\Users\filepath\targets.xlsx')
targets=df.to_numpy()
```
select the proportions of data for cross-validation
```
proportion_training = 0.9
num_train_samples = round(len(data[:,0])*0.8)
num_val_samples = round(len(data[:,0]))-num_train_samples
```
standardise:
```
mean_dat = data[:, :].mean(axis=0)
std_dat = data[:, :].std(axis=0)

data -= mean_dat

if 0 not in std_dat:
    data[:, :] /= std_dat

training_data = data[:num_train_samples, :]
training_targets = targets[:num_train_samples, :]

validation_data = data[-num_val_samples :, :]
validation_targets = targets[-num_val_samples :, :]
```
call the following
```
M = MVP.MVPR_forward(training_data, training_targets, validation_data, validation_targets)

optimum_order = M.find_order()
coefficient_matrix = M.compute_CM(optimum_order)

predicted_validation = M.compute(coefficient_matrix, optimum_order, validation_data)

df = pd.DataFrame(predicted_validation)
df.to_excel(r'C:\Users\EEE Admin\Desktop\test_MVPR\predicted.xlsx')
```
The fitted curve:

![image](https://user-images.githubusercontent.com/60707891/114898241-d592e400-9e09-11eb-9f88-28702ac9b0c8.png)



