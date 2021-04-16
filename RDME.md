# Overview

MVPR is [available on PyPI][pypi], and can be installed via
```none
pip install MVPR
```
This package fits a multi-variable polynomial equation to a set of data using cross validation. The solution is regularised using truncated singular value decomposition of the Moore-Penrose pseudo-inverse, where the truncation point is found using a golden section search.


[pypi]:  https://pypi.org/project/MVPR/

# Example

consider a 3-D set of data, plotted as follows:

![image](https://user-images.githubusercontent.com/60707891/115008840-87322380-9ea3-11eb-85b3-778c06a3db9b.png)

and another set:

![image](https://user-images.githubusercontent.com/60707891/115008872-91ecb880-9ea3-11eb-9ef9-e0dc9d2537b6.png)

We want to find some mapping function for the same input data. Using the MVPR code we can place the vectors

![image](https://user-images.githubusercontent.com/60707891/115009673-70d89780-9ea4-11eb-97f3-a02e29d4fb30.png)




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
df.to_excel(r'C:\Users\filepath\predicted.xlsx')
```
The fitted curves:

![image](https://user-images.githubusercontent.com/60707891/115009854-a5e4ea00-9ea4-11eb-8774-6c87cf89c7b5.png)

![image](https://user-images.githubusercontent.com/60707891/115009871-abdacb00-9ea4-11eb-9d12-b76d45b67835.png)

# Functions and arguments
```
MVPR.find_order()
```
This function finds the optimal order of polynomial in the range 0 to 6, using cross validation. 
```
MVPR.find_order()
```
This function finds the optimal order of polynomial in the range 0 to 6, using cross validation. 
```
MVPR.compute_CM(order)
```
This function computes the coefficient matrix which fits a polynomial to the measured data in a least squares sense. The fit is regularised using truncated singular value decomposition, which eliminates singular values under a certain threshold. Any oder can be passed into this by the user, it does not have to have the range limited inf find_oder(). 

# Theory 

 For the theory behind the code see [[1]](#1).

## References
<a id="1">[1]</a> 
Hansen, P. C.  (1997). 
Rank-deficient and Discrete Ill-posed Problems: Numerical Aspects of Linear Inversion. 



