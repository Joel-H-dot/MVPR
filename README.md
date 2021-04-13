# MVPR
Code for multi-variable polynomial regression 

This code fits a multi-variable polynmoial equation to multi-variable outputs. We first prepare the data as follows


wb_train = pd.read_excel (r'C:\Users\blah\training.xlsx')
wb_targets = pd.read_excel (r'C:\Users\blahD\targets.xlsx')

df_train = pd.DataFrame(wb_train)
df_train = df_train.to_numpy()

df_targets = pd.DataFrame(wb_targets)
df_targets = df_targets.to_numpy()

mean_dat = df_train[:,:].mean(axis=0)
std_dat = df_train[:,:].std(axis=0)

df_train -= mean_dat

if 0 not in std_dat :
    df_train[:,:] /= std_dat

num_validation = 200
training_data = df_train[:-num_validation,:]
training_targets = df_targets[:-num_validation,:]

validation_training = df_train[-num_validation:,:]
validation_targets = df_targets[-num_validation:,:]

With the standarrdised data we perform a polynomial expansion and place in the matrix X, where the rows are the observations and the columns the polynmoial terms. The code then uses TSVD 
to regularise the Moore-Penrose pseudo-inverse. The ideal truncation point is found using a golden-section search. The process is repeated for various polynmoial orders and cross-validated 
on a reserved dataset. An example may be as follows 

MVPR_model = MVPR_forward(training_data,training_targets,validation_training,validation_targets)

optimum_order = MVPR_model.find_order()
CM = MVPR_model.compute_CM(optimum_order)

predicted_targets = MVPR_model.compute(CM, optimum_order, df_train)
