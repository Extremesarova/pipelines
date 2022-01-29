Alpha version

## Dealing with missing values
~~Remark: for Kaggle competitions we can concatenate train and test datasets (if we have the test dataset) and use this dataset for values imputation to get higher score on the leaderboard. This way we will be leveraging the distribution of a test set for better imputation of missing values. But this can cause a data leakage (train-test contamination), so I am not sure about this technique.~~   

### Some rules
In real life we don't have the test data, so the general rule will be to create a correct cross-validation procedure and fit imputation methods only on train data and the transform dev and test sets. Another rule - there is no way to know beforehand which imputation method will be the best on the particular data, so check everything you can (using CV) and choose the best one.

**Rules**:
* Fit imputing methods (preprocessing methods, in general) only on train data (to avoid data leakage) and then apply them on val and test data. 
* Check different imputing methods and choose the best one using cross-validation.

### The source of the missing values 
You need to understand what are the possible sources of the missing values in your data before deciding on the approach for dealing with them.  
Some thoughts about this:  
* It could be the case that the fact that some data is missing is the feature itself
* Or the data could just be randomly missing

Next I'm going to go through the simple approach on how to deal with missing values

TO DO  
Open questions:
* How to approach the choice of missing values imputation algorithm? For example, can we choose the baseline classification algorithm, such as logistic regression, then train it with several imputation techniques and then select the best imputation algorithm, impute missing values, and try more advanced classifiers (like boosting methods) on the dataset with imputed values? Or we should somehow check all the combinations of imputation algorithms with classifiers? Or another approach?  


Reference notebooks:
* [Handling With Missing Data by Rob Mulla](https://www.kaggle.com/robikscube/handling-with-missing-data-youtube-stream/notebook)