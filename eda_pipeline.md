Alpha version

# Importing needed packages

```python 
import pandas as pd  
from matplotlib import pyplot as plt  
import seaborn as sns  
```

# Reading the data
    
```python
df = pd.read_csv(...)
```

## Printing head of the data 
```python
df.head()
```

## Printing information about the data 
```python
df.info()
# or
df.describe(include='all').T
```

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

Reference notebooks:
* [Handling With Missing Data by Rob Mulla](https://www.kaggle.com/robikscube/handling-with-missing-data-youtube-stream/notebook)

## Outliers detection
Outliers can have huge effect on the predictions (especially for regression):  
* Turkey method  
    This method defines the interquartile range comprised between 1stand 3rd quartiles of the distribution values (IQR). An outlier is a row that have a feature value outside the IQR ± an outlier step (1.5 * IQR)
* Another method

# Target variable analysis
- Plot several plots
- Determine metric

# Feature Analysis
- Looking at columns and determining feature types:
    
    categorical, discreet, ordinal (non-numerical), continuous(numerical)
    
- Summarizing data and showing some statistics:
    1. For numerical features `df.describe()`
    2. For non-numerical features `df.describe(include=["object","bool"])`
- Analysis
    
    The basic idea here is to analyze the dependency between the features and the target variable.
    Plotting correlation matrix between numerical variables and the target variable
    
    1. Categorical features (nominal variables)
    Cannot sort or give any order to such variables
        - Groupby dataset by this feature and show the target variable
        `df.groupby(["cat_1", "target"])["target"].count()` or `.mean()`
        - Bar plot
        `g = sns.barplot(x="cat_1", y="target", data=df, kind="bar")
        g = g.set_ylabel("...")`
        - Factor plot
        `g = sns.factorplot(x="cat_1", y="target", data=df)`
        - Count plot
        - `g = sns.countplot("cat_1", hue="target", data=df) orhue="other_feature"`
        - Description with the analysis of the plots and guess aboutpossible importance of the feature
    1. Ordinal features
    Can have relative ordering or sorting between the values
        - Crosstab
        `pr.crosstab(df["ord_1"], df["target"]), margins=True).stylebackground_gradient(cmap='summer_r')`
        - Bar plot
        `df["ord_1"].value_counts().plot.bar()`
        - Count plot
        `g = sns.countplot("ord_1", hue="target", data=df) orhue="other_feature"`
        - Factor plot
        `g = sns.factorplot(x="ord_1", y="target", data=df, kind="bar")`
        - Description with the analysis of the plots and guess aboutpossible importance of the feature
    1. Interaction of features
    cat_1 & ord_1 & other features
        - Crosstab
        `pd.crosstab(df["cat_1"], df["ord_1"], df["feature_3"],margins=True)`
        - Factor plot
        `sns.factorplot("cat_1", "target", hue="ord_1", data=df)
        sns.factorplot("cat_1", "target", hue="ord_1", data=df)
        sns.factorplot("ord_1", "target", hut="cat_1", col="cat_2",data=df)`
        - Description with the analysis of the plots and guess aboutpossible importance of the feature
    1. Continuous features
        - Calculate some statistics: min, max, mean
        - Interactions with categorical and ordinal features
            - Violin plot
            `sns.violinplot("ord_1", "cont_1", hue="target", data=df,split=True)
            sns.violinplot("cat_1", "cont_1", hue="target",  data=df,split=True)`
            - Facetgrid
            `g = sns.FacetGrid(df, col="target")
            g = g.map(sns.distplot, "cont_1")`
            - Hist
            `df[df["target"==0]]["cont_1"].plot.hist(bins=20,edgecolor="black", color="red")
            df[df["target"==1]]["cont_1"].plot.hist(bins=20,edgecolor="black", color="green")`
            - Factor plot
            `sns.factorplot("ord_1", "target", hut="cat_1",col="Initial", data=df)`
            `Initial` is a column from the Titanic Kaggle dataset
            - Kernel density estimation plot
            `g = sns.kdeplot(df[df["target"]==0]["cont_1"],color="red", shade=True)
            g = sns.kdeplot(df[df["target"]==1]["cont_1"],color="blue", shade=True)
            g = g.legend(["opt 0", "opt 1"])`
            - Distplot
            `sns.distplot(df[df["ord_1"]==1]["cont_2"])
            sns.distplot(df[df["ord_1"]==2]["cont_2"])
            sns.distplot(df[df["ord_1"]==3]["cont_2"])`
        - Description with the analysis of the plots and guess aboutpossible importance of the feature
    2. Discrete features ()
        - Crosstab with target
        `pd.crosstab(df["disc_1"],df["target"]).stylebackground_gradient(cmap='summer_r')`
        - Barplot
        `sns.barplot("disc_1", "target", data=df)`
        - Factor plot
        `sns.factorplot("disc_1", "target", data=df)`
        - Crosstab with other suitable features
        `pd.crosstab(df["disc_1"],df["ord_1"]).style.background_gradien(cmap='summer_r')`
        - Description with the analysis of the plots and guess aboutpossible importance of the feature
- Observations in a nutshell for all features
    
    Recap all the results of analysis
    
- Correlation between the features
    
    Analysing correlation between features so we can avoidmulticollinearity
    `sns.heatmap(df.corr(), annot=True, cmap="RdYlGn")`
    
# Feature Engineering
## Continuous features
### Binning/Discretization
    
Group a range of values into a single bin or assign them a single value.  
After binning, for example, we can do:
`sns.factorplot("cont_1_band", "target", data=df, col="ord_1"` tocheckthe dependency between new feature and old ones.  

`df["cont_1_band"].value_counts().to_frame()stylebackground_gradien(cmap='summer_r')` to check the numberofvalues in each band. 

Binning choices:  
* Manually
    
    ```python
    df['cont_1_band'] = 0
    df.loc[df['cont_1'] <= 16,'cont_1_band'] = 0
    df.loc[(df['cont_1'] > 16) & (df['cont_1'] <= 32)'cont_1_band']= 1
    df.loc[(df['cont_1'] > 32)&(df['cont_1'] <= 48)'cont_1_band'] =2
    df.loc[(df['cont_1'] > 48) & (df['cont_1'] <= 64)'cont_1_band']= 3
    df.loc[df['cont_1'] > 64,'cont_1_band'] = 4
    ```
    
* Using pandas.qcut
    
    `df["cont_2_range"] = pd.qcut(df["cont_2"], 4)`
    
    After that we should manually create new column using the unique ranges which are located in a column "cont_2_range".  
    For that, let's look at created ranges  
    `df.groupby['cont_2_range'['target'].mean().to_frame().stylebackground_gradie(cmap='summer_r')`
    
    Based on these ranges repeat manual step to create variable `cont_2_band`.
    
### Scaling & Standardization

### Normalization

### Log Transformation
Add here information from the part 1 EDA video with Heads and Tails

### Polynomial Featurizer
    
    [https://scikit-learn.org/stable/modules/generatedsklearnpreprocessing.PolynomialFeatures.html](https://scikit-learnorgstable/modules/generated/sklearn.preprocessingPolynomialFeatureshtml)
    
    [https://mlcourse.ai/articles/topic4-part3-regularization/(https:/mlcourse.ai/articles/topic4-part3-regularization/)
    
- Something else
- Encoding categorical features
- Some other features can be combined together
    - String values should be converted to numeric
        
        Before this we can analyse string features - whether we canmake from them new features. Like `Name` → `Initials` featurefrom Titanic Kaggle Competition.
        
        - Replace string values with integers (1 col → 1 col)
        `data['string_val'].replace(['option_1','option_2'],[0,1]inplace=True)`
        - `pd.get_dummies()` (1 col → n cols)
- Dropping unneeded features