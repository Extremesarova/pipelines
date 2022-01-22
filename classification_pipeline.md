- Predictive modelling
    - Specify random state for reproducibility
        
        Use it for splitting data and for initialization of models
        
    - Split data into train/test sets
        
        Describe here approaches for splitting data (with dependency on the number of examples)
        
    - Outline possible algorithms there are for the task
        - Classification:
            - Logistic Regression
            - SVM (Linear and Radial kernels)
            - Decision Trees
            - Random Forest
            - k-Nearest Neighbors
            - Naive Bayes
            - Neural Networks
            - Boostings:
                - AdaBoost
                - Gradient Boosting
                - XGBoost
                - LightGBM
                - CatBoost
        - Regression:
            - Linear Regression
            - L2, Ridge Regression
            - L1, Lasso Regression
            - Elastic-Net Regression (with combined L1 and L2 priors as regularizer)
            - Decision Tree Regression
            - Boosting Regressors:
                - AdaBoost
                - Gradient Boosting
                - XGBoost
                - LightGBM
                - CatBoost
    - Choose correct cross-validation technique
        - Leave-p-out CV
            
            Takes too much time on big datasets
            
        - k-Fold CV
            
            k = 3,5,7,10,20
            
        - Stratified k-Fold
            
            Preserves the ration between classes in splits
            
        - Time Series Split
            
            Used for time series
            
    - According to chosen CV technique train chosen models
        - Create DataFrame with training results using next columns:
            - Model's name
            - CV mean
            - CV std
        - Plot results after sorting by mean CV value
            
            Box plot is preferable for this task
            
        - Choose 2-4 best models for hyperparameter tuning and ensemble modeling
        - Check confusion matrix
            
            Interpret results
            
    - Hyperparameter tuning for the best models
        
        For hyperparameter tuning next approaches can be used:
        
        - Sklearn GridSearchCV
        - Sklearn RandomizedSearchCV
        - HyperOpt
        - Optuna
    - Plotting learning curves
        - x - number of training examples
        - y - score
        - lines:
            - training score
            - cv-score
    - Analyse feature importance of tree-based classifiers
        - Plot graphs with feature importances
            - Find common important features
            - Analyse results
            - Check correlation between predictors
                
                `g = sns.heatmap(ensemle_results.corr(), annot=True)`
                
        
        [**ELI5](https://github.com/TeamHG-Memex/eli5) to explain model predictions**
        
    - Analyse correlation
        
        If results are different, then we should consider an ensembling vote classifier at least
        
    - Ensemble modelling
        
        OOF predictions on the train dataset should be used as an input to meta-classifier
        
        Ensembling can be done in ways like:
        
        - Stacking
            - Voting
                - Soft
                Predicts class with the largest summed probability from the models
                - Hard
                Predicts the class with the largest sum of the votes from the models
                
                `sklearn.ensemble.VotingClassifier`: we can use 2-4 best models with CV
                
                `[mlxtend.classifier.EnsembleVoteClassifier`](http://rasbt.github.io/mlxtend/) 
                
            - Weighted average
            - Blending
            - Stacking
            - Super learner
        - Bagging
            
            Base estimators:
            
            - k-NN classifier
            - Decision Trees
        - Boosting
            - AdaBoost (Sklearn)
            - GradientBoosting (Sklearn)\
            - Gradient Boosting (h2o)
            - XGBoost
            - LightGBM
            - CatBoost