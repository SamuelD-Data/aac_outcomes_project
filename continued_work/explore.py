# establishing environment
import sklearn
import pandas as pd 

from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


def rfe_ranker_lr(df):
    """
    Accepts dataframe. Uses Recursive Feature Elimination to rank the given df's features in order of their usefulness in
    predicting logerror with a logistic regression model.
    """
    # creating logistic regression object
    lr = LogisticRegression()

    # fitting logistic regression model to features 
    lr.fit(df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']], df['is_adopted'])

    # creating recursive feature elimination object and specifying to only rank 1 feature as best
    rfe = RFE(lr, 1)

    # using rfe object to transform features 
    x_rfe = rfe.fit_transform(df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']], df['is_adopted'])

    # creating mask of selected feature
    feature_mask = rfe.support_

    # creating train df for rfe object 
    rfe_df = df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']]

    # creating list of the top features per rfe
    rfe_features = rfe_df.loc[:,feature_mask].columns.tolist()

    # creating ranked list 
    feature_ranks = rfe.ranking_

    # creating list of feature names
    feature_names = rfe_df.columns.tolist()

    # create df that contains all features and their ranks
    rfe_ranks_df = pd.DataFrame({'Feature': feature_names, 'Rank': feature_ranks})

    # return df sorted by rank
    return rfe_ranks_df.sort_values('Rank')

def rfe_ranker_rf(df):
    """
    Accepts dataframe. Uses Recursive Feature Elimination to rank the given df's features in order of their usefulness in
    predicting logerror with a random forest model.
    """
    # creating logistic regression object
    rf = RandomForestClassifier(max_depth = 3, random_state=123)

    # fitting logistic regression model to features 
    rf.fit(df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']], df['is_adopted'])

    # creating recursive feature elimination object and specifying to only rank 1 feature as best
    rfe = RFE(rf, 1)

    # using rfe object to transform features 
    x_rfe = rfe.fit_transform(df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']], df['is_adopted'])

    # creating mask of selected feature
    feature_mask = rfe.support_

    # creating train df for rfe object 
    rfe_df = df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']]

    # creating list of the top features per rfe
    rfe_features = rfe_df.loc[:,feature_mask].columns.tolist()

    # creating ranked list 
    feature_ranks = rfe.ranking_

    # creating list of feature names
    feature_names = rfe_df.columns.tolist()

    # create df that contains all features and their ranks
    rfe_ranks_df = pd.DataFrame({'Feature': feature_names, 'Rank': feature_ranks})

    # return df sorted by rank
    return rfe_ranks_df.sort_values('Rank')

def rfe_ranker_dtc(df):
    """
    Accepts dataframe. Uses Recursive Feature Elimination to rank the given df's features in order of their usefulness in
    predicting logerror with a decision tree model.
    """
    # creating logistic regression object
    dtc = DecisionTreeClassifier(max_depth = 3, random_state=123)

    # fitting logistic regression model to features 
    dtc.fit(df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']], df['is_adopted'])

    # creating recursive feature elimination object and specifying to only rank 1 feature as best
    rfe = RFE(dtc, 1)

    # using rfe object to transform features 
    x_rfe = rfe.fit_transform(df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']], df['is_adopted'])

    # creating mask of selected feature
    feature_mask = rfe.support_

    # creating train df for rfe object 
    rfe_df = df[['perceived_agg_breed', 'is_cat', 'is_dog', 'is_male','is_female', 'gender_unknown', 'sterilized_income',
    'is_euthanasia_request', 'is_owner_surrender','is_public_assist', 'is_stray', 'is_wildlife',
    'is_aged', 'is_feral', 'is_injured', 'is_normal', 'is_nursing','is_other', 'is_pregnant', 'is_sick', 
    'age_outcome_days_s']]

    # creating list of the top features per rfe
    rfe_features = rfe_df.loc[:,feature_mask].columns.tolist()

    # creating ranked list 
    feature_ranks = rfe.ranking_

    # creating list of feature names
    feature_names = rfe_df.columns.tolist()

    # create df that contains all features and their ranks
    rfe_ranks_df = pd.DataFrame({'Feature': feature_names, 'Rank': feature_ranks})

    # return df sorted by rank
    return rfe_ranks_df.sort_values('Rank')