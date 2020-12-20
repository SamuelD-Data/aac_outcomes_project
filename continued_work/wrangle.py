# establishing environment
import sklearn

import pandas as pd
import numpy as np

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

def get_aac():
    """
    No argument needed. Function returns aac data as pandas DF.
    """
    # acquiring data and storing as DF
    df = pd.read_csv('aac_intakes_outcomes.csv')

    # returning DF
    return df

def prep_aac(df):
    """
    Accepts DF. Returns data fully prepped and split in train, validate, and test sets
    for exploration and modeling with changes outlined in notebook.
    """
    # only keeping selected columns
    df = df[['outcome_type', 'sex_upon_outcome',
       'age_upon_outcome_(days)','outcome_datetime', 'outcome_number',
        'animal_type', 'breed', 'intake_condition', 'intake_type', 'sex_upon_intake',
       'age_upon_intake_(days)', 'intake_datetime',
       'intake_number', 'time_in_shelter_days']]

    # dropping null values
    df.dropna(inplace = True)

    # changing "livestock" and "bird" values to "other" animal type
    df['animal_type'] = np.where(((df.animal_type == 'Livestock') | (df.animal_type == 'Bird')), 'Other', df.animal_type)

    # converting animal type values to lowercase 
    df.animal_type = df.animal_type.str.lower()

    # creating dummy columns for animaly types
    a_type = pd.get_dummies(df.animal_type, prefix = 'is')

    # adding dummy columns to main DF
    df = pd.concat([df, a_type], axis = 1)

    # adding boolean columns for female, male, and unknown sex
    df['is_male'] = np.where((df.sex_upon_outcome.str.contains('Male')), 1, 0)
    df['is_female'] = np.where((df.sex_upon_outcome.str.contains('Female')), 1, 0)
    df['sex_unknown'] = np.where((df.sex_upon_outcome.str.contains('Unknown')), 1, 0)

    # creating sex column with sex stored as categorical string
    # sex_upon_outcome currently stores both sex and neutered/spayed info as single value
    df['sex'] = np.where((df.sex_upon_outcome.str.contains('Male')), 'Male', 0)
    df['sex'] = np.where((df.sex_upon_outcome.str.contains('Female')), 'Female', df.sex)
    df['sex'] = np.where((df.sex_upon_outcome.str.contains('Unknown')), 'Unknown', df.sex)

    # creating boolean sterilized_outcome column to reflect if animal was neutered or spayed at outcome
    df['sterilized_outcome'] = np.where(
    (df.sex_upon_outcome.str.contains('Neutered')) |
    (df.sex_upon_outcome.str.contains('Spayed')), 1, 0)

    # creating boolean sterilized_income column to reflect if animal was neutered or spayed at intake
    df['sterilized_income'] = np.where(
    (df.sex_upon_intake.str.contains('Neutered')) |
    (df.sex_upon_intake.str.contains('Spayed')), 1, 0)

    # get index names for all rows where pet was return to owner
    index_names = df[(df['outcome_type'].str.contains('Return')) | (df['outcome_type'].str.contains('Rto'))].index 
    # drop returned to owner rows
    df.drop(index_names, inplace = True) 

    # creating boolean is_adopted column to reflect if animal was adopted or not
    df['is_adopted'] = np.where((df.outcome_type.str.contains('Adopt')), 1, 0)

    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to age_in_weeks column and adding scaled version of column to DF
    df['age_upon_outcome_(days)_s'] = scaler.fit_transform(df[['age_upon_outcome_(days)']])

        # fitting scaler to age_in_weeks column and adding scaled version of column to DF
    df['age_upon_intake_(days)_s'] = scaler.fit_transform(df[['age_upon_intake_(days)']])

    df['agg_breed'] = np.where((df.breed.str.contains('Pit Bull')), 1, 0)
    df['agg_breed'] = np.where((df.breed.str.contains('Rottweiler')), 1, df.agg_breed)
    df['agg_breed'] = np.where((df.breed.str.contains('German Shepherd')), 1, df.agg_breed)
    df['agg_breed'] = np.where((df.breed.str.contains('Doberman')), 1, df.agg_breed)

    # reordering columns so that target variable, "is_adopted", is last
    df = df[['animal_type', 'agg_breed', 'intake_datetime', 'intake_condition','intake_type', 
    'age_upon_intake_(days)', 'age_upon_intake_(days)_s', 'intake_number', 'outcome_datetime', 'age_upon_outcome_(days)', 
    'age_upon_outcome_(days)_s', 'outcome_number',  'time_in_shelter_days', 'is_cat','is_dog', 'is_other', 'is_male', 
    'is_female', 'sex_unknown', 'sex','sterilized_outcome', 'sterilized_income', 'is_adopted']]

    # splitting data
    train_validate, test = train_test_split(df, test_size=.2, random_state=123)
    train, validate = train_test_split(train_validate, test_size=.3, random_state=123)

    # returning DFs
    return train, validate, test

def wrangle_aac():
    """
    No argument needed. Acquires and returns aac data fully prepped for exploration and modeling with changes outlined in notebook.
    """
    # using get_aac function to acquire data
    df = get_aac()

    # returning DFs prepped with prep_aac function
    return prep_aac(df)







