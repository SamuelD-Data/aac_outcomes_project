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

    # adding column that contains age groups 
    df['age_group_years'] = np.where((df['age_upon_outcome_(years)'] <= 1), 'a.0-1', None)
    df['age_group_years'] = np.where(((df['age_upon_outcome_(years)'] > 1) & (df['age_upon_outcome_(years)'] < 4)), 'b.2-3', df.age_group_years)
    df['age_group_years'] = np.where(((df['age_upon_outcome_(years)'] >= 4) & (df['age_upon_outcome_(years)'] < 6)), 'c.4-5', df.age_group_years)
    df['age_group_years'] = np.where(((df['age_upon_outcome_(years)'] >= 6) & (df['age_upon_outcome_(years)'] < 8)), 'd.6-7', df.age_group_years)
    df['age_group_years'] = np.where(((df['age_upon_outcome_(years)'] >= 8) & (df['age_upon_outcome_(years)'] < 10)), 'e.8-9', df.age_group_years)
    df['age_group_years'] = np.where(((df['age_upon_outcome_(years)'] >= 10) & (df['age_upon_outcome_(years)'] < 12)), 'f.10-11', df.age_group_years)
    df['age_group_years'] = np.where(((df['age_upon_outcome_(years)'] >= 12) & (df['age_upon_outcome_(years)'] < 14)), 'g.12-13', df.age_group_years)
    df['age_group_years'] = np.where(((df['age_upon_outcome_(years)'] >= 14) & (df['age_upon_outcome_(years)'] < 16)), 'h.14-15', df.age_group_years)
    df['age_group_years'] = np.where((df['age_upon_outcome_(years)'] >= 16), 'i.16+', df.age_group_years)

    # only keeping selected columns
    df = df[['sex_upon_outcome','age_upon_outcome_(days)', 'animal_type', 'breed', 'outcome_subtype',
    'outcome_type','sex_upon_intake', 'age_group_years']]

    # filling outcome subtype nulls with Unknown
    df['outcome_subtype'] = np.where((df.outcome_subtype.isnull() == True), 'unknown', df.outcome_subtype)

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
    df['gender_unknown'] = np.where((df.sex_upon_outcome.str.contains('Unknown')), 1, 0)

    # creating gender column with gender stored as categorical string
    # sex_upon_outcome currently stores both sex and neutered/spayed info as single value
    df['gender'] = np.where((df.sex_upon_outcome.str.contains('Male')), 'Male', 0)
    df['gender'] = np.where((df.sex_upon_outcome.str.contains('Female')), 'Female', df.gender)
    df['gender'] = np.where((df.sex_upon_outcome.str.contains('Unknown')), 'Unknown', df.gender)

    # creating boolean sterilized_income column to reflect if animal was neutered or spayed at intake
    df['sterilized_income'] = np.where(
    (df.sex_upon_intake.str.contains('Neutered')) |
    (df.sex_upon_intake.str.contains('Spayed')), 1, 0)

    # get index names for all rows where pet was returned to owner, was deceased upon arrival or was not a cat or dog
    index_names = df[(df['outcome_type'].str.contains('Return')) | (df['outcome_type'].str.contains('Rto')) | 
    (df['outcome_type'].str.contains('Disposal')) | (df['animal_type'].str.contains('other'))].index 

    # drop rows based on index_names contents
    df.drop(index_names, inplace = True) 

    # creating boolean is_adopted column to reflect if animal was adopted or not
    df['is_adopted'] = np.where((df.outcome_type.str.contains('Adopt')), 1, 0)

    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to various columns and adding scaled versions of each to DF
    df['age_upon_outcome_(days)_s'] = scaler.fit_transform(df[['age_upon_outcome_(days)']])

    # adding agg_breed columns. represents if animal is of breed commonly perceived to be aggressive
    df['perceived_agg_breed'] = np.where((df.breed.str.contains('Pit Bull')), 1, 0)
    df['perceived_agg_breed'] = np.where((df.breed.str.contains('Rottweiler')), 1, df.perceived_agg_breed)
    df['perceived_agg_breed'] = np.where((df.breed.str.contains('Chow')), 1, df.perceived_agg_breed)
    df['perceived_agg_breed'] = np.where((df.breed.str.contains('Doberman')), 1, df.perceived_agg_breed)

    # making all column names lower case
    df.columns = df.columns.str.lower()

    # reordering columns
    df = df[['perceived_agg_breed','age_upon_outcome_(days)', 'age_upon_outcome_(days)_s','age_group_years', 
    'is_cat', 'is_dog', 'animal_type', 'is_male', 'is_female', 'gender_unknown', 'gender', 'sterilized_income',
       'is_adopted']]

    df.columns = ['perceived_agg_breed', 'age_outcome_days', 'age_outcome_days_s', 'age_group_years', 
    'is_cat', 'is_dog', 'species', 'is_male', 'is_female', 'gender_unknown', 'gender', 'sterilized_income',
    'is_adopted']

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







