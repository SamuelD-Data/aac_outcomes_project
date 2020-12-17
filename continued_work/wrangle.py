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
    df = df[['age_upon_outcome', 'animal_type', 'sex_upon_outcome', 'outcome_type']]

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

    # creating boolean is_neutered_or_spayed column to reflect if animal was neutered or spayed
    df['is_neutered_or_spayed'] = np.where(
    (df.sex_upon_outcome.str.contains('Neutered')) |
    (df.sex_upon_outcome.str.contains('Spayed')), 1, 0)

    # creating boolean is_adopted column to reflect if animal was adopted or not
    df['is_adopted'] = np.where((df.outcome_type.str.contains('Adopt')), 1, 0)

    # creating age_split DF that contains numerical age and age unit of measure as separate columns
    age_split = df.age_upon_outcome.str.split(expand=True)

    # renaming columns
    age_split.columns = (['num', 'period'])

    # converting num column to numeric value
    age_split.num = pd.to_numeric(age_split.num)

    # creating age_in_weeks column that holds animal's age measured in weeks
    # animals less than one week old are rounded to 1 week old
    age_split['age_in_weeks'] = np.where((age_split.period.str.contains('day')), 1, 0)
    age_split['age_in_weeks'] = np.where((age_split.period.str.contains('week')), age_split.num, age_split.age_in_weeks)
    age_split['age_in_weeks'] = np.where((age_split.period.str.contains('month')), age_split.num * 4, age_split.age_in_weeks)
    age_split['age_in_weeks'] = np.where((age_split.period.str.contains('year')), age_split.num * 52, age_split.age_in_weeks)

    # adding age_in_weeks column to main DF
    df = pd.concat([df, age_split['age_in_weeks']], axis = 1)

    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to age_in_weeks column and adding scaled version of column to DF
    df['age_in_weeks_s'] = scaler.fit_transform(df[['age_in_weeks']])

    # reordering columns so that target variable, "is_adopted", is last
    df = df[['animal_type', 'is_cat', 'is_dog', 'is_other', 'sex', 'is_male', 
    'is_female', 'sex_unknown', 'is_neutered_or_spayed', 'age_in_weeks', 'age_in_weeks_s', 'is_adopted']]

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







