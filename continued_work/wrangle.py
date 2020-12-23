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
    df['age_group_year'] = np.where((df['age_upon_outcome_(years)'] <= 1), 'a.0-1', None)
    df['age_group_year'] = np.where(((df['age_upon_outcome_(years)'] > 1) & (df['age_upon_outcome_(years)'] < 4)), 'b.2-3', df.age_group_year)
    df['age_group_year'] = np.where(((df['age_upon_outcome_(years)'] >= 4) & (df['age_upon_outcome_(years)'] < 6)), 'c.4-5', df.age_group_year)
    df['age_group_year'] = np.where(((df['age_upon_outcome_(years)'] >= 6) & (df['age_upon_outcome_(years)'] < 8)), 'd.6-7', df.age_group_year)
    df['age_group_year'] = np.where(((df['age_upon_outcome_(years)'] >= 8) & (df['age_upon_outcome_(years)'] < 10)), 'e.8-9', df.age_group_year)
    df['age_group_year'] = np.where(((df['age_upon_outcome_(years)'] >= 10) & (df['age_upon_outcome_(years)'] < 12)), 'f.10-11', df.age_group_year)
    df['age_group_year'] = np.where(((df['age_upon_outcome_(years)'] >= 12) & (df['age_upon_outcome_(years)'] < 14)), 'g.12-13', df.age_group_year)
    df['age_group_year'] = np.where(((df['age_upon_outcome_(years)'] >= 14) & (df['age_upon_outcome_(years)'] < 16)), 'h.14-15', df.age_group_year)
    df['age_group_year'] = np.where((df['age_upon_outcome_(years)'] >= 16), 'i.16+', df.age_group_year)

    # only keeping selected columns
    df = df[['outcome_type', 'sex_upon_outcome',
       'age_upon_outcome_(days)','outcome_datetime', 'outcome_number',
        'animal_type', 'breed', 'intake_condition', 'intake_type', 'sex_upon_intake',
       'age_upon_intake_(days)', 'intake_datetime',
       'intake_number', 'time_in_shelter_days','age_group_year']]

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

    # get index names for all rows where pet was returned to owner, was deceased upon arrival or was not a cat or dog
    index_names = df[(df['outcome_type'].str.contains('Return')) | (df['outcome_type'].str.contains('Rto')) | 
    (df['outcome_type'].str.contains('Disposal')) | (df['animal_type'].str.contains('other'))].index 

    # drop returned to owner rows
    df.drop(index_names, inplace = True) 

    # creating boolean is_adopted column to reflect if animal was adopted or not
    df['is_adopted'] = np.where((df.outcome_type.str.contains('Adopt')), 1, 0)

    # creating scaler object
    scaler = sklearn.preprocessing.MinMaxScaler()

    # fitting scaler to various columns and adding scaled versions of each to DF
    df['age_upon_outcome_(days)_s'] = scaler.fit_transform(df[['age_upon_outcome_(days)']])
    df['age_upon_intake_(days)_s'] = scaler.fit_transform(df[['age_upon_intake_(days)']])
    df['intake_number_s'] = scaler.fit_transform(df[['intake_number']])
    df['outcome_number_s'] = scaler.fit_transform(df[['outcome_number']])
    df['time_in_shelter_days_s'] = scaler.fit_transform(df[['time_in_shelter_days']])

    # adding agg_breed columns. represents if animal is of breed commonly perceived to be aggressive
    df['agg_breed'] = np.where((df.breed.str.contains('Pit Bull')), 1, 0)
    df['agg_breed'] = np.where((df.breed.str.contains('Rottweiler')), 1, df.agg_breed)
    df['agg_breed'] = np.where((df.breed.str.contains('German Shepherd')), 1, df.agg_breed)
    df['agg_breed'] = np.where((df.breed.str.contains('Doberman')), 1, df.agg_breed)

    # creating dummy columns for intake_condition and intake_type
    dummy_df = pd.get_dummies(data = df, columns=['intake_condition', 'intake_type'])

    # creating df that holds source columns from dummies
    type_con = df[['intake_type', 'intake_condition']]

    # adding dummy columns sources back to df
    df = pd.concat([type_con, dummy_df], axis=1)

    # making all column names lower case
    df.columns = df.columns.str.lower()

    # reordering columns
    df = df[['agg_breed', 'intake_datetime',
       'age_upon_intake_(days)', 'age_upon_intake_(days)_s', 'intake_number',
       'intake_number_s', 'outcome_datetime', 'age_upon_outcome_(days)',
       'age_upon_outcome_(days)_s', 'outcome_number', 'outcome_number_s',
       'time_in_shelter_days', 'time_in_shelter_days_s', 'is_cat', 'is_dog',
       'is_other', 'animal_type', 'is_male', 'is_female', 'sex_unknown', 'sex',
       'sterilized_outcome', 'sterilized_income',
       'intake_condition_aged', 'intake_condition_feral',
       'intake_condition_injured', 'intake_condition_normal',
       'intake_condition_nursing', 'intake_condition_other',
       'intake_condition_pregnant', 'intake_condition_sick', 'intake_condition',
       'intake_type_euthanasia request', 'intake_type_owner surrender',
       'intake_type_public assist', 'intake_type_stray', 'intake_type', 'age_group_year','is_adopted']]

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







