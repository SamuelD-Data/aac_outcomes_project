The files in this repository contain information from the Austin Animal Center Outcomes database, which is made available
at the URL below under the Open Database License (ODbL).

https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238

The data contains intakes and outcomes of animals entering the Austin Animal Center from the beginning of October 2013 to the present day. The datasets are also freely available on the Socrata Open Data Access API and are updated daily.

https://dev.socrata.com

This project and repository were not made for commercial use.

# Project Name

Predicting Animal Adoptions at the AAC (Austin Animal Center)

# Background

From: https://www.kaggle.com/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes

"The Austin Animal Center is the largest no-kill animal shelter in the United States that provides care and shelter to over 18,000 animals each year. As part of the AAC's efforts to help and care for animals in need, the organization makes available its accumulated data and statistics as part of the city of Austin's Open Data Initiative."

# Goals

- Create a model that predicts cat and dog adoptions at the AAC

I will also deliver the following:

- aac_project_notebook.ipynb

    - A Jupyter Notebook that provides a breakdown of the project at each phase of the data science pipeline

- README.md

    - A markdown file that includes a background summary of the project goals, data dictionary, reasons for selected columns, project plan, steps for how to reproduce the project, and key findings / takeaways.

- wrangle.py

    - A python file that contains all functions needed to acquire and prep the data as shown in the aac_project_notebook file

- A presentation that summarizes the major findings of this project

# Data Dictionary 

Defining all columns that were used in exploration and modeling.

age_group_years: animal's age group in terms of years (>1, 2-3, 4-5, etc.)

age_outcome_days: animal's age at time of outcome in terms of days

age_outcome_days_s: animal's age at time of outcome in terms of days, scaled from 0 to 1

species: the species of the animal (cat, dog, unknown)

is_adopted: boolean column representing if an animal was adopted (1 = True, 0 = False)

is_cat: boolean column representing if an animal is a cat (1 = True, 0 = False)	

is_dog: boolean column representing if an animal is a dog (1 = True, 0 = False)	

is_female: boolean column representing if an animal is female (1 = True, 0 = False)

is_male: boolean column representing if an animal is male (1 = True, 0 = False)	

perceived_agg_breed: boolean column that represents if an animal's breed is commonly perceived as aggressive (ie. chow, doberman, rottweiller, pit bull) (1 = True, 0 = False)

gender: gender of the animal (male, female, unknown)

gender_unknown: boolean column representing if the sex of an animal was unknown (1 = True, 0 = False)

sterilized_income: boolean column representing if animal was sterilized (neutered or spayed) prior to entering the AAC

# Project Plan

- Acquire
    - Download data from online source in csv format as local files
    - Use python function to acquire data from local files

- Prepare
    - Prepare data as needed for exploration including but not limited to
        - Addressing null values
            - Impute if reasonable given turnaround timeframe or too much data will be lost by dropping
            - Drop otherwise
    - Make sure all columns have an appropriate data type
    - Drop columns as needed for reasons including but not limited to
        - Being duplicate of another column
        - Majority of column values are missing
        - Only 1 unique value in data
    - Scale non-target variable numerical columns
    - Encode categorical columns

- Explore
    - Create plots to explore relationship between variables
    - Perform hypothesis tests to see if relationships between variables are statistically significant

- Model
    - Create baseline that predicts adopted or not adopted (whichever is more common) in 100% of cases 
    - Create alternate models that will fit to and predict train set
    - Top 2 models that outperform baseline on train sample will be used on validate sample
    - Best model in validate phase will be used on test set
    - Use best model on test set and evaluate results

- Conclusion
    - Summarize the following
        - Acquisition of data
        - Preparation of data
        - Findings from exploration
        - Best model's profile and results
        - Recommendations
        - Expectations
        - What I would like to add to this project in the future

# How to Reproduce

Download data into your working directory. (Links below)

https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238 (Routinely updated with new data)

https://www.kaggle.com/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes (Not routinely updated, contains only data from this project as of December 5th, 2020)

Install wrangle.py into your working directory.

Run the jupyter notebook.

# Key Findings and Takeaways
    
### Explore
- Some visualizations showed slight to stark differences between cats and dogs with varying characteristics whereas others did not 

- Two-sample, one-tailed t-test suggested that the average age of adopted animals is lower than the average age of animals that are not adopted

- Chi square tests showed that is_adopted and all of the following variables are not independent of each other
    - animal_type (cat or dog)
    - gender (male, female, unknown)
    - agg_breed (if a breed of dog is commonly perceived as aggressive)
    - sterilized_income (if an animal was sterilized prior to entering the AAC)
    - intake_condition
    - intake_type
    
### Model
- Created baseline model that produced 56% accuracy on train data
- Created 4 alternate models using various algorithms
- Best Model was created with the following profile
    - Type: Random Forest
    - Features: 
        - age_outcome_days_s (animal's age)
        - is_dog (represents if animal is a dog)
        - gender_unknown (if animal's gender is unknown)
        - sterilized_income (if animal was sterile at time of intake into AAC)
        - perceived_agg_breed (if animal's breed is commonly perceived as aggressive)
- Peformed with 71% accuracy on train (in-sample) data
- Peformed with 70% accuracy on validatea and test (out-of-sample) data

### Recommendations
- Develop a program that aims to pair older animals with suitable homes
- Sterilize animals prior to adoption
- Use website and promotional material to advocate for increased cat adoption

### Predictions
- By following the recommendations above, the AAC may be able to increase their adoption rates

### Plans for the future
- I'd like to explore the features that I was not able to explore in this iteration of the project in the interest of time.