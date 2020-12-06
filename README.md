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

- Create a model that predicts animal adoptions at the AAC

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

age_in_weeks: animal's age at time of outcome expressed as weeks

age_in_weeks_s: animal's age at time of outcome expressed as weeks (scaled from 0 to 1)

age_upon_outcome: animal's age at time of outcome (adoption, transfer, etc.)

animal_type: the species of the animal (cat, dog, unknown)

is_adopted: boolean column representing if an animal was adopted (1 = True, 0 = False)

is_cat: boolean column representing if an animal is a cat (1 = True, 0 = False)	

is_dog: boolean column representing if an animal is a dog (1 = True, 0 = False)	

is_female: boolean column representing if an animal is female (1 = True, 0 = False)

is_male: boolean column representing if an animal is male (1 = True, 0 = False)	

is_neutered_or_spaded: sterilization status of the animal (1 = Animal was sterilized at time of outcome, 0 = Animal was not sterilized or sterilization status was unknown at time of outcome)

is_other: : boolean column representing if an animal was not identified as a dog or cat (1 = True, 0 = False)

sex: sex of the animal (male, female, unknown)

sex_unknown: boolean column representing if the sex of an animal was unknown (1 = True, 0 = False)

sex_upon_outcome: The sex of the animal and its sterilization status at time of outcome

# Reasons for Selected Columns

With the exception of age_in_weeks and age_in_weeks_s, all selected columns possesed information that no other columns held.

Age_in_weeks was kept so that the plot relating to age would reflect normal values. Age_in_weeks_s, the scaled age column, was created for use in modeling.

Boolean columns (is_dog, is_cat, etc) were created for us in modeling. Their source columns (animal_type, etc.) were retained to simplify the process of creating crosstabs. The exception being, is_adopted which replaced outcome_type as it was viable for both crosstabs and modeling.

# Project Plan

- Acquire
    - Download data from online source in csv format as local files
    - Use python function to acquire data from local files

- Prepare
    - Prepare data as needed for exploration including but not limited to
        - Addressing null values
            - Impute if reasonable given turnaround timeframe or too much data will be lost by dropping
            - Drop otherwise
    - Data types make sure all columns have an appropriate data type
    - Drop columns as needed for reasons including
        - Being duplicate of another column
        - Majority of column values are missing
        - Only 1 unique value in data
    - Scale non-target variable numerical columns
    - Encode categorical columns

- Explore
    - Plot each non-target variable's relation to target varialbe, adoption
    - Perform hypothesis test to confirm or deny if this relationship statistically present
    - Identify all variables that were statistically significant for use as features in model

- Model
    - Create baseline that predicts adopted or not adopted (whichever is more common) in 100% of cases 
    - Create alternate models that will fit to and predict train set
    - Top 2 models that outperform baseline will be used on validate set
    - Best model in validate phase will be used on test set
    - Use best model on test set and evaluate results

- Conclusion
    - Summarize the following
        - Acquisition of data
        - Preparation of data
        - Findings from exploration
        - Drivers of adoption
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
    
- Explore
    - Crosstabs and bar plots showed lower rates of adoption for animals that were not identified as cats or dogs, those of an unidentified sex, and those that had not yet been sterilized
    - Chi-squared tests showed that animal type, sex, and sterilization status are not independent of whether an animal is adopted
    - T-test gave evidence that adopted animals were younger on average than those than had not been adopted

    - In summary, the drivers of adoption appear to be
        - animal species
        - sterilization status (neutered or spaded)
        - age
        - sex
    
- Model
    - Created baseline model that produced 57% accuracy on train data
    - Created 4 alternate models using various algorithms
    - Best Model was created with the following profile
        - Type: Random Forest
        - Features: 
            - age_in_weeks_s
            - is_cat, is_dog, is_other
            - is_male, is_female, is_unknown
            - is_neutered_or_spayed
    - Best model maintained 76% accuracy on all datasets

- Recommendations
    - When feasible, spay or neuter animals to increase their likelihood of adoption
    - Develop a program that aims to pair older animals with suitable homes 

- Predictions
    - By following the recommendations above, the AAC may be able to increase their adoption rates via finding homes for animals who were at high risk of not being adopted

- Plans for the future
    - I'd like to focus on exploring the connections between various features
    - I'd also like to incorporate more features, such as color
    - I'll also being incorporating data about each animal's induction into the shelter to gain further insights