Contains information from the Austin Animal Center Outcomes database, which is made available
at the URL below under the Open Database License (ODbL).

https://data.austintexas.gov/Health-and-Community-Services/Austin-Animal-Center-Outcomes/9t4d-g238

The data contains intakes and outcomes of animals entering the Austin Animal Center from the beginning of October 2013 to the present day. The datasets are also freely available on the Socrata Open Data Access API and are updated daily.

https://dev.socrata.com

# Title

Predicting Animal Adoptions and Identifying the Drivers of Adoption

# Background

From: https://www.kaggle.com/aaronschlegel/austin-animal-center-shelter-intakes-and-outcomes

"The Austin Animal Center is the largest no-kill animal shelter in the United States that provides care and shelter to over 18,000 animals each year. As part of the AAC's efforts to help and care for animals in need, the organization makes available its accumulated data and statistics as part of the city of Austin's Open Data Initiative."

# Goals

- Identify drivers of animal adoption at the AAC

- Create a model that predicts animal adoption

I will also deliver the following:

- aac_project_notebook.ipynb

    - A Jupyter Notebook that provides a breakdown of the project at each phase of the data science pipeline

- README.md

    - A markdown file that includes a background summary of the project goals, data dictionary, reasons for selected columns, initial thoughts, project plan, steps for how to reproduce the project, and key findings / takeaways.

- wrangle.py

    - A python file that contains all functions needed to acquire and prep the data as shown in the aac_project_notebook file

- A presentation that summarizes the major findings of this project

Data Dictionary

Defining all columns that were used in exploration and beyond in addition to heating_system columns since they were a major part of preparation.

bathroom_cnt: Number of bathrooms in property (renamed to bathroom_count)

bathroom_count: Number of bathrooms in property

bedroom_cnt: Number of bedrooms in property

bedroom_count: Number of bedrooms in property (renamed to bedroom_count)

calculatedfinishedsquarefeet: Total living square feet within property (renamed to property_sq_ft)

property_sq_ft: Total living square feet within property

taxdvalueollarcount: Total tax value of property (renamed to tax_dollar_value)

tax_dollar_value: Total tax value of property

heatingorsystemtypeid: Code for type of heating system in property (encoded and split into heating_system_type_x)

heating_system_type_2: Central heating system in property

heating_system_type_20: Floor/Wall heating system in property

heating_system_type_7: Solar heating system in property

Reasons for Selected Columns

bathroom_count: Found this column was a near duplicate of two other columns (fullbathcnt and calculatedbathnbr). Only 46 rows differed between them so I didn't perceive any significant impact of their differences. Chose this one as the name sounded the closest to what I needed, a count of the bathrooms in the property.

property_sq_ft: Found this column was a near duplicate of finishedsquarefeet12. Only 48 rows differed between them so I didn't perceive any significant impact of their differences. Decided to use this column since the name sounded the closest to what I wanted, the square footage within the property.

tax_dollar_value: Represents the sum of landtaxvaluedollarcnt and structuretaxvaluedollarcnt. I felt the sum of the tax value from both of the originating values would be more effective in my exploration and modeling. If this feature was found to be ineffective I would have considered using its source values instead.

All other columns have unique values that were not represented directly or indirectly in other columns. Thus they were chosen as they were the only sources for their data.

Initial Thoughts

How will I handle exploring clusters?

Use visualizations to see cluster relationship with logerror, the perform hypothesis test to evaluate observation
How can I perform a hypothesis test on a cluster variable with 3 or cluster types?

Use ANOVA test since it allows for more than 2 variable means to be tested simeltaneously
How will I know how many clusters to make for each feature set?

Use subplots or elbow test to identify viable cluster amount
Initial Hypothesis

Log errors will push farther away from 0 in cases where the rarity of variables value increases.
For example, if only a few properties we've ever evaluated have more than 20 bathrooms, we're going to have trouble evaluating it's value accurately because we haven't encountered a lot of properties with that rare of a variable that relates to value.
Update: After exploring and modeling, this appears to be untrue, it seemed that as bathroom count and other variables decreased, log errors became more numerous and extreme, despite there being an abundance of properties with low bathroom counts.
Project Plan

Acquire
Use function with SQL query to acquire data from data science database
Prepare
Prepare data as needed for exploration including but not limited to
Addressing null values
Impute if
reasonable given turnaround timeframe
too much data will be lost by dropping
Drop otherwise
Addressing outliers
focus on extreme outliers (k=6) to preserve data
drop to conserve time
otherwise transform to upper/lower boundaries if too much data will be lost by dropping
Data types
make sure all columns have an appropriate datatype
Dropping columns
remove columns as needed and for reasons such as
being duplicate of another column
majority of column values are missing
only 1 unique value in data
Scale non-target variable numerical columns
Encode categorical columns via get_dummies
Use RFE on columns remaining after prep
For simplicity, only take top 3 columns to exploration
Can retrieve more if deemed necessary later
Explore
Plot each feature relation to logerror
Identify the relationship between them (example, as x increases, log_error increases past 0)
Perform hypothesis test to confirm or deny if this relationship statistically present
Create clusters using each unique pair of features
Use subplots with varying number of clusters per pair of features to identify a cluster amount that produces strong separation between clusters
Create clusters with amount prescribed by subplots
Perform hypothesis on each set of clusters to see if log error varies between them
Take all non-clustered and clustered variables that were statistically significant to modeling phase to use as features in model
Model
Create baseline that predicts mean of logerror and calculate RMSE against actual logerror values in train data
Create 3 alternate models with varying features
use 3 models on train set
top 2 models that outperform baseline will go to validate
Use top 2 models on validate set, model with best RMSE goes to test
Use best model on test set and evaluate results
Conclude
Document the following
identifed drivers of log error
best model's details
evaluate effectiveness of clusters as drivers and model features
recommendations
expectations
what I would like to do in the future with regard to this project
How to Reproduce

Download data from Kaggle into your working directory. (You must be logged in to your Kaggle account.)

Install acquire.py, prepare.py and model.py into your working directory.

Run the jupyter notebook.

Key Findings and Takeaways

Summary of Key Findings:

Through visualizations, hypothesis tests and modeling, we discovered evidence that drivers of log_error may include

bedroom_count
property_sq_ft
tax_dollar_value
clusters created from a combination of bedroom_count and property_sq_ft
We created several models including a baseline that always predicted logerror to be the sample average

Each model's performance was evaluated based on the RMSE value produced by comparing its prediction of logerror values vs. actual log error values from the data it was predicting with

Model 2 was the best performer (specs listed below)

Type: Linear Regression
Features: Uses all features listed above, except for clusters
Although this model did not use clusters as features and outperformed models that did, our clustering algorithm was very new and with time could be improved and incorporated into this model to possibly improve its effectiveness

It should be noted that our second best model used clusters on the validate (out of sample) data to outperform our baseline model which was using in-sample data. This is further evidence that clusters may still be useful as tool for predicting log errors.

Recommendation:

Begin a project to improve the accuracy of our zillow estimate software using the insights and model generated from this project

Expectations:

By improving the accuracy of our zestimates we will increase satisfaction among our current users and make our services more attractive to potential users.

In the future:

I'd like to revisit this project and explore / model with clusters more. A new combination of cluster features may generate clusters that prove to be very useful in predicting log error. I'd also like to try imputing some of the null values we dropped and observe how that influences our hypothesis tests and modeling.