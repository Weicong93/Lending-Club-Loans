# Lending Club Loans Default Prediction

Flask app for ML model prediction via (1) HTML Form (2) API requests.

## Model Used
XGBoost. Output: proba instead of fixed 1/0 prediction. 
1 = Will default
0 = Will not default
To separately decide what is the proba threshold to determine whether outcome is 1 or 0 

## Key Features For Model Prediction
	Feature	Importance
0	86630	0.205654
1	93700	0.197696
2	11650	0.178153
3	29597	0.091814
4	grade	0.090895
5	05113	0.084256
6	00813	0.077627
7	term	0.031843
8	int_rate	0.009855
9	ID_delta	0.005838
10	mort_acc	0.005682
11	dti	0.005633
12	annual_inc	0.003378
13	emp_length	0.002254
14	loan_amnt	0.002237
15	revol_util	0.001662
16	installment	0.001290
17	ECL_delta	0.001119
18	open_acc	0.001093
19	70466	0.000979
20	22690	0.000561
21	48052	0.000482
22	30723	0.000000

## Data Processing To Get Features for Prediction
1. Zipcodes - Direct input on web app
2. Grade - LC assigned. Direct input. Grade to be transformed according to label encoding used
3. Term - Number of payments on loan. 36 or 60. Direct input and transformed according to label encoding
4. int_rate - Direct input, transform using normalizer used in training model
5. ID_delta - Input issue date, which gets transformed according to the formula used in model to get delta, and then normalized
6. mort_acc


## Other Info


## Acknowledgements

Dataset Source: provided by Heicoder Academy, taken from Kaggle