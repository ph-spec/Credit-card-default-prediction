![image](https://github.com/user-attachments/assets/d94a8a4b-0e7e-401c-aff0-cf93aa2cd04e)



##Format of dataset 

Contains Data with 30,000 Instances on the following 23 Features as shown in the table below: 
Variable Name	Type	Description	Missing Value

ID	Integer	ID	No

X1	Integer	LIMIT_BAL	No

X2	Integer	SEX	No

X3	Integer	EDUCATION	No

X4	Integer	MARRIAGE	No

X5	Integer	AGE	No

X6	Integer	PAY_0	No

X7	Integer	PAY_2	No

X8	Integer	PAY_3	No

X9	Integer	PAY_4	No

X10	Integer	PAY_5	No

X11	Integer	PAY_6	No

X12	Integer	BILL_AMT1	No

X13	Integer	BILL_AMT2	No

X14	Integer	BILL_AMT3	No

X15	Integer	BILL_AMT4	No

X16	Integer	BILL_AMT5	No

X17	Integer	BILL_AMT6	No

X18	Integer	PAY_AMT1	No

X19	Integer	PAY_AMT2	No

X20	Integer	PAY_AMT3	No

X21	Integer	PAY_AMT4	No

X22	Integer	PAY_AMT5	No

X23	Integer	PAY_AMT6	No

Y	Binary	default payment next month	No

The default payment which is a binary variable takes (Yes =1, No = 0), as the response variable.

The following 23 variables used in the research as explanatory variables:

X1: Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary)credit.

X2: Gender (1 = male; 2 = female).

X3: Education (1 = graduate school; 2 = university; 3 = high school; 4 = others). 

X4: Marital status (1 = married; 2 = single; 3 = others). 

X5: Age (year). X6 - X11: History of past payment. We tracked the past monthly payment records (from April to September 2005) as follows: 

X6 = the repayment status in September 2005. 

X7 = the repayment status in August 2005; . . . X11 = the repayment status in April 2005. The measurement scale for the repayment status is: -1 = pay duly; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above. 

X12-X17: Amount of bill statement (NT dollar). X12 = amount of bill statement in September 2005. 

X13 = amount of bill statement in August 2005; . . .; X17 = amount of bill statement in April 2005. 

X18-X23: Amount of previous payment (NT dollar). X18 = amount paid in September 2005; X19 = amount paid in August 2005; . . . X23 = amount paid in April 2005.

##Algorithms used 

1. Logistic Regression
2. Random Forest
3. Support Vector Machine(SVM) + LASSO(least absolute shrinkage and selection)
4. Gradient Boosting
5. Ensemble model 1 - Logistic regression, Random forest, Gradient boost
6. Ensemble model 2- Logistic Regression, SVM(support vector machine) + LASSO(least absolute shrinkage and selection) ,Gradient Boostin
7. Ensemble 3 -Random Forest, gradient boosting

 ##project files description
   
   This project includes 1 colab notebook 4 jupyter notebook
   
   Executable files
   
   credit card default prediction - includes all exploratory data analysisand all the algorithms used in this project
   
   output:
   
   google colab:all output are visible in the provided coab notebook
   
   jupyter: all outputs are visible in the provided jupyter note book

##Conclusion

Among the 6 models, XGBoost performed excellently with  (train accuracy of 1 and test accuracy of 0.995)
•	XGBoost model exhibits top-tier performance across all criteria, both as an individual and in ensembles. Also, the ensemble approaches utilising LR, RF, SVM + LASSO, and XGBoost, especially when it comes to AUC performed excellently.
   
•	SVM + LASSO model consistently exhibits good performance, especially in AUC, which is an important metric for class distinction.
 
