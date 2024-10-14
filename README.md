# Columbia_Engineering_Data_Analytics_Project_4

Below Readme is just a template of the project IV

<h2>Alphabet Soup Charity</h2>

<h3>Overview</h3>

<p> 
This dataset provides valuable insights into customer loans, capturing key demographic details, loan characteristics, and loan default statuses. It offers an opportunity for data analysis and machine learning tasks, particularly in predicting the risk of loan default. The primary objective is to predict whether a customer will default on their loan, which is defined as failing to make the required loan payments over an extended period. Typically, after 90 days of missed payments, the loan is classified as "in default."
The target variable in this analysis is the Current_loan_status, which has two possible outcomes: DEFAULT (indicating the borrower has missed payments) and NO DEFAULT (indicating the borrower is meeting their payment obligations). The remaining columns in the dataset, such as customer age, income, loan amount, loan grade, and credit history length, serve as features for the model, allowing us to assess the likelihood of default based on a customer’s demographic and financial information.

</p>

<hr/>

<h3>Goal</h3>

<ul>
<li>Understainding dataset</li>
<li>preprocessing dataset for each summary</li>
<li>goal_3</li>
<li>goal_4</li>
</ul>

</hr>

<h3>Tools and Techniques</h3>

<ul>
<li>Python</li>
<li>Pandas</li>
<li>Tensorflow</li>
<li>Scikit-learn</li>
<li>Google Colab</li>
</ul>

</hr>

<h3>Project Structure</h3>
<hr/>
<h4>Part I: Summary of Demographic Dataset</h4>

<p> Age Distribution: This graph shows the age range of applicants and helps identify which age groups are most likely to apply for loans. 

KDE (Kernel Density Estimate): The curvy line you see on top of the histogram is the KDE. This is a smoothed version of the histogram, which estimates the probability density function of the age variable. The KDE line shows where the data points are concentrated, providing a clearer view of the distribution's shape compared to the histogram alone.</p>

![Pic_1](https://github.com/user-attachments/assets/99ed9c4b-161b-4df2-aa1d-7244125a6933)


<p> Home Ownership vs Loan Acceptance: This count plot illustrates the relationship between home ownership and historical defaults, informing potential risk factors for loan acceptance.</p>


![pic_2](https://github.com/user-attachments/assets/86fba4d5-2d2c-4361-b7e3-065ffbd3da12)


<p> # Count of Loan Applicants by Income Category: This count plot provides insights into how many people fall into each income category, guiding the bank's loan policies.</p>


![pic_4](https://github.com/user-attachments/assets/4afbcb0e-ef5c-4c5c-9912-c97f70d35448)



<hr/>

<h4>Part II: Machine Learning</h4>
</br>

<p>
Logistic Regression and Forest Random Tree

<b>Model Summary:</b>

<ul>
<li>2 hidden layers (8 and 3 nodes)</li>
<li>50 EPOCH</li>
<li>Activation Functions: ReLU and sigmoid (Output)</li>
<li>Result: 72.82%</li>
</ul>

</p>


```python

# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf

#  Import and read the charity_data.csv.
import pandas as pd
application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
application_df.head()
```
<hr/>

<h4>Part III: CNN Convolutional Neuron Network</h4>

<p>
Deep Learning Model
</p>

```python

# Import our dependencies
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import tensorflow as tf

#  Import and read the charity_data.csv.
import pandas as pd
application_df = pd.read_csv("https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv")
application_df.head()


```
<p>
<b>Optimization Model 1 Summary:</b>

<ul>
<li>2 hidden layers (8 and 3 nodes)</li>
<li>50 EPOCH</li>
<li>Activation Functions: ReLU, <b>ELU</b> and Sigmoid (Output)</li>
<li>Result: 79.44%</li>
</ul>

<b>Optimization Model 2 Summary:</b>

<ul>
<li>3 hidden layers (8, 5, 3 nodes)</li>
<li>50 EPOCH</li>
<li>Activation Functions: ReLU and sigmoid (Output)</li>
<li>Result: 79.46%</li>
</ul>

<b>Optimization Model 3 Summary:</b>

<ul>
<li>3 hidden layers (8, 5, 3 nodes)</li>
<li>100 EPOCH</li>
<li>Activation Functions: ReLU, ELU and Sigmoid (Output)</li>
<li style="color: green;"><b>Result: 79.52%</b></li>
</ul>

</p>

<hr/>

<h3>Final Summary</h3>

<p>
With the loan dataset, we could ...
</p>
