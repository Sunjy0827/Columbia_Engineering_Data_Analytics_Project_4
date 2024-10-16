# Columbia_Engineering_Data_Analytics_Project_4

<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Optimization Models Summary</title>
  <style>
    table {
      width: 100%;
      border-collapse: collapse;
    }
    th, td {
      border: 2px solid black;
      padding: 8px;
      text-align: left;
    }
    th {
      background-color: #f2f2f2;
      font-weight: bold;
    }
    td {
      color: black;
    }
    .highlight_1 {
      color: green;
      font-weight: bold;
    }
    .highlight_2 {
        color: red;
        font-weight:bold;
    }
    ul {
      padding-left: 20px;
    }
    ul ul {
      padding-left: 15px;
    }
  </style>
</head>

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
<li>Cleaning dataset</li>
<li>Preprocessing dataset</li>
<li>Making Machine Learning Models
<ul>
<li>Logistic Regression Model</li>
<li>Decision Tree Model</li>
<li>Random Forest Model</li>
<li>Neural Network Model</li>
<li>Deep Learning</li>
</ul>
</li>
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
<p>
Before the model was fit to the data, the data was split into training and testing subsets, and categorical variables were encoded into dummy variables. The training and testing sets contained 75% and 25% of the total dataset, respectively. The same training and testing sets were used for all 3 types of models (logistic, tree, random forest). Dummy variables were a method to make all the feature variables numeric and feedable into the sigmoid function underlying logistic regression. 
</p>

<p>
In addition to predicting whether a loan is defaulted or not, the three types of models (logistic, tree, and random) predict the <i> probability </i> of the loan being defaulted or not. In fact, the logistic regression uses this estimated probability to predict which one of the 2 categories (default or not) the loan falls. In a tree model, the probability of being defaulted for a test data point is the proportion of training data that have a target variable values of 'default' in the leaf in which the test data point falls (https://medium.com/ml-byte-size/how-does-decision-tree-output-predict-proba-12c78634c9d5). For the probability of not being defaulted, the same principle applies. 
</p> 

<p>
The three types of models were evaluated with confusion matrices and classification reports to determine their robustness. The following statistics were generated. 
</p>


<p>
<b>Logistic Regression Summary:</b>
</p>

<ul>
<li>Accuracy: 0.95 (No Default)</li>
<li>Precision: 0.97 (No Default), 0.88 (Default)</li>

<li>Recall: 0.97 (No Default), 0.87 (Default)</li>

</ul>

</p>

<p>
<b>Tree Summary:</b>
</p>

<ul>
<li>Accuracy: </li>
<li>Precision:  (No Default),  (Default)</li>

<li>Recall:  (No Default),  (Default)</li>

</ul>

</p>

<p>
<b>Random Forest Summary:</b>
</p>

<ul>
<li>Accuracy: </li>
<li>Precision:  (No Default),  (Default)</li>

<li>Recall:  (No Default),  (Default)</li>

</ul>

</p>




<hr/>

<h4>Part III: Neural Network and Deep Learning Model</h4>

<p>
<b>Dataset Preparation</b><br/>
The cleaned dataset was divided into two subsets:
<ul>
<li>df_w_bucket: Includes bucketed or categorized variables.</li>
<li>df_wo_bucket: Excludes bucketed or categorized variables.</li>
</ul>
The purpose of this separation was to analyze which type of variables—numerical or categorical—perform better when applied to neural network models.

<b>Experimentation with Neural Network Models</b><br/>
Several experiments were conducted using the two datasets to optimize the neural network’s performance. Key hyperparameters were adjusted to explore their impact on model accuracy:
<ul>
<li>Number of nodes in each layer</li>
<li>Number of hidden layers</li>
</ul>
<b>Results and Insights</b><br/>
The results were highly promising, with all tested models achieving over 90% accuracy. This indicates that the models are well-suited for practical applications.
</br></br>

<b>In conclusion</b>, the neural network models performed consistently across various configurations and can be effectively utilized in real-world scenarios. Further tuning may enhance the results, but the current models are already reliable and robust.
</p>

<table>
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Model</th>
      <th>Hidden Layers</th>
      <th>EPOCH</th>
      <th>Activation Functions</th>
      <th>Result</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td class="highlight_1" rowspan="3">df_w_buckets</td>
      <td><b>Optimization Model 1</b></td>
      <td>2 layers (8 and 4 nodes)</td>
      <td>50</td>
      <td>ReLU, Sigmoid (Output)</td>
      <td>95.7%</td>
    </tr>
    <tr>
      <td><b>Optimization Model 2</b></td>
      <td>2 layers (6, 4 nodes)</td>
      <td>50</td>
      <td>ReLU, Sigmoid (Output)</td>
      <td>95.6%</td>
    </tr>
    <tr>
      <td><b>Optimization Model 3</b></td>
      <td>1 layers (8 nodes)</td>
      <td>50</td>
      <td>ReLU, Sigmoid (Output)</td>
      <td>96.2%</td>
    </tr>
    <tr>
      <td class="highlight_2" rowspan="3">df_wo_buckets</td>
      <td><b>Optimization Model 1</b></td>
      <td>2 layers (8 and 4 nodes)</td>
      <td>50</td>
      <td>ReLU, <b>ELU</b>, Sigmoid (Output)</td>
      <td>93.0%</td>
    </tr>
    <tr>
      <td><b>Optimization Model 2</b></td>
      <td>2 layers (6, 4 nodes)</td>
      <td>50</td>
      <td>ReLU, Sigmoid (Output)</td>
      <td>93.1%</td>
    </tr>
    <tr>
      <td><b>Optimization Model 3</b></td>
      <td>1 layers (8 nodes)</td>
      <td>50</td>
      <td>ReLU, Sigmoid (Output)</td>
      <td>93.2%</td>
    </tr>
  </tbody>
</table>

</body>
</html>






<hr/>

<h3>Final Summary</h3>

<p>
With the loan dataset, we could ...
</p>
