# Columbia_Engineering_Data_Analytics_Project_4

<h2>Machine Learning Model with Loan-Dataset </h2>

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
<ul style= "padding-left: 15px;">
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

<p> Home Ownership vs Loan Acceptance:
The graph shows that homeowners have lower loan default rates than renters. Homeownership promotes financial stability, as homeowners prioritize mortgage payments, while renters face more volatility, increasing their default risk.</p>

![pic_2](https://github.com/user-attachments/assets/547f5b47-845d-4fe9-8cad-d3580be79f35)


<p>Count of Loan Applicants by Income Category:
The data shows loan applicants by income level, revealing that lower-income individuals have higher default rates, while higher-income applicants are less likely to default. This highlights the link between income and loan repayment behavior.</p>

![pic_4](https://github.com/user-attachments/assets/5d1b2e92-72d3-41b2-84be-d3e8dfd5b50d)



<p> Count of Loan Applicants by Income Category:  
The data shows historical defaults by age bracket, revealing that younger borrowers have higher default rates, while older groups tend to default less. This highlights the influence of age on loan repayment behavior.
</p>

![pic_6](https://github.com/user-attachments/assets/d2c26932-3328-42e7-b9f5-7f24c16b04fc)


<hr/>


<p> Number of Customers vs Employment Duration:
The bar chart represents the distribution of customers categorized by their employment duration and whether they have defaulted on a loan or not.</p>

<img src= "https://github.com/Sunjy0827/Columbia_Engineering_Data_Analytics_Project_4/blob/main/Images/pic_7.png" alt ="pic_7"/>

<p> Default Vs No Default by loan categorization
The donut charts provide a detailed visualization of loan defaults and no defaults categorized by the intent of the loan.
</p>

<img src= "https://github.com/Sunjy0827/Columbia_Engineering_Data_Analytics_Project_4/blob/main/Images/pic_8.png" alt ="pic_8"/>



<p> Number of Loans Vs Loan Grade with indication of Default and No Default  
The stacked bar chart shows the number of loans by loan grade and default status, as well as the default rate for each loan grade.

</p>

<img src="https://github.com/Sunjy0827/Columbia_Engineering_Data_Analytics_Project_4/blob/main/Images/pic_9.png" alt ="pic_9"/>

<hr/>


<hr/>
<h4>Part II: Machine Learning</h4>
</br>

<p>
Logistic Regression, Decision Trees, and Random Forest

<b>Model Summary:</b>
<p>
Before the models were fit to the data, the data was split into training and testing subsets, and categorical variables were encoded into dummy variables. The training and testing sets contained 75% and 25% of the total dataset, respectively. The same training and testing sets were used for all 3 types of models (logistic, tree, random forest). Dummy variables were a method to make all the feature variables numeric and feedable into the sigmoid function underlying logistic regression. For the tree and random forest models, the data was scaled. 
</p>

<p>
Next, the logistic regression, tree, and random forest models were fit to the data. For the trees, we initially did not set a limit on the depth of the tree but upon seeing its massive size, considerations about trees' tendency to overfit the training data, and our desire to build a model that includes the probabilities of an observation/loan being defaulted or not, we opted to set limits on the max depth, trying both a max depth of 5 and a max depth of 10. For our random forest model, we initially used 5000 trees for the model, but with a slow processing time, we adjusted the number of trees to 100. Since the metrics accuracy, precision, and recall seemed to be more or less the same as the forest with 5000 trees, we chose 100 trees for the random forest. 

</p>

<p>
In addition to predicting whether a loan is defaulted or not, the three types of models (logistic, tree, and random) predict the <i> probability </i> of the loan being defaulted or not. In fact, the logistic regression uses this estimated probability to predict which one of the 2 categories (default or not) the loan falls. In a tree model, the probability of being defaulted for a test data point is the proportion of training data that have a target variable values of 'default' in the leaf in which the test data point falls (https://medium.com/ml-byte-size/how-does-decision-tree-output-predict-proba-12c78634c9d5). For the probability of not being defaulted, the same principle applies. In a random forest model, the probability of a loan being defaulted is calculated by the proportion of trees in the forest that predict the loan will be defaulted (Cite this later).
</p> 

<p>
In printing out the tree, we noticed the term, 'gini index' in each node. Upon some research online, we clarified that gini index was a quantity between 0 and 1 that measured the degree of training observations in a tree node in falling into 1 class. A gini index closer to 1 means a larger amount of training observations not belonging to the same class. A gini index closer to 0 means a larger amount of training observations belonging to the same class. In other words, the gini index of a node measures how effective the split was that resulted in that particular node. 

</p>

<p>
The three types of models were evaluated with confusion matrices and classification reports to determine their robustness. The following statistics were generated. 
</p>


<p>
<b>Logistic Regression:</b>
</p>

<ul>
<li>Accuracy: 0.95 (No Default)</li>
<li>Precision: 0.97 (No Default), 0.88 (Default)</li>

<li>Recall: 0.97 (No Default), 0.87 (Default)</li>

</ul>

</p>

<p>
<b>Tree 1 (Max Depth 5):</b>
</p>

<ul>
<li> Accuracy: 0.94 </li>
<li>Precision: (No Default) 0.94, 0.94 (Default)</li>
<li>Recall: 0.99 (No Default),  0.74 (Default)</li>
</ul>

</p>


<p>
<b>Tree 2 (Max Depth 10):</b>
</p>


<ul>
<li> Accuracy: 0.96 </li>
<li>Precision: (No Default) 0.97, 0.93 (Default)</li>
<li>Recall: 0.98 (No Default),  0.87 (Default)</li>
</ul>

</p>

<p>
<b>Random Forest:</b>
</p>

<ul>
<li>Accuracy: 0.97</li>
<li>Precision: 0.97 (No Default), 0.94 (Default)</li>

<li>Recall: 0.98 (No Default), 0.89 (Default)</li>

</ul>

</p>

<p>
In evaluating the models, we focused on accuracy and emphasized recall(default) especially because this number signified the number of loans predicted to default out of all the loans that did actually default. In other words, it represented the financial risk of the lending business resulting from its failure to detect the customers who would default by using one of these models. First, logistic regression had a high accuracy of 95% but had a 87% recall (default), signaling a 13% financial loss for the company. For the trees, tree1 was disregarded simply for its lowest recall (default) of 74%. Next, even though logistic regression had an 87% recall (default), which was the same as tree2, tree2 outperformed logistic regression in all othe measures. Nonetheless, the random forest model either performed as well as or surpassed the tree and logistic regression models in all metrics of accuracy, precision, and accuracy, signaling its strong predictive abilities. 



<hr/>

<h4>Part III: Neural Network and Deep Learning Model</h4>

<p>
<b>Dataset Preparation</b><br/>
The cleaned dataset was divided into two subsets:
<ul>
<li>df_w_bucket: Includes bucketed or categorized variables.</li>
<li>df_wo_bucket: Excludes bucketed or categorized variables.</li>
</ul>
The purpose of this separation was to analyze which type of variables—numerical or categorical— perform better when applied to neural network models.

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
  <thead background-color: #f2f2f2;
      font-weight: bold;>
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
      <td rowspan="3" style="color: green; font-weight: bold;">df_w_buckets</td>
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
      <td rowspan="3" style="color: red; font-weight: bold;">df_wo_buckets</td>
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
<img src="https://github.com/Sunjy0827/Columbia_Engineering_Data_Analytics_Project_4/blob/main/Images/Model%20I-VI.png" alt="accuracy_progress_combined_image"/>
<hr/>
<h3>Final Summary</h3>
<p>
We developed a total of five different machine learning models, with 11 variations, including six versions based on neural networks and deep learning. Each model achieved an accuracy of over 93%, indicating strong performance suitable for real-world applications. However, to ensure transparency in the decision-making process, we recommend focusing on Logistic Regression and Decision Tree models, as these allow loan departments to better understand which variables most influence loan eligibility decisions. Additionally, instead of limiting the model to binary outcomes, we propose creating a multi-class model that categorizes applicants into different loan eligibility tiers, making the results more practical and aligned with real-world lending scenarios.

<img src="https://github.com/Sunjy0827/Columbia_Engineering_Data_Analytics_Project_4/blob/main/Images/final_image.PNG" alt="final_image"/>

<img >
</p>
