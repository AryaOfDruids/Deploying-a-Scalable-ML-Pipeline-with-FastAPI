# Model Card

For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
This model was built by Tabitha Rhodes (trhod65) for the Udacity "Deploying a Scalable ML Pipeline with FastAPI" project in October 2026. It is a binary classifier that predicts whether a person earns more than $50K a year based on census information. The model is a scikit-learn `GradientBoostingClassifier` with 200 trees, a maximum tree depth of 5, a learning rate of 0.1, and a random state of 42. Categorical features are one-hot encoded and the label is binarized with the `process_data` function in `ml/data.py`. The trained model and encoder are saved as `model/model.pkl` and `model/encoder.pkl`.

A random forest was also tried. It scored a lower F1 (about 0.69 compared to 0.72), and its saved file was about 57 MB compared to under 1 MB for gradient boosting, so gradient boosting was chosen.

## Intended Use
The model is intended for learning and demonstration purposes, showing how to train, test, and deploy a model through a REST API built with FastAPI. It can be used to explore how demographic and work-related features relate to income in the 1994 census. It should not be used to make real decisions about individuals, such as hiring, lending, housing, or pay.

## Training Data
The data comes from the UCI Census Income (Adult) dataset, which was extracted from the 1994 U.S. Census database (https://archive.ics.uci.edu/ml/datasets/census+income). The provided `census.csv` file has 32,561 rows and 15 columns: 14 features and the `salary` label. Eight features are categorical (workclass, education, marital-status, occupation, relationship, race, sex, and native-country) and six are numeric (age, fnlgt, education-num, capital-gain, capital-loss, and hours-per-week). About 76% of people earn <=50K and about 24% earn >50K, so the classes are imbalanced. Missing values in workclass, occupation, and native-country are recorded as "?" and were kept as their own category.

The data was split into 80% training data (26,048 rows) and 20% test data (6,513 rows) using `train_test_split` with a random state of 42. The split was stratified on `salary` so both sets keep the same class balance.

## Evaluation Data
The model was evaluated on the 20% test set (6,513 rows) that was held out from training. The test set was processed with the encoder and label binarizer that were fit on the training data. The model was also evaluated on slices of the test set for every value of every categorical feature, and those results are saved in `slice_output.txt`.

## Metrics
The model was evaluated with precision, recall, and F1 score, where >50K is the positive class. Precision is the share of people predicted to earn >50K who really do, recall is the share of people who really earn >50K that the model finds, and F1 is the harmonic mean of the two.

On the full test set the model scored:

| Metric    | Score  |
|-----------|--------|
| Precision | 0.7850 |
| Recall    | 0.6728 |
| F1        | 0.7246 |

Some examples of performance on data slices from `slice_output.txt`:

| Slice                        | Count | Precision | Recall | F1     |
|------------------------------|-------|-----------|--------|--------|
| sex: Female                  | 2,158 | 0.7990    | 0.6327 | 0.7062 |
| sex: Male                    | 4,355 | 0.7826    | 0.6803 | 0.7279 |
| race: Black                  | 662   | 0.7941    | 0.5806 | 0.6708 |
| race: White                  | 5,533 | 0.7882    | 0.6809 | 0.7306 |
| education: Bachelors         | 1,096 | 0.7745    | 0.8330 | 0.8026 |
| education: HS-grad           | 2,120 | 0.8039    | 0.3628 | 0.5000 |
| native-country: Mexico       | 111   | 0.5000    | 0.3333 | 0.4000 |

## Ethical Considerations
The dataset includes sensitive attributes such as race, sex, marital status, and native country, and the model uses them as features. The slice results show that recall is lower for women than for men and lower for Black people than for White people, which means the model misses more high earners in those groups. The data is also from 1994, so it reflects the pay gaps and social conditions of that time, and the model can repeat those biases. Small groups in the data, such as some countries and some workclass values, have very few rows, so their scores are not reliable.

## Caveats and Recommendations
The model has higher precision than recall, so it is better at being right when it predicts >50K than at finding everyone who earns >50K. Slices with very few rows can show misleading scores. For example, slices with no positive examples or no positive predictions can show a precision or recall of 1.0 because `zero_division=1` is used. Income values from 1994 are not adjusted for inflation, so the >50K cutoff means something very different today. Future work could include tuning hyperparameters with cross-validation, adjusting the decision threshold to improve recall, scaling the numeric features, and using more recent census data. Any real-world use would need a much deeper fairness review first.
