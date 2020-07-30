---
layout: post
title: Naive Bayes Classification Algorithm
subtitle: With Numpy
bigimg: /img/Bayes-Theorum.png
tags: [Algorithms, Bayes Theorum, Computer Science]

---
# Description
If you want to know about Naive Bayesian Classification Algorithms, you first need to know about what they are based on. In 
statistics, there is a family of classification algorithms that rely on the underlying probabilistic assumptions about the
distribution of features, datapoints and outcomes in a dataset. These algorithms and classifiers are based on applying Bayes' 
Theorum to a problem that can be classified by a defined state -- for example, A or B, or, 0 or 1.

Theorectically, Bayes' Theorum describes the probability of an event occuring, predicated on the prior knowledge related to 
that event and the subsequent outcome. For example, if it is known that a person has a particular gene associated with cancer, 
Bayes' Theorum can help assess the probabilistic risk of developing cancer by evaluating various criterion, including that 
specific genes prescence. This allows for more nuanced assessment of a particular combination of factors specific to that 
context. In application, when interpreting the probability, the theorum expresses how the probability of an event should be 
updated to reflect the changing evidence over time. This implies that as more information comes to light, the probability 
associated with a specific outcome will change according to the new information and the distribution of the factors that are 
being evaluated. For example, as medicine becomes more advanced, if a breakthrough comes out that allows for a change to made 
to a persons DNA, and they have a cancer-causing gene, the probability you develop cancer, if you have this procedure, will 
fall -- which in term can be accounted for with Bayes' Theorum (more specifically a Naive Bayes Classification (NBC) 
Algorithm).

<img src="/img/bayes-equation.png">

# Use Case
Naive Bayes' Algorithms are so called because they make naive assumptions about the relationships between features,
datapoints and outcomes. One element that underlies all NBC's is that the assumed value associtaed with a particular feature is 
independent of all others -- meaning that it does not consider the correlation between features. This is a major factor to 
consider when modeling since most features in large datasets will be correlated with one another in some way, therefore, using 
an NBC might not be the best choice with large data. A major benefit of NBC's is the fact they only require a small number of 
training data to estimate the parameters necessary for modeling; this is why they are most often used on small datasets.

According to the Python Data Science Handbook, there are several advantages to using NBC's:
- They train and predict quickly
- Easily interpretable, probabilitics predictions
- Few tunable parameters

These advantages make NBC's a great model to use as a baseline and to build on with more complicated modeling as you analyze 
the data across various relationships and feature types. They perform best when model complexity is less important and the 
naive assumptions underlying the model actually match the those in the data; this includes if the features in the data are free 
of multicollinearity (high correlation between features) and / or the dimensionality (absolute number of features) is high.

# How - To
This section will talk about the mathematics and process that I went through in order to create a Naive Bayes' Classification 
Algorithm using Numpy.

## DAY 1
First, I needed to understand and look at the probabilistic model that underlies the algorithm. The foundational premise is 
based on the conditional probability of features (x<sub>1<sub> ,..., x<sub>n<sub>), in relation to each outcome(k) -- 
probability of k, based on x<sub>1<sub> through x<sub>n<sub>. Using Bayes' Theorum, the conditional probability can be 
expressed by the form shown in the image above, reproduced here.

<img src="/img/bayes-equation.png">

It can be concisely written as:

```python
                 prior x likelihood
    posterior = --------------------
                      evidence
```

Second, I needed to make this equation interpretable -- in practice, since we are only interested in the probability in 
relation to k possible outcomes, we care about the numerator of this equation, since the denominator does not contain k. In the 
code provided below, you can see how I calculated the prior and likelihood, and then calling them jointly in my fit function.

```python
def calculate_priors(self, X, y):
  # class_types, count_per_class per each classification outcome type
  class_types, count_per_class = np.unique(y, return_counts=True)
  # zip the class_types and n_class_types together so it's easier to
  # reference later when I calculate probilities
  class_type_dist = list(zip(class_types, count_per_class))
  # find the total number of datapoints, and sum them together
  total_ = [c[1] for c in class_type_dist]
  total = sum(total_)
  # find the percentage distribution of each class_type -- set the priors 
  # parameter equal to the percentage of prior occurrences
  self.priors = np.array([c[1] / total for c in class_type_dist])

def calculate_likelihood(self, X, y):
  # identify the unique number of classification types (class_types)
  # set parameter class_types equal to them
  self.class_types = np.unique(y)
  # identify the number_rows, number_features in the dataset from X
  n_rows, n_features = X.shape
  # set parameters mean, variance equal to numpy arrays with zeros
  self.mean = np.zeros((n_rows, n_features), dtype=np.float64)
  self.variance = np.zeros((n_rows, n_features), dtype=np.float64)
  # loop over the different idx and class_type in the class_types parameter
  for idx_class_type, class_type in enumerate(self.class_types):
      # set the X datapoints equal to the respective class_type
      X_classes = X[y == class_type]
      # calculate the mean of the X datapoints for each feature and set them
      # equal to the self.mean parameter for the specified idx
      self.mean[idx_class_type, :] = X_classes.mean(axis=0)
      # calculate the variance of the X datapoints for each feature and set them
      # equal to the self.mean parameter for the specified idx
      self.variance[idx_class_type, :] = X_classes.var(axis=0)

## FIT
def fit(self, X, y):
    # calling calculate_priors function
    self.calculate_priors(X, y)
    # calling calculate_likelihood function
    self.calculate_likelihood(X, y)
```

## DAY 2
