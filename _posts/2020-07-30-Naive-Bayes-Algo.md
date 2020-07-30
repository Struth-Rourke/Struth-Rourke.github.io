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
On the first day of this project, I struggled more than I care to admit. In following with the UPER (Understand, Plan, Execute, 
Review) methodology, I sought out to understand the mathematics and theory behind Naive Bayesian Algorithms. I was able to get 
a pretty intuitive understanding of the math after half the day and then sought out to Plan and Execute my code from there. 
Unfortunately, I neither understood nor planned enough, and so when I went to try and Execute by writing code, I ran into 
immediate problems. Including: failing to recognize the data type of my input variables, not referencing the correct indices, 
not using the inherent parameters in my class enough, and many other issues that all stem from not being as organized or 
systemized as I should've been. I finished the day discouraged and ready to change my project because I was not satisified with 
my progress and I didn't want to get behind.

Fortunately, tomorrow is always another day.

## DAY 2
So, on Day 2, I restarted and reframed everything as if I was operating on a blank slate.

First, I needed to understand and look at the probabilistic model that underlies the algorithm. The foundational premise is 
based on the conditional probability of features (x<sub>n<sub>), in relation to each outcome(k) -- probability of k, based on 
x<sub>1<sub> through x<sub>n<sub>. Using Bayes' Theorum, the conditional probability can be expressed by the form shown in the 
image above, reproduced here.

<img src="/img/bayes-equation.png">

It can be concisely written as:

```python
                 prior x likelihood
    posterior = --------------------
                      evidence
```

Second, I needed to make this equation interpretable -- in practice, since we are only interested in the probability in 
relation to k possible outcomes, we care about the numerator of this equation, since the denominator does not contain k. In the 
code provided below, you can see how I calculated the prior probabilities with a function and then called it in my fit 
function, jointly with my calculated likelihood function (for more specifics into the code, you can visit my repo named 
Naive_Bayes_Algo). By the end of the day, I had finished my fit method and had a more defined path for the rest of the project.

```python
def calculate_priors(self, X, y):
  class_types, count_per_class = np.unique(y, return_counts=True)
  class_type_dist = list(zip(class_types, count_per_class))
  total_ = [c[1] for c in class_type_dist]
  total = sum(total_)
  self.priors = np.array([c[1] / total for c in class_type_dist])

def fit(self, X, y):
    self.calculate_priors(X, y)
    self.calculate_likelihood(X, y)
```

## DAY 3
On the third day, I needed to better understand the process of using likelihood as an input into the probability denisty 
function (PDF) of a normal distribution. Below is the function that I used to calculate the PDF.

```python
def prob_density_function(self, mean, variance, x):
    exponent = np.exp((-(x - mean) ** 2) / (2 * variance))
    p_x_given_y = 1 / np.sqrt(2 * np.pi * variance) * exponent
    return p_x_given_y

```

Essentially, the PDF is the probability that a random variable falls within a particular range of values. It is the relative 
likelihood of that value occuring in relation to another. This is critical to understand because the posterior probabilities 
that we are calculating later on, as part of the predicted output probability, depend on the probability density function of 
each likelihood term.

Finally, using the mean, variance, and priors as inputs, I created a function to calculate the posterior probabilities 
associated with each classification outcome (or type) and returned the classification type with the highest projected 
probability given the features and their respective values. (For a more in-depth look at the code, you can visit my repo titled
Naive_Bayes_Algo). This allows for the model to assign an outcome label to a particular observation based on the features used 
and their values; over time, as more oberservations occur the distribution of outcomes changes, and therefore alters the 
classification type. Because I created a Gaussian Naive Bayes Classifier, which works best on normal distributions, if the 
distribution is indeed Gaussian the classification type will more than likely stay the same for a given obeservation regardless 
of sample size -- since samples approximate the population when they are Gaussian.

## DAY 4
The fourth and final day has been all about cleaning up, refactoring, polishing off the code and writing this article as a 
lookback at the entire project. Despite the struggle and difficulties I experienced early on, I'm very happy with the outcome 
and for sticking with it. It was challenging, but I learned a lot and appreciate the struggle that goes into learning 
something that is abstract.

