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

First, I needed to understand and look at the probabilistic model that underlies the algorithm. The foundational premise is 
based on the conditional probability of features (x<sub>1<sub> ,..., x<sub>n<sub>), in relation to each outcome(k) -- 
probability of k, based on x<sub>1<sub> through x<sub>n<sub>. Using Bayes' Theorum, the conditional probability can be 
expressed by the form shown in the image above, reproduced here.

<img src="/img/bayes-equation.png">

It can be concisely written as:

```py
                 prior x likelihood
    posterior = --------------------
                      evidence
```

Second, I needed to make this equation interpretable -- in practice, since we are only interested in the probability in 
relation to k possible outcomes, we care about the numerator of this equation, since the denominator does not contain k. In the 
code provided below, 

