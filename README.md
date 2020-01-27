# A Multidimensional Comparative Analysis of Hate Speech Classifiers

This repository is a work in progress for my undergraduate thesis.

### Table of Contents

1. Data Preprocessing
2. Training and Testing Classifier
3. Data Augmentation
99. Early Implementation with Class Breakdown

### Short Summaries

**1. Data Preprocessing.**
Focuses on showing text data preprocessing step by step. It makes use of helper functions in `utils.py` available in this repo. The data preprocessing in later notebooks is largely hidden under the hood due to fast.ai's API.

**2. Training and Testing Classifier.**
Aims to replicate Hemker (2018) and makes heavy use of helper function in `classifier_utils.py`. The notebooks demonstrates how Hemker's (2018) stated parameters do not yield a funtional classifier.

**3. Data Augmentation.** Augments the dataset according to Hemker's (2018) threshold augmentation procedure. Note that we use a slightly preprocessed version of the original dataset (available in the data folder) to facilitate augmentation.




### References

I have drawn extensively from other authors to execute this implementation. I have tried to recognize their contribution by citing all works in the _References_ section at the end of this README and referencing the citation in the individual scripts where the contribution is featured.

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May). Automated hate speech detection and the problem of offensive language. In _Eleventh International AAAI Conference on Web and Social Media._

Hemker, K. (2018). Data Augmentation and Deep Learning for Hate Speech Detection (Master's thesis). Retrieved from https://bit.ly/2SnjylP

imanzabet. (2017, Aug 16). NLTK Named Entity recognition to a Python list [Forum response]. Retrieved from https://stackoverflow.com/a/31838373

Udacity. (2019). TV Script Generation. In _Deep Learning (PyTorch)_. Retrieved from https://github.com/udacity/deep-learning-v2-pytorch
