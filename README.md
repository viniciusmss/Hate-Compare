# A Critical Evaluation of Hate Speech Classifiers

This contains replication and auxiliary resources to my Senior thesis, which is available [here](https://github.com/viniciusmss/Hate-Compare/blob/master/Capstone.pdf). My goal was to replicate three prominent papers in the field of hate speech classification and provide an application of transfer learning via ULMFiT to this task.

### Table of Contents

1. Data Preprocessing
2. Training and Testing Classifier
3. Data Augmentation
4. ULMFit
5. Augmented ULMFit
6. CV Augmented ULMFit

99\. Early Implementation with Class Breakdown

### Short Summaries

**1. Data Preprocessing.**
Focuses on showing text data preprocessing step by step. It makes use of helper functions in `utils.py` available in this repo. The data preprocessing in later notebooks is largely hidden under the hood due to fast.ai's API.

**2. Training and Testing Classifier.**
Aims to replicate Hemker (2018) and makes heavy use of helper functions in `classifier_utils.py`. The notebooks demonstrates how the parameters reported by Hemker (2018) do not yield a funtional classifier.

**3. Data Augmentation.** Augments the dataset according to Hemker's (2018) threshold augmentation procedure. Note that we use a slightly preprocessed version of the original dataset (available in the data folder) to facilitate augmentation.

**4. ULMFit.** A step-by-step implementation of the ULMFiT (Howard & Ruder, 2018) NLP transfer learning model. I implement discriminative fine-tuning, gradual unfreezing, and bidirectional models according to the original paper and fastai's (2019) repository, from where much of the code here is based.

**5. Augmented ULMFit.** Similar to \#4 but now we augment the hate speech class in the training set. 

**6. CV Augmented ULMFit.** Similar to \#5 but now we perform 5-fold cross validation.

**99. Early Implementation with Class Breakdown.** This is an early, draft implementation of \#1 and \#2 together while implementing the helper functions within the notebook. It's kept here for educational purposes (possibly). It is also contains the unit tests for the helper functions.

### Software Requirements

The `envs` folder contains `.yml` files that specify package versions and allow you to quickly build conda environment to reproduce the results on this repo. Note that:

- `environment1.yml` applies to notebooks 1-3 and the Davidson et al. (2017) replication. The environment was built and run on a 64-bit Windows 10 Home PC.
- `environment2.yml` applies to notebooks 4-6. The environment was built on a [Paperspace](https://www.paperspace.com/) Free-P5000 machine initialized on fast.ai's container.
- `environmentpy27.yml` applies to the replication of Badjatiya et al. (2017). It was built on a 64-bit Ubuntu Linux virtual machine running on a 64-bit Windows 10 home PC.

### References

I have drawn extensively from other authors to execute this implementation. I have tried to recognize their contribution by citing all works in the _References_ section at the end of this README and referencing the citation in the individual scripts where the contribution is featured.

Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May). Automated hate speech detection and the problem of offensive language. In _Eleventh International AAAI Conference on Web and Social Media._

fastai (2019). A Code-First Introduction to NLP course [GitHub repository]. Retrieved from https://github.com/fastai/course-nlp

Hemker, K. (2018). Data Augmentation and Deep Learning for Hate Speech Detection (Master's thesis). Retrieved from https://bit.ly/2SnjylP

Howard, J., & Ruder, S. (2018). Universal language model fine-tuning for text classification. _arXiv preprint arXiv:1801.06146_.

imanzabet. (2017, Aug 16). NLTK Named Entity recognition to a Python list [Forum response]. Retrieved from https://stackoverflow.com/a/31838373

Udacity. (2019). TV Script Generation. In _Deep Learning (PyTorch)_. Retrieved from https://github.com/udacity/deep-learning-v2-pytorch
