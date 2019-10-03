# Replication Reports

### Table of Replications

1. Thomas Davidson, Dana Warmsley, Michael Macy, and Ingmar Weber. 2017. "Automated Hate Speech Detection and the Problem of Offensive Language." ICWSM. [Original Repository](https://github.com/t-davidson/hate-speech-and-offensive-language).

---

### Automated Hate Speech Detection and the Problem of Offensive Language
_[Cloned repository with replicated files](./Replications/Davidson2017)_

This paper has been hard to replicate due to the contrived nature of the information provided by the authors in this repository. Some problems I have identified include:
1. The lack of a clear list of software requirements complicates managing a replication environment,
2. There are missing files in the repository which are mentioned within the notebooks (e.g., the `final_classifier.ipynb` notebook which mentions an "operational_classifier" notebook which does not exist),
3. The preprocessing steps inside of the notebook are faulty (e.g., URLs are replaced by the empty string instead of by a dedicated token, as otherwise indicated by the code comments),
4. The grid search pipeline shown in the `src` folder does not include the models and hyperparameters the authors investigated,
5. The authors do not explicitly discuss whether they are building character- or word-level models. Their code implements the former, but the feature count in the original notebooks indicates the latter. 
6. Stop words are not tokenized before being passed to the vectorizer, which is a mistake in the data preprocessing.

Thankfully, I was able to obtain the results from the top-performing model originally selected by the authors. That model is loaded and tested on the original dataset in the [Final Model Loader](https://github.com/viniciusmss/Hate-Compare/blob/master/Replications/Davidson2017/Final%20Model%20Loader.ipynb) notebook. Alternatively, the [classifier.py](https://github.com/viniciusmss/Hate-Compare/blob/master/Replications/Davidson2017/classifier/classifier.py) script in the `classifier` folder is also functional and can be used to test the top-performing classifier on new data.
