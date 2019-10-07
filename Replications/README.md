# Replication Reports

### Table of Replications

1. Davidson, T., Warmsley, D., Macy, M., & Weber, I. (2017, May). Automated hate speech detection and the problem of offensive language. In _Eleventh international aaai conference on web and social media_. [Original Repository](https://github.com/t-davidson/hate-speech-and-offensive-language).
2. Badjatiya, P., Gupta, S., Gupta, M., & Varma, V. (2017, April). Deep learning for hate speech detection in tweets. In _Proceedings of the 26th International Conference on World Wide Web Companion_ (pp. 759-760). International World Wide Web Conferences Steering Committee. [Original Repository](https://github.com/pinkeshbadjatiya/twitter-hatespeech).

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

---

### Deep learning for hate speech detection in tweets
_[Cloned repository with replicated files](./Replications/Badjatiya2017)_

Replicating this research was very laboursome both due to the large amount of models the authors implemented and due to several faults of their source code. Finally, the principal result published by the authors is computed incorrectly and is thus false. Most of the problems which can be addressed have been correct in my version of the source code. A non-exhaustive list of issues follows.
1. The authors provide instructions to obtain the dataset from the repository of the original author which contains tweet IDs and labels. Many of these tweets cannot be retrieved by Twitter's API since most users associated with hate speech tweets are no longer on the platform. The only way to obtain the original dataset is to contact the original author. To hide this fact is a shame since the attempt to retrieve these tweets through the API is an unnecessary waste of time. 
2. Neither the paper nor the repository clearly indicates the hyperparamaters used in training any of the models. For the deep learning models, the problem is mostly one of obscurity since the architecture of the networks is hard-coded and can hence be abstracted from the code. However, for the other models the reader can to intuit which hyperparameters were tried. One eventually succeeds in finding the right values after perusing the code base for long enough, but the lack of clarity burdens the replication unnecessarily.
3. The source code is messy and at times contains several variables and functions which are not utilized and whose purpose is unclear. 
4. There are serious mistakes in the repository, such as the one pointed out in this [closed issue](https://github.com/pinkeshbadjatiya/twitter-hatespeech/issues/6). Although the mistake is resolved in `lstm.py`, it is still present is `cnn.py`. This specific mistake was not part of the code which the authors used to obtain their results, so it does not affect the validity of the metrics presented.
5. The authors seem to consider `fastText` the simple procedure of updating word embeddings during training via backpropagation, which is certainly not the case. The authors do not use the `fastText` library at any point in their code and use pre-trained `GloVe` embeddings isntead of `fastText` embeddings. Furthermore, their implementation of the representation of a sentence as the average of its word embeddings is also incorrect, since their use of an average 1D pooling layer does not specify the right kernel size to average across the entire tweet, but rather only around two words at a time (i.e., the default parametrization of the layer with `kernel_size=2`)
6. Finally and most crucially, in Part C the authors represent a combination of a deep learning model and XGBoost. Their implementation is incorrect since the two models have different train and test sets. In other words, the deep learning model trained on the 10th fold of its cross-validation is saved to disk. Then, this model is loaded and its embeddings used as features for the XGBoost classifier. However, nowhere do the authors make sure that the training and test sets for these two models are the same. As a result, the XGBoost classifier is tested on the data that the deep learning model has already seen, explaining the extraordinary metric jump that the authors report as a result. Hence, the results reported in part C of this paper are __false__. 

5. 
5. 

