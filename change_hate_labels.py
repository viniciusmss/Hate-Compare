from hate_classification import hate_classification


def change_hate_labels(tweets, raw_labels):
    ''' Change hate speech labels (0) to directed (3) / generalized labels (4)
        Shifts class numbers to the left so that class labels start from zero.
        Returned labels:

            (0) : Offensive
            (1) : Neither
            (2) : Directed hate speech
            (3) : Generalized hate speech

    '''
    labels = raw_labels.copy()

    for i, (tweet, label) in enumerate(zip(tweets, raw_labels)):

        if label == 0:  # If hate speech
            labels[i] = hate_classification(tweet)

    return labels - 1
