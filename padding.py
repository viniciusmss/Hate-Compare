def create_pad_fn(max_length):

    def pad_tweets(tweet, max_length=max_length):
        # Do not cut tweet short if it's too long

        # Retrieve tweet word count
        word_count = len(tweet.split())

        # Check how much padding will be needed
        n = max_length - word_count if word_count < max_length else 0

        # Pad tweet
        padded_tweet = ''.join(['<PAD> '] * n + [tweet])

        return padded_tweet

    return pad_tweets

def pad_tweets(tweet, max_length=10):
    # Do not cut tweet short if it's too long

    # Retrieve tweet word count
    word_count = len(tweet.split())

    # Check how much padding will be needed
    n = max_length - word_count if word_count < max_length else 0

    # Pad tweet
    padded_tweet = ''.join(['<PAD> '] * n + [tweet])

    return padded_tweet
