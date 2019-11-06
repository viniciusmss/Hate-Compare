#!/bin/bash
echo"Downloading Google News vectors"
wget https://doc-0g-8s-docs.googleusercontent.com/docs/securesc/ha0ro937gcuc7l7deffksulhg5h7mbp1/ica2ltpf5rm3rrrdira9l1jqgd6ftgt2/1536703200000/06848720943842814915/*/0B7XkCwpI5KDYNlNUTTlSS21pQmM?e=download -O src/GoogleNews-vectors-negative.tar.gz


echo"Downloading GloVe Vectors"
wget https://nlp.stanford.edu/data/glove.42B.300d.zip -O src/glove.42B.300d.zip

echo"Downloading FastText Vectors"
wget https://s3-us-west-1.amazonaws.com/fasttext-vectors/crawl-300d-2M.vec.zip -O src/crawl-300d-2M.vec.zip

