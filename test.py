from preprocess import preprocess

def _test_preprocess():

    assert " HASHTAGHERE " == preprocess("#iam1hashtag")
    assert " URLHERE " == preprocess("https://seminar.minerva.kgi.edu")
    assert " MENTIONHERE " == preprocess("@vinimiranda")
    assert ' ' == preprocess("        ")
    assert " & MENTIONHERE URLHERE HASHTAGHERE " == \
        preprocess("&amp;@vinimiranda    https://seminar.minerva.kgi.edu     #minerva    ")

if __name__ == "__main__":
    print("Testing preprocessing function...")
    _test_preprocess()

    print("All tests were successful.")
