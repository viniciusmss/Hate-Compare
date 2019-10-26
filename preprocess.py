import re
import html
from string import punctuation

def preprocess(text_string):

    # Casing should not make a difference in our case
    text_string = text_string.lower()

    # Regex
    html_pattern = r'(&(?:\#(?:(?:[0-9]+)|[Xx](?:[0-9A-Fa-f]+))|(?:[A-Za-z0-9]+));)'
    space_pattern = '\s+'
    giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    mention_regex = '@[\w\-]+'
    hashtag_regex = '#[\w\-]+'

    # First, add space surrounding HTML entities
    text_string = re.sub(html_pattern, r' \1 ', text_string)

    # Now, if we wish to find hashtags, we have to unescape HTML entities
    text_string = html.unescape(text_string)

    # From Udacity TV script generation project
    # Replace some punctuation by dedicated tokens
    symbol_to_token = {
        '.' : '||Period||',
        ',' : '||Comma||',
        '"' : '||Quotation_Mark||',
        ';' : '||Semicolon||',
        '!' : '||Exclamation_Mark||',
        '?' : '||Question_Mark||',
        '(' : '||Left_Parenthesis||',
        ')' : '||Right_Parenthesis||',
        '-' : '||Dash||',
        '\n' : '||Return||'
    }

    # Next, find URLs
    text_string = re.sub(giant_url_regex, ' URLHERE ', text_string)

    # Then, tokenize punctuation
    for key, token in symbol_to_token.items():
        text_string = text_string.replace(key, ' {} '.format(token))

    # Finally, remove spaces and find mentions and hashtags
    text_string = re.sub(hashtag_regex, ' HASHTAGHERE ', text_string)
    text_string = re.sub(mention_regex, ' MENTIONHERE ', text_string)
    text_string = re.sub(space_pattern, ' ', text_string)

    return text_string
