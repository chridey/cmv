import nltk
import re

url_re = re.compile(r"\(? *http:.+?\)?( |\Z)")

def cleanup(line):
    #convert all URLs to <URL>
    #remove lines starting with or close to the word "edit" CHECK
    #remove all sentences with CMV
    #any words starting with . are probably typos
    ret = []
    for sentence in nltk.sent_tokenize(line):
        if sentence.lstrip().startswith("&gt;") or sentence.lstrip().startswith("____"):
            continue

        words = nltk.word_tokenize(re.sub(url_re, " URLTOKEN ", sentence))
    
        if "edit" in ' '.join(words[:2]).lower():
            continue

        word_string = ' '.join(words).lower()

        if any(x in word_string for x in ('cmv',
                                          'change my view',
                                          'c my v',
                                          'delta',
                                          u'\u2206',
                                          'this is a footnote from your moderators',
                                          'just like to remind you of a couple of things',
                                          'firstly , please remember to',
                                          'if you see a comment that has broken one',
                                          'it is more effective to report it than downvote it',
                                          'speaking of which , downvotes',
                                          'any questions or concerns ?',
                                          'feel free to message us')) or \
            'speaking of which' and 'change views' in word_string or \
            'feel free' and 'message us' in word_string:
            continue

        ret.append(words)

    return ret
