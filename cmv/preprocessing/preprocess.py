# -*- coding: utf-8 -*-
"""
In preprocessing, we remove the edit lines in a reply, and normalize all quote lines and urls into special tokens.
"""

import re

MD_LINK = re.compile(r"\[(.*?)\]\((.+?)\)", re.DOTALL)


urls = '(?: %s)' % '|'.join("""http telnet gopher file wais
ftp https""".split())
ltrs = r'\w'
gunk = r'/#~:.?+=&%@!\-'
punc = r'.:?\-'
any_char = "%(ltrs)s%(gunk)s%(punc)s" % { 'ltrs' : ltrs,
                                     'gunk' : gunk,
                                     'punc' : punc }

url = r"""
    \b                            # start at word boundary
        %(urls)s    :             # need resource and a colon
        [%(any_char)s]  +?             # followed by one or more
                                  #  of any valid character, but
                                  #  be conservative and take only
                                  #  what you need to....
    (?=                           # look-ahead non-consumptive assertion
            [%(punc)s]*           # either 0 or more punctuation
            (?:   [^%(any_char)s]      #  followed by a non-url char
                |                 #   or end of the string
                  $
            )
    )
    """ % {'urls' : urls,
           'any_char' : any_char,
           'punc' : punc }
URL_REGEX = re.compile(url, re.VERBOSE | re.MULTILINE)


BOLD_PATTERN = re.compile(r'(\*\*|__)(.*?)\1')
ITALICS_PATTERN = re.compile(r'(\*|_)(.*?)\1')
EDIT_TOKEN = '_'.join('EDIT')
QUOTE_TOKEN = '_'.join('QUOTE')
URL_TOKEN = '_'.join('URL')


def is_edit_word(word):
    return len(word) < 6 and word.startswith('edit')


def is_edit_line(words):
    if words and is_edit_word(words[0]):
        return True
    if len(words) > 1 and is_edit_word(words[1]):
        return True
    return False


def preprocess(body, lower=True):
    """Preprocess the body in our data, replace edits and quotes
    to avoid information leak"""
    proc = []
    for line in body.splitlines():
        words = re.findall("\w+", line.lower())
        if is_edit_line(words):
            proc.append(EDIT_TOKEN)
        elif line.startswith("&gt;"):
            proc.append(QUOTE_TOKEN)
        else:
            proc.append(line.lower() if lower else line)
    return "\n".join(proc)


def clean_up_links(links):
    if not links:
        return links
    clean_links = [links[0]]
    for (i, v) in enumerate(links[1:len(links)]):
        pos, l, url_str = v
        if pos < links[i - 1][0] + len(links[i - 1][2]):
            # url regular expression failed, remove next link
            continue
        clean_links.append(v)
    return clean_links


def normalize_url(body, return_links=False):
    """NORMALIZE all url links into URL_TOKEN"""
    link_txt = {link: "" for link in URL_REGEX.findall(body)}
    for text, link in MD_LINK.findall(body):
        link_txt[link] = text
    links = []
    # get position of links first
    for k, v in link_txt.items():
        if v:
            url_str = '[%s](%s)' % (v, k)
        else:
            url_str = k
        pos = body.find(url_str)
        while pos != -1:
            links.append((pos, k, url_str))
            pos = body.find(url_str, pos + len(url_str))
    links.sort()
    links = clean_up_links(links)
    links = [v[1].lower() for v in links]
    link_txt = {k: link_txt[k] for k in link_txt if k.lower() in links}
    for k, v in link_txt.items():
        if v:
            body = body.replace('[%s](%s)' % (v, k),
                    '%s %s' % (v, URL_TOKEN))
        else:
            body = body.replace(k, URL_TOKEN)
    if return_links:
        return body, links
    return body


def is_all_underlines(s):
    match = re.match('_+', s)
    if match:
        return len(match.group()) == len(s)
    return False


def filter_words_edit(sentences, op=False):
    """Remove edit lines for body"""
    result = []
    sentences = [s for s in sentences if s]
    for (i, s) in enumerate(sentences):
        if op and is_all_underlines(s):
            if i > len(sentences) - 3:
                # this is the new bot generated split lines followed by a quote
                break
            else:
                continue
        words = re.findall('\w+', s.lower())
        if not words:
            continue
        if s == EDIT_TOKEN:
            continue
        result.append(s)
    return result


def normalize_from_body(body, op=False):
    '''Normalize a body after removing edits.'''
    return "\n".join(filter_words_edit(
            normalize_url(preprocess(body)).split('\n'),
            op=op))
    
def remove_special_token(text):
    return text.replace(QUOTE_TOKEN, " ") \
            .replace(URL_TOKEN, " ") \
            .replace(EDIT_TOKEN, " ")

