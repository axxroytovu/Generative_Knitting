import nltk
from nltk.corpus import stopwords
import numpy as np
from nltk.chunk.regexp import *
from gensim.models import Word2Vec
import gensim.downloader as api
import gensim
import ssl
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import requests

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

model_glove_wiki = api.load("glove-wiki-gigaword-100")
nltk.download("stopwords")
stop_words = set(stopwords.words('english'))

def parse_str(instring):
    c = [kword for kword in instring.split(" ") if kword not in stop_words]
    if not c:
        return instring.split(" ")
    return c

def tree_parse(tree):
    patterns = []
    colors = []
    phrases = []
    #print(tree.label())
    match tree.label():
        case "S" | "VP":
            for i in range(len(tree)):
                #print(tree[i])
                if not isinstance(tree[i], nltk.tree.Tree):
                    continue
                pat, col, phr = tree_parse(tree[i])
                patterns.extend(pat)
                colors.extend(col)
                phrases.extend(phr)
        case "NP" | "N":
            colors.append(" ".join(tree.leaves()))
        case "V":
            patterns.append(" ".join(tree.leaves()))
        case "PP":
            prep = ""
            cols = []
            for i in range(len(tree)):
                #print(tree[i])
                if tree[i].label() == "P":
                    prep = " ".join(tree[i].leaves())
                else:
                    pat, col, phr = tree_parse(tree[i])
                    cols.extend(col)
            phrases.append((prep, cols))
    return patterns, colors, phrases
                    
color_file = requests.get("https://xkcd.com/color/rgb.txt")
valid_colors = [c.split("\t")[0] for c in color_file.text.split("\n")]

print(valid_colors[0])

chunk_parser = RegexpParser("""
S: {<NP> <VP>}
VP: {<RB>? <V> <N.*>?}
PP: {<P> <NP>} # PrepPhrase
NP: {<DT>? <RB>? <JJ>* <N>*} # NP
N: {<NN.*|PRP.*|WDT|WP.*>}
P: {<IN>} # Prepositions
V: {<V.*>} # Verb
""")

sentences = [
    "(S (NP sokka) (VP (V is) (NP sorry)))",
    "(S (NP something) (VP (V struck) (NP sokka)) (PP (P in) (NP the rear)))",
    "(S (NP sokka) (VP (RB just) (V arrived) (NP here)))",
    "(S (NP five seven then five syllables) (VP (V mark) (NP haiku)))",
    "(S (NP sokka) (VP (V is) (NP a remarkable oaf)))",
    "(S (NP other people) (VP (V call) (NP me) (NP sokka)) (PP (P in) (NP the water tribe)))",
    "(S (NP sokka) (VP (V is) (NP not an oaf)))",
    "(S (NP a chittering monkey) (VP (V climbs) (NP treetops)))",
    "(S (NP a chittering monkey) (VP (V thinks) (NP himself) (NP tall)))",
    "(S (NP teacher) (VP (V thinks) (NP teacher smart)) (PP (P with) (NP your fancy words)))",
    "(S (NP haiku) (VP (V is) (NP not so hard)))",
    "(S (NP people) (VP (V spend) (NP seasons) (NP mastering the form the style)))",
    "(S (NP nobody) (VP (V calls) (NP haiku) (NP easy)))",
    "(S (NP sokka) (VP (V calls) (NP haiku) (NP easy)))",
    "(S (NP sokka) (VP (V paddles) (NP my canoe)))",
    "(S (NP sokka) (VP (V paddles) (NP your butt)))",
    "(S (NP nuts) (NP fruits) (NP plums) (VP (V drop)))",
    "(S (NP nuts) (NP fruits) (NP plums) (VP (V is) (NP ready to be squashed)))"
]
locked_colors = {}
used_colors = set()
for s in sentences:
    entities = nltk.tree.Tree.fromstring(s)
    entities.pretty_print()
    patterns, colors, phrases = tree_parse(entities)
    print(patterns, colors, phrases)
    determined_colors = [locked_colors[c] for c in colors if c in locked_colors]
    for c in colors:
        if c in locked_colors:
            continue
        cols = sorted(valid_colors, key=lambda k: model_glove_wiki.n_similarity(parse_str(k), parse_str(c)), reverse=True)
        print(model_glove_wiki.most_similar(positive=parse_str(c))[:3])
        i = 0
        while "xkcd:"+cols[i] in determined_colors or "xkcd:"+cols[i] in used_colors:
            i += 1
        determined_colors.append("xkcd:"+cols[i])
        locked_colors[c] = "xkcd:"+cols[i]
        used_colors.add("xkcd:"+cols[i])
        print(c, cols[i])
    for p in patterns:
        img = model_glove_wiki[p]
        plt.imshow(img.reshape((10,10)), cmap=ListedColormap(determined_colors))
        plt.title(s)
        plt.show()
        
        
