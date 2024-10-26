import nltk
import numpy as np
from nltk.chunk.regexp import *

def tree_parse(tree):
    patterns = []
    colors = []
    phrases = []
    print(tree.label())
    match tree.label():
        case "S" | "VP":
            for i in range(len(tree)):
                print(tree[i])
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
                print(tree[i])
                if tree[i].label() == "P":
                    prep = " ".join(tree[i].leaves())
                else:
                    pat, col, phr = tree_parse(tree[i])
                    cols.extend(col)
            phrases.append((prep, cols))
    return patterns, colors, phrases
                    
            


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
    "(S (NP I) (VP (V 'm) (NP so sorry)))",
    "(S (NP Something) (VP (V struck) (NP me)) (PP (P in) (NP the rear)))",
    "(S (NP I) (VP (RB just) (V ended up) (NP here)))"
]

for s in sentences:
    entities = nltk.tree.Tree.fromstring(s)
    entities.pretty_print()
    patterns, colors, phrases = tree_parse(entities)
    print(patterns, colors, phrases)
