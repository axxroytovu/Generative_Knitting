import nltk
import numpy as np
from nltk.chunk.regexp import *
from nltk.corpus import wordnet as wn
from gensim.models import Word2Vec
import gensim.downloader as api
import gensim
import ssl
import matplotlib.pyplot as plt

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

model_glove_twitter = api.load("glove-twitter-25")

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
                    
colors = """alizarin
amaranth
amber
amethyst
apricot
aqua
aquamarine
asparagus
auburn
azure
beige
bistre
black
blue
blue-green
blue-violet
bondi-blue
brass
bronze
brown
buff
burgundy
camouflage-green
caput-mortuum
cardinal
carmine
carrot-orange
celadon
cerise
cerulean
champagne
charcoal
chartreuse
cherry-blossom-pink
chestnut
chocolate
cinnabar
cinnamon
cobalt
copper
coral
corn
cornflower
cream
crimson
cyan
dandelion
denim
ecru
emerald
eggplant
falu-red
fern-green
firebrick
flax
forest-green
french-rose
fuchsia
gamboge
gold
goldenrod
green
grey
han-purple
harlequin
heliotrope
hollywood-cerise
indigo
ivory
jade
kelly-green
khaki
lavender
lawn-green
lemon
lemon-chiffon
lilac
lime
lime-green
linen
magenta
magnolia
malachite
maroon
mauve
midnight-blue
mint-green
misty-rose
moss-green
mustard
myrtle
navajo-white
navy-blue
ochre
office-green
olive
olivine
orange
orchid
papaya-whip
peach
pear
periwinkle
persimmon
pine-green
pink
platinum
plum
powder-blue
puce
prussian-blue
psychedelic-purple
pumpkin
purple
quartz-grey
raw-umber
razzmatazz
red
robin-egg-blue
rose
royal-blue
royal-purple
ruby
russet
rust
safety-orange
saffron
salmon
sandy-brown
sangria
sapphire
scarlet
school-bus-yellow
sea-green
seashell
sepia
shamrock-green
shocking-pink
silver
sky-blue
slate-grey
smalt
spring-bud
spring-green
steel-blue
tan
tangerine
taupe
teal
tenné-(tawny)
terra-cotta
thistle
titanium-white
tomato
turquoise
tyrian-purple
ultramarine
van-dyke-brown
vermilion
violet
viridian
wheat
white
wisteria
yellow
zucchini"""

cparsed = [c.replace("-", " ") for c in colors.split("\n")]
cparsed = [c for c in cparsed if c in model_glove_twitter]
print(cparsed[0])

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
    "(S (NP me) (VP (V am) (NP sorry)))",
    "(S (NP something) (VP (V struck) (NP me)) (PP (P in) (NP the rear)))",
    "(S (NP me) (VP (RB just) (V arrived) (NP here)))"
]

for s in sentences:
    entities = nltk.tree.Tree.fromstring(s)
    entities.pretty_print()
    patterns, colors, phrases = tree_parse(entities)
    print(patterns, colors, phrases)
    for p in patterns:
        img = model_glove_twitter[p] > 0
        plt.imshow(img.reshape((5,5)))
        plt.title(p)
        plt.show()
    for c in colors:
        cols = sorted(cparsed, key=lambda k: model_glove_twitter.similarity(k, c.lower()), reverse=True)
        print(c, cols[:3])
        
        
