import pandas as pd
import wikipedia
import os
from tqdm import tqdm
from joblib import Parallel, delayed
import multiprocessing
import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from ipywidgets import interact, interactive, fixed, interact_manual,widgets
from IPython.display import display
import json
num_cores = multiprocessing.cpu_count()
print(num_cores)
wikipedia.set_lang("ru")
# DATA_PATH_LIST = ['D:','src2','taxonomy-enrichment','data','training_data']
DATA_PATH_LIST = ['.']
EMBEDDING_MODEL_FILENAME = "wiki_node2vec.bin"
DATA_PATH="/".join(DATA_PATH_LIST+["training_nouns.tsv"])
df = pd.read_csv(DATA_PATH,sep='\t')

def prestr(x):
    return str(x).replace('\"','').replace("'",'"')


class DefDict(defaultdict):
    def __missing__(self, key):
        self[key] = key
        return key


idx2syns = DefDict(lambda x: x)
for val in df.values:
    idx2syns[val[0]] = val[1]
    try:
        pidxs = json.loads(prestr(val[2]))
        concp = [el.split(",")[0] for el in json.loads(prestr(val[3]))]
        idx2syns.update(dict(zip(pidxs, concp)))
    except:
        print(prestr(val[2]))
        print(prestr(val[3]))

button = widgets.Button(description="Draw")
query = widgets.Text(
    value='МАТЬ',
    placeholder='Query',
    description='String:',
    disabled=False
)
display(button, query)


def creategraph(df):
    res = []
    for row in df.values:
        cohyps = row[1].split(",")
        for idx, cohyp in enumerate(cohyps):
            for parent in json.loads(prestr(row[2])):
                res.append((row[0] + '-' + str(idx), parent))
    return res


def graphdraw(b):
    print("graphdraw", query.value)
    subset = df[df['TEXT'].str.contains(query.value.upper())]
    g = nx.DiGraph()
    for el in subset.values:
        cohyps = el[1].split(",")
        print(cohyps)
        syns = idx2syns[el[0]]
        for child in cohyps:
            for parent in json.loads(prestr(el[2])):
                ed = g.add_edge(child, idx2syns[parent], label="is a")

    plt.figure(figsize=(15, 15))
    pos = nx.nx_agraph.graphviz_layout(g)
    nx.draw(g, with_labels=True, pos=pos)
    #     edge_labels=nx.draw_networkx_edge_labels(g,pos=pos)
    plt.show()


button.on_click(graphdraw)

from yargy.tokenizer import MorphTokenizer
tokenizer = MorphTokenizer()
text = '''Ростов-на-Дону
Длительностью 18ч. 10мин.
Яндекс.Такси
π ≈ 3.1415
1 500 000$
http://vk.com
'''
for line in text.splitlines():
    print([_.value for _ in tokenizer(line)])

from yargy import or_, rule
from yargy.predicates import normalized

RULE = or_(
    rule(normalized('dvd'), '-', normalized('диск')),
    rule(normalized('видео'), normalized('файл'))
)

from yargy import Parser
from yargy.pipelines import morph_pipeline


RULE = morph_pipeline([
    'dvd-диск',
    'видео файл',
    'видеофильм',
    'газета',
    'электронный дневник',
    'эссе',
])

parser = Parser(RULE)
text = 'Видео файл на dvd-диске'
for match in parser.findall(text):
    print([_.value for _ in match.tokens])

from yargy import Parser, rule, and_
from yargy.predicates import gram, is_capitalized, dictionary


GEO = rule(
    and_(
        gram('ADJF'),  # так помечается прилагательное, остальные пометки описаны в
                       # http://pymorphy2.readthedocs.io/en/latest/user/grammemes.html
        is_capitalized()
    ),
    gram('ADJF').optional().repeatable(),
    dictionary({
        'федерация',
        'республика'
    })
)


parser = Parser(GEO)
text = '''
В Чеченской республике на день рождения ...
Донецкая народная республика провозгласила ...
Башня Федерация — одна из самых высоких ...
'''
for match in parser.findall(text):
    print([_.value for _ in match.tokens])

from yargy import and_, not_
from yargy.tokenizer import MorphTokenizer
from yargy.predicates import is_capitalized, eq


tokenizer = MorphTokenizer()
token = next(tokenizer('Стали'))

predicate = is_capitalized()
print(predicate(token))

predicate = and_(
    is_capitalized(),
    not_(eq('марки'))
)
print(predicate(token))

from yargy import rule, or_


KEY = or_(
    rule('р', '.'),
    rule('размер')
).named('KEY')
VALUE = or_(
    rule('S'),
    rule('M'),
    rule('L'),
    rule('XS'),
).named('VALUE')
SIZE = rule(
    KEY,
    VALUE
).named('SIZE')
SIZE.normalized.as_bnf

parser = Parser(
    SIZE
)
text = 'размер M; размер A; размер XS;'
for match in parser.findall(text):
    print([_.value for _ in match.tokens])

from yargy import Parser, rule, and_, or_, not_
from yargy.interpretation import fact, attribute
from yargy.predicates import gram, is_capitalized, dictionary, eq
import re
import pandas as pd
from tqdm import tqdm_notebook
from gensim import utils

START = rule(
    or_(
        rule(gram('ADJF')),
        rule(gram('NOUN'))
    ).optional(),
    gram('NOUN')
)

START_S = or_(
    eq('такой'),
    eq('такие'),
)

KAK = eq('как')
INCLUDING = or_(
    or_(
        eq('в'),
        eq('том'),
        eq('числе'),
    ),
    eq('включающий'),
    or_(
        eq('включающий'),
        eq('в'),
        eq('себя'),
    ),
    or_(
        eq('включающие'),
        eq('в'),
        eq('себя'),
    ),
    eq('включающие'),
    eq('особенно'),

)

MID_S = or_(
    rule(
        or_(
            eq('такой'),
            eq('такие'),
        ),
        eq('как')
    )
)
ATAKJE = rule(
    eq(','),
    eq('а'),
    eq('также')
)

MID = or_(
    rule(
        eq('это')
    ),
    rule(
        eq('—')
    ),
    rule(
        eq('—'),
        eq('это')
    ),
    rule(
        eq('—'),
        not_(eq('км'))
    ),
    rule(
        or_(
            eq('и'),
            eq('или'),
        ),
        eq('другие')
    )
)

END = or_(
    rule(
        gram('NOUN'),
        gram('NOUN')
    ),
    rule(
        gram('ADJF').repeatable(),
        gram('NOUN')
    ),
    rule(
        gram('ADJF'),
        gram('ADJF').repeatable(),
        gram('NOUN')
    ),
    rule(
        gram('NOUN').repeatable(),
        gram('ADJF'),
        gram('NOUN').repeatable()
    ),
    rule(
        gram('NOUN').repeatable()
    )
)

Item = fact(
    'Item',
    [attribute('titles').repeatable()]
)


IGNORE = rule(
    '(',
    not_(eq(')')).repeatable(),
    ')'
)

ITEM = rule(
    IGNORE.interpretation(
        Item.titles
    ),
    eq(',').optional()
).repeatable().interpretation(
    Item
)

def get_hyperonyms(main_word):
    HYPONYM = eq(utils.deaccent(main_word))
    RULE = or_(
        rule(HYPONYM, ATAKJE, START, MID, END),
        rule(HYPONYM, MID, END),
        rule(START_S, END, KAK, HYPONYM),
        rule(END, INCLUDING, HYPONYM)
    )
    parser = Parser(RULE)
    text = utils.deaccent(wikipedia.summary(main_word))
    print(text)
    text = re.sub(r'\(.+?\)', '', text)
    text = text.lower().replace('* сергии радонежскии* ', '')
    for idx, match in enumerate(parser.findall(text.lower())):
        k = [_.value for _ in match.tokens]
        print(k)

get_hyperonyms("банан")

homework_data = pd.read_excel('Hyperonyms .xlsx', sheet_name='Лист1')

print(homework_data.head())


print('ОКАПЫВАНИЕ')
print('\n')
get_hyperonyms('ОКАПЫВАНИЕ')
print('\n\n')
for word in list(homework_data['ОКАПЫВАНИЕ']):
    print(word)
    print('\n')
    get_hyperonyms(word)
    print('\n\n')
import pandas as pd

CHARGE_NUMBER = 17

df = pd.read_csv("/content/gdrive/My Drive/lab2_words.csv")
charge = df.loc[df['IN CHARGE NUMBER'] == CHARGE_NUMBER]
charge.head(5)

for index, row in charge.iterrows():
    try:
      print(row['HYPONYM'])
      get_hyperonyms(row['HYPONYM'])
      print("-"*50)
    except:
        pass