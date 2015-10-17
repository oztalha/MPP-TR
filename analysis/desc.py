import pandas as pd
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot') #print (plt.style.available)
plt.rcParams['savefig.dpi'] = 150
from itertools import combinations
from collections import defaultdict
from collections import Counter
import seaborn as sns
import networkx as nx


def read_outlets():
    """ read outlets info in json files """
    tip = 'outlets'
    #read data
    with open('data/'+tip+'/list.json') as f:
        names = json.load(f)

    m = {} #2013-12 and 2015-07 datasets        
    for k,vl in names.items():
        m[k] = {} #set of subscribers for each account
        for v in vl:
            fname = 'data/'+tip+'/'+k+'/'+v+'.json'
            with open(fname) as f:
                m[k][v] = set(json.load(f))
    return m


def read_parties():
    """ read parties info in json files """
    tip = 'parties'
    #read data
    with open('data/'+tip+'/list.json') as f:
        names = json.load(f)

    p = {} #2013-12 and 2015-07 datasets        
    for k,vl in names.items():
        p[k] = {} #set of subscribers for each account
        for vp,ac in vl.items():
            p[k][vp] = set()
            for v in ac:
                fname = 'data/'+tip+'/'+k+'/'+v+'.json'
                with open(fname) as f:
                    p[k][vp] |= set(json.load(f))
    return p



m = read_outlets()
ms = m['2015-07']
p = read_parties()
ps = dict(p['2015-07'])
del ps['MFG']


def getPoliticalness(ps,ms):
    df = pd.Series(name='Politicalness')
    united = set.union(*ps.values())
    for k,v in ms.items():
        df[k] = len(v & united)/len(v)
    return df

df = getPoliticalness(ps,ms)

c = Counter()

for k,v in ps.items():
    c.update(v)

united = set.union(*ps.values())
def getMediaOverParties(m,c,ps,ms,united):
    mp = {}
    for p in ps.keys():
        mp[p] = sum([1.0/c[v] for v in (ms[m] & ps[p])])/len(ms[m] & united)
    return pd.DataFrame(mp,columns=['AKP','CHP','MHP','HDP'],index=[m])
    
    
    
mps = pd.DataFrame(columns=['AKP','CHP','MHP','HDP'])
for m in ms.keys():
    mps = mps.append(getMediaOverParties(m,c,ps,ms,united))

colors = ['orange','red','green','purple']
mps.index = mps.index.map(str.lower)
mps = mps.sort_index()

mps.transpose().plot(kind='pie',subplots=True,layout=(6,7),legend=False,
              labels=None,figsize=(10,8.5),colors=colors,autopct='%.2f');

f, axes = plt.subplots(7, 6)
for i in range(7):
    for j in range(6):
        if 6*i+j >= len(mps):
            break
        m = mps.ix[6*i+j]
        ax = axes[i][j]
        patches, texts = ax.pie(m,colors=colors)
        ax.set_title(m.name,fontsize=10,y=0.9)
axes[-1, -1].axis('off')
f.set_size_inches(9.5, 10)
plt.legend(patches, ['AKP','CHP','MHP','HDP'],loc='best')
f.suptitle('Party Leanings of News Audience on Twitter (2015-07)',
           x=0.15,y=0.09,horizontalalignment='left',fontsize=15)
f.savefig('outputs/figures/leanings.png', dpi=100,bbox_inches='tight')



enps = mps.apply(lambda x: 1/sum((v/sum(x))**2 for v in x),axis=1)
enps.sort()
enps.plot(kind='barh',title='Effective Number of Parties',figsize=(8,8))


def getAllSimilarities(s,metric='directional'):
    """
    Expects a media subscribers dictionary s where each key is a media name
    and each value is a set of subscriber ids for that organizations.
    
    metric can be 'directional' or 'meet-min'
    
    Returns meet/min similarity matrix as a dataframe
    """
    
    medsim = defaultdict(dict)
    media = sorted(list(s.keys()))
    pairs = combinations(s.keys(),2)
    
    for m in media:
        medsim[m][m] = 1
    for m1,m2 in pairs:
        if metric == 'meet-min':
            sim = 1.0*len(s[m1].intersection(s[m2]))/min(len(s[m1]),len(s[m2]))
            medsim[m1][m2] = sim
            medsim[m2][m1] = sim
        if metric == 'directional':
            intersection = 1.0*len(s[m1].intersection(s[m2]))
            medsim[m1][m2] = intersection/len(s[m1])
            medsim[m2][m1] = intersection/len(s[m2])
    
    df = pd.DataFrame(data=medsim,index=media,columns=media)
    return df
    
# needs to be normalized by the number of followers who follows a second one?
ds13 = getAllSimilarities(m['2013-12'])
ds15 = getAllSimilarities(m['2015-07'])
fig, ax = plt.subplots(figsize=(15,12))
ax = sns.heatmap(ds15-ds13)
ax.set_xlabel('B')
ax.set_ylabel('A')
ax.set_title('The Change in Audience Similarities from 2013-12 to 2015-07\n'\
    '(Audience Similarity: What percent of A is also subscribed to B)\n'\
    '(For SHaber,imc and ozgurgundem only 2015-07 data available)')
fig.savefig('outputs/charts/sim-diff-directional.png',bbox_inches='tight')



df = ds15
edges = df.stack(level=0).reset_index()
edges.columns = ['source','target','weight']
edges.to_csv('data/outputs/networks/edges15.csv',index=False)
G = nx.from_numpy_matrix(df.as_matrix(),create_using=nx.DiGraph())







##get meetmin similarities
#s13 = getAllSimilarities(m['2013-12'],metric='meet-min')
#s15 = getAllSimilarities(m['2015-07'],metric='meet-min')
#
#fig, ax = plt.subplots(figsize=(15,12))
#ax = sns.heatmap(s15)
#ax.set_title('Audience MeetMin Similarities as 2015-07')
#fig.savefig('outputs/charts/sim-2015-meetmin.png',bbox_inches='tight')
#
#fig, ax = plt.subplots(figsize=(15,12))
#ax = sns.heatmap(s15-s13)
#ax.set_title('Change in Audience Similarities from 2013-12 to 2015-07')
#fig.savefig('outputs/charts/sim-diff-meetmin.png',bbox_inches='tight')