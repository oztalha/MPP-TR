import pandas as pd
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['savefig.dpi'] = 150
from itertools import combinations
from collections import defaultdict
import seaborn as sns
import networkx as nx

#read outlets data
with open('data/outlets/outlets_list.json') as f:
    outlets = json.load(f)

m = {} #2013-12 and 2015-07 datasets
for k,vl in outlets.items():
    m[k] = {} #set of subscribers for each news media
    for v in vl:
        fname = 'data/outlets/'+k+'/'+v+'.json'
        with open(fname) as f:
            m[k][v] = set(json.load(f))
            
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