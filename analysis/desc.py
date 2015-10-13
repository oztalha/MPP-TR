import pandas as pd
import json
import matplotlib.pyplot as plt
plt.style.use('ggplot')
plt.rcParams['savefig.dpi'] = 150
from itertools import combinations
from collections import defaultdict
import seaborn as sns


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
            
#get meetmin similarities
s13 = getPairwiseSimilarities(m['2013-12'])
s15 = getPairwiseSimilarities(m['2015-07'])

fig, ax = plt.subplots(figsize=(15,12))
ax = sns.heatmap(s15-s13)
ax.set_title('Change in Audience Similarities from 2013-12 to 2015-07')
fig.savefig('outputs/charts/sim-diff.png',bbox_inches='tight')








def getPairwiseSimilarities(s):
    """
    Expects a media subscribers dictionary s where each key is a media name
    and each value is a set of subscriber ids for that organizations.
    
    Returns meet/min similarity matrix as a dataframe
    """
    
    medsim = defaultdict(dict)
    media = list(s.keys())
    pairs = combinations(s.keys(),2)
    
    for m in media:
        medsim[m][m] = 1
    for m1,m2 in pairs:
        sim = 1.0*len(s[m1].intersection(s[m2]))/min(len(s[m1]),len(s[m2]))
        #medsim.update({(m1,m2):sim})
        medsim[m1][m2] = sim
        medsim[m2][m1] = sim
    df = pd.DataFrame(data=medsim,index=s.keys(),columns=s.keys())
    return df
