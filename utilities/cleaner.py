# -*- coding: utf-8 -*-
"""
Created on Wed Jul 29 16:44:52 2015

@author: Talha
"""

import pandas as pd
import json
import glob
import os

# 2013-12 outlets cleaned ====================================
#read 2013 news data
m13 = json.load(open('data/outlets/2013-12.json'))
for k,v in m13.items():
    m13[k] = v

m13['evrenselgzt'] = m13.pop('emekevrenseldir')
m13['ulusalkanalcom'] = m13.pop('ulusalkanal')
m13['Taraf_Medya'] = m13.pop('Tarafinternet')
m13['cnnturk'] = m13.pop('cnnturkcom')
m13['Aksam'] = m13.pop('aksam_gazetesi')

for m in m13.keys():
    with open('data/outlets/2013-12/'+m+'.json','w') as f:
        json.dump(m13[m],f)


# ==============================================================
# 2015-07 outlets cleaned ======================================
def k(x):
    """ ntv and mongodb issue """
    if(type(x)==int):
        return x
    return int(list(x.values())[0])


path = 'data/outlets/2015-07/'
m15files = glob.glob('data/outlets/2015-07/*')
m15 = {}
for fname in m15files:
    with open(fname) as f:
        m = json.load(f)
        try: #downloaded using dd-css
            m15[m['parameters']['screen_name']] = list(map(k,m['data']['followers']))
        except: #downloaded directly
            outlet = os.path.basename(fname)[:-5]
            m15[outlet] = m

fcnt = {}
for k,v in m15.items():
    fcnt[k]=len(v)
    
sorted(fcnt.items(), key=lambda x: x[1])

m15.pop('emekevrenseldir');
m15.pop('ulusalkanal');
m15.pop('aksam_gazetesi');
m15.pop('Tarafinternet');
m15.pop('cnnturkcom');

for m in m15.keys():
    with open('data/outlets/2015-08/'+m+'.json','w') as f:
        json.dump(m15[m],f)

#==============================================
# outlets meta file created ===================
mlist = {}
mlist['2015-07']=sorted(list(m15.keys()))
mlist['2013-12']=sorted(list(m13.keys()))
with open('data/outlets/outlets_list.json','w') as f:
    json.dump(mlist,f)
#==============================================
p13 = {}
for fname in glob.glob('data/parties/2014-01/*'):
    with open(fname) as f:
        p = json.load(f)
        for k in p.keys():
            p13[k] = p[k]['followers']

fcnt = {}
for k,v in p13.items():
    fcnt[k]=len(v)
sorted(fcnt.items(), key=lambda x: x[1])

for m in p13.keys():
    with open('data/parties/2014-01/'+m+'.json','w') as f:
        json.dump(p13[m],f)
        
p15 = {}
for fname in glob.glob('data/parties/2015-07/*'):
    with open(fname) as f:
        party = os.path.basename(fname)[:-5]
        p15[party] = set(json.load(f))

fcnt = {}
for k,v in p15.items():
    fcnt[k]=len(v)
sorted(fcnt.items(), key=lambda x: x[1])


#==============================================
# parties meta file created ===================
ptw15 = {}
ptw15['AKP'] = ['Akparti','AkTanitimMedya']
ptw15['CHP'] = ['herkesicinCHP']
ptw15['MHP'] = ['MHP_Bilgi','Ulku_Ocaklari']
ptw15['HDP'] = ['HDPgenelmerkezi','HDPonline']
ptw15['MFG'] = ['FGulencomTR','Herkul_Nagme']
ptw13 = {}
ptw13['AKP'] = ['Akparti','AkTanitimMedya','AKKULIS']
ptw13['CHP'] = ['CHP_online','herkesicinCHP']
ptw13['MHP'] = ['MHP_Bilgi','Ulku_Ocaklari']
ptw13['HDP'] = ['BDPgenelmerkez','HDP_Kongre']
ptw13['MFG'] = ['FGulencomTR','Herkul_Nagme']

mlist = {}
mlist['2015-07']={}
mlist['2014-01']={}
groups = 'AKP CHP MHP HDP MFG'.split()
for g in groups:
    mlist['2014-01'][g] = ptw13[g]
    mlist['2015-07'][g] = ptw15[g]
with open('data/parties/groups_list.json','w') as f:
    json.dump(mlist,f)
#==============================================

