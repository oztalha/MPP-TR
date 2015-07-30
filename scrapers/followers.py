# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 16:54:52 2015

@author: Talha
"""

def saveFollowers(followees,twitter_api,path='data/followers/'):
    """
    Makes Twitter followers ids requests for TBMM deputies
    Saves user IDs into files with the filename of deputies screen names
    """
    from utilities.twhelper import get_followers_ids
    import json
    
    for followee in (followees):
        try:
            followers = get_followers_ids(twitter_api, screen_name=followee)
            with open(path+followee+'.json', 'w') as outfile:
                json.dump(followers, outfile)
        except Exception as e:
            print('[ERROR]: ',followee,e)

    
import pandas as pd   
from auth import keys   
from utilities.twhelper import oauth_login  

twitter_api = oauth_login(keys.user['hesobi'])
df = pd.read_table('data/deputies.csv')
deputies = df.screen_name.tolist()
saveFollowers(deputies,twitter_api,'data/deputies')

twitter_api = oauth_login(keys.user['manqili'])
parties = ['Akparti']
saveFollowers(parties,twitter_api,'data/parties')

media = ['t24comtr','SHaberTV']
twitter_api = oauth_login(keys.user['manqili'])
saveFollowers(media,twitter_api,'data/outlets/2015-07/')