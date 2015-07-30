"""
Collect and analyze Twitter followers of the traditional media in Turkey
"""

import twitter
from functools import partial
import io, json
import pymongo
from sys import maxint
import sys
import time
import datetime
from urllib2 import URLError
from httplib import BadStatusLine
import operator
from itertools import combinations
import csv
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#import pylab as pl
from operator import itemgetter
import networkx as nx
from networkx.readwrite import json_graph
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import matplotlib.cm as cm
from collections import defaultdict
import math
#from numpy import genfromtxt
#from matplotlib.ticker import MaxNLocator

def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw): 
	
	# A nested helper function that handles common HTTPErrors. Return an updated
	# value for wait_period if the problem is a 500 level error. Block until the
	# rate limit is reset if it's a rate limiting issue (429 error). Returns None
	# for 401 and 404 errors, which requires special handling by the caller.
	def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
	
		if wait_period > 3600: # Seconds
			print >> sys.stderr, 'Too many retries. Quitting.'
			raise e
	
		# See https://dev.twitter.com/docs/error-codes-responses for common codes
	
		if e.e.code == 401:
			print >> sys.stderr, 'Encountered 401 Error (Not Authorized)'
			return None
		elif e.e.code == 404:
			print >> sys.stderr, 'Encountered 404 Error (Not Found)'
			return None
		elif e.e.code == 429: 
			print >> sys.stderr, 'Encountered 429 Error (Rate Limit Exceeded)'
			if sleep_when_rate_limited:
				print >> sys.stderr, "Retrying in 15 minutes...ZzZ..."
				sys.stderr.flush()
				time.sleep(60*15 + 5)
				print >> sys.stderr, '...ZzZ...Awake now and trying again.'
				return 2
			else:
				raise e # Caller must handle the rate limiting issue
		elif e.e.code in (500, 502, 503, 504):
			print >> sys.stderr, 'Encountered %i Error. Retrying in %i seconds' % \
				(e.e.code, wait_period)
			time.sleep(wait_period)
			wait_period *= 1.5
			return wait_period
		else:
			raise e

	# End of nested helper function
	
	wait_period = 2 
	error_count = 0 

	while True:
		try:
			return twitter_api_func(*args, **kw)
		except twitter.api.TwitterHTTPError, e:
			error_count = 0 
			wait_period = handle_twitter_http_error(e, wait_period)
			if wait_period is None:
				return
		except URLError, e:
			error_count += 1
			time.sleep(wait_period)
			wait_period *= 1.5
			print >> sys.stderr, "URLError encountered. Continuing."
			if error_count > max_errors:
				print >> sys.stderr, "Too many consecutive errors...bailing out."
				raise
		except BadStatusLine, e:
			error_count += 1
			time.sleep(wait_period)
			wait_period *= 1.5
			print >> sys.stderr, "BadStatusLine encountered. Continuing."
			if error_count > max_errors:
				print >> sys.stderr, "Too many consecutive errors...bailing out."
				raise

def get_friends_followers_ids(twitter_api, screen_name=None, user_id=None,
							  friends_limit=maxint, followers_limit=maxint):
	
	# Must have either screen_name or user_id (logical xor)
	assert (screen_name != None) != (user_id != None), \
	"Must have screen_name or user_id, but not both"
	
	# See https://dev.twitter.com/docs/api/1.1/get/friends/ids and
	# https://dev.twitter.com/docs/api/1.1/get/followers/ids for details
	# on API parameters
	
	# count is the number of users to return per page, up to a maximum of 200. Defaults to 20.
	get_friends_ids = partial(make_twitter_request, twitter_api.friends.ids, 
							  count=5000)
	get_followers_ids = partial(make_twitter_request, twitter_api.followers.ids, 
								count=5000)

	friends_ids, followers_ids = [], []
	
	for twitter_api_func, limit, ids, label in [
					[get_friends_ids, friends_limit, friends_ids, "friends"], 
					[get_followers_ids, followers_limit, followers_ids, "followers"]
				]:
		
		if limit == 0: continue
		
		cursor = -1
		while cursor != 0:
		
			# Use make_twitter_request via the partially bound callable...
			if screen_name: 
				response = twitter_api_func(screen_name=screen_name, cursor=cursor)
			else: # user_id
				response = twitter_api_func(user_id=user_id, cursor=cursor)

			if response is not None:
				ids += response['ids']
				cursor = response['next_cursor']
		
			print >> sys.stderr, 'Fetched {0} total {1} ids for {2}. next_cursor: {3}'.format(len(ids), 
													label, (user_id or screen_name), cursor)
		
			# XXX: You may want to store data during each iteration to provide an 
			# an additional layer of protection from exceptional circumstances
		
			if len(ids) >= limit or response is None:
				break

	# Do something useful with the IDs, like store them to disk...
	return friends_ids[:friends_limit], followers_ids[:followers_limit]




def save_json(filename, data):
	with io.open('{0}.json'.format(filename), 'w', encoding='utf-8') as f:
		f.write(unicode(json.dumps(data, ensure_ascii=False)))

def load_json(filename,encoding='utf-8'):
	with open(filename) as infile:
		data = json.load(infile)
	return data
	# with io.open('{0}'.format(filename), encoding=encoding) as f:
	#	return f.read()


def oauth_login():    
	CONSUMER_KEY = ''
	CONSUMER_SECRET =''
	OAUTH_TOKEN = ''
	OAUTH_TOKEN_SECRET = ''
	auth = twitter.oauth.OAuth(OAUTH_TOKEN, OAUTH_TOKEN_SECRET,CONSUMER_KEY, CONSUMER_SECRET)
	twitter_api = twitter.Twitter(auth=auth)
	return twitter_api

def getData(screen_names):
	twitter_api = oauth_login()
	res = defaultdict(dict)
	#mymedia = set('zamancomtr postacomtr SozcuGazetesi turkiyegazetesi bugun emekevrenseldir cnnturkcom trthaber dhainternet ntv milliyet Sabah Vatan BirGun_Gazetesi ulusalkanal yurtgazetesi'.split())
	#mymedia = set('yenisafak aksam_gazetesi Tarafinternet Halk_TV t24comtr Haber7 tvahaber HaberMynet zaytung Cihan_Haber ihacomtr Haberturk radikal bbcturkce medyatavacom Hurriyet'.split())
	for name in screen_names:
		friends_ids, followers_ids = get_friends_followers_ids(twitter_api, screen_name=name)#, friends_limit=1000000, followers_limit=1000000
		res[name].update({'friends':friends_ids[:]})
		res[name].update({'followers':followers_ids[:]})
	print 'elh...'
	return res


def remap_keys(mapping):
	return [{'key':k, 'value': v} for k, v in mapping.iteritems()]

def doublePairwiseSimilarity(medsim,filename='pairwiseSim.csv'):
	#fp = open(filename, 'wb')
	#csvfile = csv.writer(fp, delimiter=',')
	pairs = medsim.keys()
	for j,pair in enumerate(pairs):
		sim = medsim[(pair[0],pair[1])]
		print j,
		print pair[0],
		print pair[1],
		print sim
		#csvfile.writerow([pair[0][0],pair[1][0],sim])
		medsim.update({(pair[1],pair[0]):sim})

	#fp.close()
	#sorted_medsim = sorted(medsim.iteritems(), key=operator.itemgetter(1))
	return medsim

def writePairwiseSimilarity(medsim,filename='pairwiseSim.csv'):
	fp = open(filename, 'wb')
	csvfile = csv.writer(fp, delimiter=',')
	pairs = medsim.keys()
	for j,pair in enumerate(pairs):
		sim = medsim[(pair[0],pair[1])]
		print j,
		print pair[0],
		print pair[1],
		print sim
		csvfile.writerow([pair[0],pair[1],sim])
	fp.close()
	#sorted_medsim = sorted(medsim.iteritems(), key=operator.itemgetter(1))

def getPairwiseSimilarity(data,filename='pairwiseSim.csv'):
	"""
	returns similarity as a dict where key is a tuple and value is their similarity
	"""
	#fp = open(filename, 'wb')
	#csvfile = csv.writer(fp, delimiter=',')
	medsim={}
	pairs = combinations(data.keys(),2)
	for j,pair in enumerate(pairs):
		sim = follower_similarity(set(data[pair[0]]),set(data[pair[1]]))
		print j,
		print pair[0],
		print pair[1],
		print sim
		#csvfile.writerow([pair[0][0],pair[1][0],sim])
		medsim.update({(pair[0],pair[1]):sim})

	#fp.close()
	#sorted_medsim = sorted(medsim.iteritems(), key=operator.itemgetter(1))
	return medsim

def groups_media(group,media):
	medsim={}
	g=set()
	for v in group.itervalues():
		g = g | set(v['followers'])

	for m in media:
		if m['key'][1] == u'followers':
			sim = follower_similarity(g,set(m['value']))
			medsim.update({m['key'][0]:sim})
	return medsim

def follower_similarity(m1,m2,metric='meetmin'):
	"""
	parameters
	-------------------
	two sets, similarity metric
	-------------------
	"""
	if metric == 'meetmin':
		return 100.0*len(m1.intersection(m2))/min(len(m1),len(m2))
		
def saveCellectedData(data,filename='trmedia.json'):
	with open(filename, 'wb') as fp:
		json.dump(remap_keys(data), fp)
"""
def saveMedia(data,filename='trmedialist.json'):
	with open(filename, 'wb') as fp:
		json.dump(data, fp)

def loadData(filename='trmedia.json'):
	with open(filename) as infile:
		data = json.load(infile)
	return data
"""
def getSimMatrix(medsim,media):
	sim = np.zeros([len(media),len(media)],dtype=float)
	for i in range(len(media)):
		a=[]
		for j in range(len(media)):
			if i==j:
				a.append(100.0)
			else:
				a.append(medsim[(media[i],media[j])])
		sim[i] = np.array(a)
	return sim

def drawNeato(sim,media):
	# http://www.graphviz.org/pdf/neatoguide.pdf
	G = nx.from_numpy_matrix(sim)
	G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),media)))
	G = nx.to_agraph(G)
	G.node_attr.update(shape="plaintext",width=.5)
	G.graph_attr.update(splines=None,overlap="scale")
	G.draw('out.png', format='png', prog='neato',args='-Gepsilon=0.001')

def saveGraphAsJson(sim,media,filename='g.json'):
	#http://www.slideshare.net/fullscreen/arnicas/a-quick-and-dirty-intro-to-networkx-and-d3/3
	G = nx.from_numpy_matrix(sim)
	G = nx.relabel_nodes(G, dict(zip(range(len(G.nodes())),media)))
	g_json = json_graph.node_link_data(G)
	g_json['nodes'] = [{'name':n['id']} for n in g_json['nodes']]
	json.dump(g_json,open(filename,'w'))

def plotBarchart(data,filename="followercounts.png"):
	plt.close('all')
	xy = [(k,len(v)) for k,v in data.iteritems()]
	xy.sort(key=itemgetter(1))
	media = [k[0] for k in xy]
	followers = [k[1] for k in xy]
	#media = data.keys() #media names
	#followers = [len(k) for k in data.values()] #follower counts
	numBars = len(media)
	fig, ax1 = plt.subplots(figsize=(9, 13))
	pos = np.arange(numBars)+0.5    # Center bars on the Y-axis ticks

	#pylab.yticks(pos, media)
	rects = ax1.barh(pos, followers, align='center', height=0.9, color='m')

	# Lastly, write in the ranking inside each bar to aid in interpretation
	for i,rect in enumerate(rects):
		# Rectangle widths are already integer-valued but are floating
		# type, so it helps to remove the trailing decimal point and 0 by
		# converting width to int type
		width = int(rect.get_width())
		if followers[i]>1000000:
			inbarStr = media[i]+" ("+ str(followers[i]/1000000)+"."+ str((followers[i]%1000000)/10000)+"M"
		else:
			inbarStr = media[i]+" ("+ str(followers[i]/1000)+"K"
		inbarStr +=")"

		if (width < 666666):    # The bars aren't wide enough to print the ranking inside
			xloc = width + 9000   # Shift the text to the right side of the right edge
			clr = 'black'      # Black against white background
			align = 'left'
		else:
			xloc = 0.98*width  # Shift the text to the left side of the right edge
			clr = 'white'      # White on magenta
			align = 'right'

		# Center the text vertically in the bar
		yloc = rect.get_y() + rect.get_height()/2.0
		ax1.text(xloc, yloc, inbarStr, horizontalalignment=align,
			verticalalignment='center', color=clr, weight='bold')
	plt.axis('tight')
	plt.axis('off')
	plt.savefig(filename,dpi=600,bbox_inches='tight',facecolor='white', edgecolor='none')
	plt.show()



def plotPopularity(data):
	followercounts = {}
	[followercounts.update({k:len(v)}) for k,v in data.iteritems()]
	# x axis : xy[:,0], y axis : xy[:,1]
	xy = [(k[0],int(k[1])) for k in followercounts.iteritems()]
	#xy.sort(key=lambda tup: tup[1])
	xy.sort(key=itemgetter(1))
	media = [k[0] for k in xy]
	followers = [k[1] for k in xy]
	numBars = len(media)

	fig, ax1 = plt.subplots(figsize=(9, 13))
	plt.subplots_adjust(left=0.115, right=0.88)
	pos = np.arange(numBars)+0.5    # Center bars on the Y-axis ticks
	#pylab.yticks(pos, media)
	rects = ax1.barh(pos, followers, align='center', height=0.5, color='m')

	# Lastly, write in the ranking inside each bar to aid in interpretation
	for i,rect in enumerate(rects):
		# Rectangle widths are already integer-valued but are floating
		# type, so it helps to remove the trailing decimal point and 0 by
		# converting width to int type
		width = int(rect.get_width())
		inbarStr = media[i]
		if (width < 333333):    # The bars aren't wide enough to print the ranking inside
			xloc = width + 9000   # Shift the text to the right side of the right edge
			clr = 'black'      # Black against white background
			align = 'left'
		else:
			xloc = 0.98*width  # Shift the text to the left side of the right edge
			clr = 'white'      # White on magenta
			align = 'right'

		# Center the text vertically in the bar
		yloc = rect.get_y()+rect.get_height()/2.0
		ax1.text(xloc, yloc, inbarStr, horizontalalignment=align,
			verticalalignment='center', color=clr, weight='bold')
	#plt.yscale('log')
	plt.show()


def load_Similarity(filename='pairwiseSim.csv'):
	medsim = {}
	with open(filename, 'rU') as csvfile:
		simreader = csv.reader(csvfile)
		simreader.next() #skip the header line
		for row in simreader:
			medsim.update({(row[0],row[1]):float(row[2])})
	return medsim

def invertSim(sim):
	#multiplicative inversion did not yield nice results
	#sim = 1/sim
	np.fill_diagonal(sim,0)
	sim = sim.max() - sim + 5
	return sim

def saveScattered(labels,sim,filename='a.png',metric=True,method='MDS',transparent='True',fontsize=12):
	plt.close('all')
	fig = plt.figure(figsize=(10, 8))#figsize=(30, 30)
	plt.axis('off')
	#axes = fig.add_subplot(1, 1, 1, axisbg='black')
	dissim = 100 - sim #invertSim(sim)
	if method=='MDS':
		seed = 5
		mds = MDS(n_components=2, metric=metric,max_iter=1000,random_state=seed,dissimilarity='precomputed')
		data = mds.fit_transform(dissim.astype(np.float64))
	if method=='PCA':
		data = PCA(n_components=2).fit_transform(sim)

	data = data + np.abs(data.min())
	plt.scatter(data[:, 0], data[:, 1], s = 1, marker = 'o')
	
	a = load_json('g.json')['nodes']
	#http://wiki.scipy.org/Cookbook/Matplotlib/Show_colormaps
	#colors = cm.rainbow(np.linspace(0, 1, 11))
	colors = cm.Spectral(np.linspace(0, 1, 11))
	b = {}
	#[b.update({a[i]['name']:colors[a[i]['group']]}) for i in range(len(a))]
	[b.update({a[i]['name']:colors[int(data[i,0]/6) %11]}) for i in range(len(a))]

	for label, x, y in zip(labels, data[:, 0], data[:, 1]):
		"""
		if label=='ihacomtr' or label=='Haberturk':
			plt.annotate(label, xy = (x, y), fontsize=fontsize, ha = 'right',weight='bold',color = b[label])
		elif label=='yenisafak' or label== 'cnnturkcom':
			plt.annotate(label, xy = (x, y), fontsize=fontsize, va = 'top',weight='bold',color = b[label])
		else:
			plt.annotate(label, xy = (x, y), fontsize=fontsize, ha = 'center',weight='bold',color = b[label]) # str(x)+","+str(y)
		"""
		plt.annotate(label, xy = (x, y), fontsize=fontsize, ha = 'center',weight='bold') # str(x)+","+str(y)
		#xytext = (-20, 20),
		#textcoords = 'offset points', ha = 'right', va = 'bottom',
		#bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
		#arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0')
	plt.axis('tight')
	plt.savefig(filename,dpi=600,transparent=transparent,bbox_inches='tight', edgecolor='none')#,facecolor='black'

def demap_keys(data):
	followers = defaultdict(set)
	[followers.update({k['key'][0]:set(k['value'])}) for k in data if k['key'][1] == u'followers']
	return followers

def createGroupMedia(filenames):
	data = []
	for f in filenames:
		data.append(load_json(f+".json"))
	fieldnames = data[0].keys()
	test_file = open('readme-states-age.csv','wb')
	csvwriter = csv.DictWriter(test_file, delimiter=',', fieldnames=fieldnames)
	csvwriter.writerow(dict((fn,fn) for fn in fieldnames))
	for row in data:
		csvwriter.writerow(row)
	test_file.close()


if __name__ == '__main__':
	# data = getData()
	# data = saveCellectedData(data)
	# data = load_json('trmedia.json')
	# plotPopularity(data)
	# medsim = getPairwiseSimilarity(data)
	medsim = load_Similarity()
	media = load_json('trmedialist.json')
	sim = getSimMatrix(medsim,media)
	# saveGraphAsJson(sim,media)
	# saveScattered(media,sim,'a.png')