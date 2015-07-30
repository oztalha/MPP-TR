from __future__ import print_function
import twitter
import sys
import time
from urllib.request import URLError #urllib2
from http.client import BadStatusLine #httplib
from sys import maxsize
from functools import partial

def oauth_login(tw):
    """Twitter authorization
       Expects a dictionary with 4 k,v pairs.
       Returns twitter_api handle
    """
    auth = twitter.oauth.OAuth(tw['OAUTH_TOKEN'], tw['OAUTH_TOKEN_SECRET'],
            tw['CONSUMER_KEY'], tw['CONSUMER_SECRET'])
    twitter_api = twitter.Twitter(auth=auth)
    return twitter_api
    

def make_twitter_request(twitter_api_func, max_errors=10, *args, **kw): 
    
    # A nested helper function that handles common HTTPErrors. Return an updated
    # value for wait_period if the problem is a 500 level error. Block until the
    # rate limit is reset if it's a rate limiting issue (429 error). Returns None
    # for 401 and 404 errors, which requires special handling by the caller.
    def handle_twitter_http_error(e, wait_period=2, sleep_when_rate_limited=True):
    
        if wait_period > 3600: # Seconds
            print ('Too many retries. Quitting.',file=sys.stderr)
            raise e
    
        # See https://dev.twitter.com/docs/error-codes-responses for common codes
    
        if e.e.code == 401:
            print ('Encountered 401 Error (Not Authorized)',file=sys.stderr)
            return None
        elif e.e.code == 404:
            print ('Encountered 404 Error (Not Found)',file=sys.stderr)
            return None
        elif e.e.code == 429: 
            print ('Encountered 429 Error (Rate Limit Exceeded)',file=sys.stderr)
            if sleep_when_rate_limited:
                print ("Retrying in 15 minutes...ZzZ...",file=sys.stderr)
                sys.stderr.flush()
                time.sleep(60*15 + 5)
                print ('...ZzZ...Awake now and trying again.',file=sys.stderr)
                return 2
            else:
                raise e # Caller must handle the rate limiting issue
        elif e.e.code in (500, 502, 503, 504):
            print ('Encountered',e.e.code,'Error. Retrying in',wait_period,'seconds',file=sys.stderr)
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
        except twitter.api.TwitterHTTPError as e:
            error_count = 0 
            wait_period = handle_twitter_http_error(e, wait_period)
            if wait_period is None:
                return
        except URLError as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print ("URLError encountered. Continuing.",file = sys.stderr)
            if error_count > max_errors:
                print ("Too many consecutive errors...bailing out.",file=sys.stderr)
                raise
        except BadStatusLine as e:
            error_count += 1
            time.sleep(wait_period)
            wait_period *= 1.5
            print ("BadStatusLine encountered. Continuing.",file=sys.stderr)
            if error_count > max_errors:
                print ("Too many consecutive errors...bailing out.",file=sys.stderr)
                raise



def get_followers_ids(twitter_api, screen_name=None, user_id=None,limit=maxsize):
    # Must have either screen_name or user_id (logical xor)
    assert (screen_name != None) != (user_id != None), \
    "Must have screen_name or user_id, but not both"
    
    get_followers_ids = partial(make_twitter_request,twitter_api.followers.ids, count=5000)
    ids = []
    label = "followers"

    cursor = -1
    while cursor != 0:
        # Use make_twitter_request via the partially bound callable...
        if screen_name:
            response = get_followers_ids(screen_name=screen_name, cursor=cursor)
        else: # user_id
            response = get_followers_ids(user_id=user_id, cursor=cursor)

        if response is not None:
            ids += response['ids']
            cursor = response['next_cursor']
        
        print('Fetched {0} total {1} ids for {2}. next_cursor: {3}'.format(
            len(ids), label, (user_id or screen_name), cursor))
        
        # XXX: You may want to store data during each iteration to provide an 
        # an additional layer of protection from exceptional circumstances
        
        if len(ids) >= limit or response is None:
            break

    # Do something useful with the IDs, like store them to disk...
    return ids[:limit]