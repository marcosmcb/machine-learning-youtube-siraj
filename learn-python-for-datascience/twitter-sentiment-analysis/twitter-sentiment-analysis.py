import tweepy
from textblob import TextBlob

consumer_key = 'c4QOddHyAURXKN0BcKAHwPb9g'
consumer_secret = 'y7OS50WAISgubinJcLyL6WIuNWf3gEZS2zP5M0YcIPRPUu8lOf'

access_token = '342526912-9sEzcdpakIUyRCFtXPNb9nKByfEV9Q4K5B7rJwSV'
access_token_secret = 'TKcmZKHAARiHHecpHnCaIWg3zRdij67JBHqLcayGVcfPu'


auth = tweepy.OAuthHandler( consumer_key, consumer_secret )
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)

public_tweets = api.search('Formula 1')

for tweet in public_tweets:
    print( tweet.text )
    analysis = TextBlob( tweet.text )
    print( analysis.sentiment )