import googletrans
import streamlit
from googletrans import Translator
import tweepy
from textblob import TextBlob
from wordcloud import WordCloud
import pandas as pd
import re
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
translator = Translator()
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import streamlit as st
from PIL import Image
import seaborn as sns
import numpy as np

@st.cache(allow_output_mutation=True)
def get_model():
    roberta = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
    model = AutoModelForSequenceClassification.from_pretrained(roberta)
    tokenizer = AutoTokenizer.from_pretrained(roberta)
    return tokenizer, model

tokenizer, model = get_model()

consumerKey = "SCXJOdWHrc9zWEFHf1Ha7kmc1"
consumerSecret = "72QzzKBXBTCFb0Po2Np4yNUrZ7yEHImlfxLyQxcCL3QV8S3CPN"
accessToken = "1270624123329208320-g6dwbbOVoEPAqDJRpJrfg1smboKJ33"
accessTokenSecret = "s2ee7pvNL7XUePYEilI30hA3izU6YLj3pDSSWXYqZHOZ8"

# Create the authentication object
authenticate = tweepy.OAuthHandler(consumerKey, consumerSecret)

# Set the access token and access token secret
authenticate.set_access_token(accessToken, accessTokenSecret)

# Creating the API object while passing in auth information
api = tweepy.API(authenticate, wait_on_rate_limit=True)


def app():
    st.title("Tweet Analyzer 🤓")

    activities = ["Home","Tweet Analyze","Generate Twitter Data"] #Generate Twitter Data

    choice = st.sidebar.selectbox("Select Your Activity", activities)
    if choice=="Home":
        st.markdown("Please select an option from ***Select your Activity*** drop down menu")


    elif choice=="Tweet Analyze":
        #st.subheader("Analyze the tweets of your favourite Personalities")
        st.markdown("Welcome to Twitter Sentiment Analyzer. Here you get to perform the following functions :")
        st.write("1. Fetch recent Tweets from a particular user")
        st.write("2. Generates  Word Cloud")
        st.write("3. Displays Sentiments using a bar plot")
        st.markdown("*Enter the user name and the number of tweets you wish to fetch in the fields below*")

        twitter_user = st.text_input("*Enter User ID of the Twitter handler without '@' in the empty field below*")
        #limit_of_tweets = int(st.text_input("*How many recent Tweets you want to fetch?*"))
        limit_of_tweets = st.text_input("*How many recent Tweets you want to fetch?*") # Passing the value as string. This value will be converted to Int below
        Analyzer_choice = st.selectbox("Select the Activities",["Show Recent Tweets", "Generate WordCloud", "Visualize the Sentiment Analysis"])

        if st.button("Analyze Tweet"):
            # SHow Recent Tweets
            if Analyzer_choice=='Show Recent Tweets':
                st.success("Fetching the last %d Tweets by the user, please wait..."%(int(limit_of_tweets)))

                def show_recent_tweets(twitter_user):

                    posts = api.user_timeline(screen_name=twitter_user, count=limit_of_tweets, lang="en", tweet_mode="extended")

                    def get_tweets():
                        print("Show the 5 recent tweets : \n")
                        l=[]
                        i=1
                        for tweet in posts[0:int(limit_of_tweets)]:
                            l.append(tweet.full_text)
                            i= i + 1
                        return l
                    recent_tweets=get_tweets()
                    return recent_tweets
                recent_tweets=show_recent_tweets(twitter_user)
                st.write(recent_tweets)

            # Generating Wordcloud
            elif Analyzer_choice=='Generate WordCloud':
                #st.success("Fetching the last %d Tweets by the user, please wait..."%(limit_of_tweets))
                st.success("Generating Word Cloud of recent %d Tweets as well as Translating any Non-English Tweets to English, please wait..."%(int(limit_of_tweets)))
                def get_wordcloud():
                    tweets = tweepy.Cursor(api.user_timeline, screen_name=twitter_user, tweet_mode='extended').items(int(limit_of_tweets))
                    columns = ['Tweets']  # ['Time', 'User', 'Tweet']
                    data = []
                    for tweet in tweets:
                        data.append([tweet.full_text])

                    df = pd.DataFrame(data, columns=columns)

                    #posts = api.user_timeline(screen_name=twitter_user, count=200, lang="en", tweet_mode="extended")

                    # Creating Dataframe of the tweets
                    #df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
                    tweet_list = []
                    for i in range(len(df.Tweets)):
                        result = translator.translate(df.Tweets[i], dest="en")
                        tweet_list.append(result.text)
                    df1 = pd.DataFrame(tweet_list, columns=['Translated_Tweets'])

                    def preprocess(text):
                        new_text = []
                        for t in text.split(" "):
                            t = '@user' if t.startswith('@') and len(t) > 1 else t
                            t = 'http' if t.startswith('http') else t
                            new_text.append(t)
                        return " ".join(new_text)

                    df1['Translated_Tweets'] = df1['Translated_Tweets'].apply(preprocess)

                    def cleantext(text):
                        text = re.sub(r'RT[\s]+', '', text)  # removes RT
                        return text

                    df1['Translated_Tweets'] = df1['Translated_Tweets'].apply(cleantext)

                    # word cloud visualization
                    allWords = ' '.join([twts for twts in df1['Translated_Tweets']])
                    wordcloud = WordCloud(width=500, height=300,random_state=30, max_font_size=119,min_font_size = 10).generate(allWords)
                    plt.figure(figsize=(13, 13), facecolor=None)
                    #plt.imshow(wordcloud, interpolation="bilinear")
                    plt.imshow(wordcloud)
                    plt.axis('off')
                    plt.tight_layout(pad=0)
                    plt.savefig('WC.jpg')
                    img = Image.open("WC.jpg")
                    return img

                img = get_wordcloud()
                st.image(img)

                #You can convert your PIL Image to bytes using the following snippet
                from io import BytesIO
                buf = BytesIO()
                img.save(buf, format="JPEG")
                byte_im = buf.getvalue()

                btn = st.download_button(
                    label="Download Image",data=byte_im,file_name="imagename.png",mime="image/jpeg")


            else:

                def plot_analysis():
                    st.success("Generating visualization for Sentiment Analysis of recent %d tweets, also Translating Non-English Tweets to English, please wait..."%(int(limit_of_tweets)))
                    #posts = api.user_timeline(screen_name=twitter_user, count=500, lang="en", tweet_mode="extended")
                    #limit = 500
                    tweets = tweepy.Cursor(api.user_timeline, screen_name=twitter_user, tweet_mode='extended').items(int(limit_of_tweets))
                    columns = ['Tweets']  # ['Time', 'User', 'Tweet']
                    data = []
                    for tweet in tweets:
                        data.append([tweet.full_text])

                    df = pd.DataFrame(data, columns=columns)
                    # Creating Dataframe of the tweets
                    #df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
                    #Translating tweets
                    tweet_list = []
                    for i in range(len(df.Tweets)):
                        result = translator.translate(df.Tweets[i], dest="en")
                        tweet_list.append(result.text)
                    df1 = pd.DataFrame(tweet_list, columns=['Translated_Tweets'])

                    def preprocess(text):
                        new_text = []
                        for t in text.split(" "):
                            t = '@user' if t.startswith('@') and len(t) > 1 else t
                            t = 'http' if t.startswith('http') else t
                            new_text.append(t)
                        return " ".join(new_text)

                    df1['Translated_Tweets'] = df1['Translated_Tweets'].apply(preprocess)

                    def cleantext(text):
                        text = re.sub(r'RT[\s]+', '', text)  # removes RT
                        return text

                    df1['Translated_Tweets'] = df1['Translated_Tweets'].apply(cleantext)

                    def sentiment_score(review):
                        tokens = tokenizer.encode(review, return_tensors='pt')
                        result = model(tokens)
                        return int(torch.argmax(result.logits))
                    df1['sentiment'] = df1['Translated_Tweets'].apply(lambda x: sentiment_score(x[:512]))

                    def Getanalysis(score):
                        if score == 0:
                            return 'Negative'
                        elif score == 1:
                            return 'Neutral'
                        elif score == 2:
                            return 'Positive'
                    df1['Analysis'] = df1['sentiment'].apply(Getanalysis)

                    return df1

                df1= plot_analysis()
                st.write(sns.countplot(x=df1["Analysis"], data=df1))
                st.set_option('deprecation.showPyplotGlobalUse', False)
                st.pyplot(use_container_width=True)

                #You can convert your PIL Image to bytes using the following snippet
                #from io import BytesIO
                #buf = BytesIO()
                #st.write(sns.countplot(x=df1["Analysis"], data=df1)).save(buf, format="JPEG")
                #byte_im = buf.getvalue()

                #btn = st.download_button(label="Download Image",data=st.pyplot(use_container_width=True),file_name="imagename.png",mime="image/jpeg")



    else:
        st.markdown("**Here we fetch the lastest tweets and can perform the following tasks on them**")

        st.write("1. Preprocess, Clean and Translate Tweets to English Language")
        st.write("2. Analyze the Tweets using Roberta model and label the tweets as Positive, Negative or Neutral")
        st.write("3. Display the data as a Dataframe")
        twitter_id = st.text_input("*Enter User ID of the Twitter handler without '@' in the empty field below*")
        limit_of_tweets = st.text_input("*How many recent Tweets you want to fetch?*")

        def get_data(twitter_id):
            #posts = api.user_timeline(screen_name=twitter_id, count=100, lang="en", tweet_mode="extended")
            #df = pd.DataFrame([tweet.full_text for tweet in posts], columns=['Tweets'])
            #df['Translated_tweets']=df['Tweets']
            tweets = tweepy.Cursor(api.user_timeline, screen_name=twitter_id, tweet_mode='extended').items(int(limit_of_tweets))
            columns = ['Tweets']  # ['Time', 'User', 'Tweet']
            data = []
            for tweet in tweets:
                data.append([tweet.full_text])

            df = pd.DataFrame(data, columns=columns)

            tweet_list = []
            for i in range(len(df.Tweets)):
                result = translator.translate(df.Tweets[i], dest="en")
                tweet_list.append(result.text)
            df1 = pd.DataFrame(tweet_list, columns=['Translated_Tweets'])

            def preprocess(text):
                new_text = []
                for t in text.split(" "):
                    t = '@user' if t.startswith('@') and len(t) > 1 else t
                    t = 'http' if t.startswith('http') else t
                    new_text.append(t)
                return " ".join(new_text)

            df1['Translated_Tweets'] = df1['Translated_Tweets'].apply(preprocess)

            def sentiment_score(review):
                tokens = tokenizer.encode(review, return_tensors='pt')
                result = model(tokens)
                return int(torch.argmax(result.logits))

            df1['sentiment'] = df1['Translated_Tweets'].apply(lambda x: sentiment_score(x[:512]))

            def Getanalysis(score):
                if score == 0:
                    return 'Negative'
                elif score == 1:
                    return 'Neutral'
                elif score == 2:
                    return 'Positive'

            df1['Analysis'] = df1['sentiment'].apply(Getanalysis)

            return df1
        if st.button('Show Data'):
            st.success("Feteching recent %d tweets, Please wait..."%(int(limit_of_tweets)))
            df1=get_data(twitter_id)
            st.write(df1)

            @st.cache
            def convert_df(dataframe):
                return dataframe.to_csv().encode('utf-8')
            csv = convert_df(df1)
            st.download_button(label="Download CSV",data=csv,file_name='output_df.csv',mime='text/csv')




if __name__ == "__main__":
    app()



