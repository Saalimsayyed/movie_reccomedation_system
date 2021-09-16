# movie_reccomedation_system
The purpose of a recommender system is to suggest users something based on their interest or usage history. So next time Amazon suggests you a product, or Netflix recommends you a tv show or medium display a great post on your feed, understand that there is a recommendation system working under the hood.
 
Requirements:
dataset : For extracting data from csv file
NumPy : For fast matrix operations.
pandas : For analysing and getting insights from datasets.
matplotlib : For creating graphs and plots.
seaborn : For enhancing the style of matplotlib plots.
sk-learn: for using a Machine Learning Model


Code
import pandas as pd
import numpy as np

columns_names=['user_id','item_id','rating','timestamp']


df=pd.read_csv('D:\csv\\u.data',sep='\t',names=columns_names)


df.head()


movie_title=pd.read_csv('D:\csv\Movie_Id_Titles.csv')


movie_title.head()


data=pd.merge(df,movie_title,on='item_id')
data.head()


data.groupby('title')['rating'].mean().sort_values(ascending=False).head()
data.groupby('title')['rating'].count().sort_values(ascending=False).head()


ratings=pd.DataFrame(data.groupby('title')['rating'].mean())
ratings.head()


ratings['num of ratings']=data.groupby('title')['rating'].count()


ratings['num of ratings'].hist(bins=70)

ratings["num of ratings"]


ratings['rating'].hist(bins=70)


import seaborn as sns
ratings


sns.jointplot(x='rating',y='num of ratings',data=ratings,alpha=0.5)


moviemat=data.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


ratings.sort_values('num of ratings',ascending=False).head(10)



starwar_user_ratings=moviemat['Star Wars (1977)']
liarliar_user_ratings=moviemat['Liar Liar (1997)']
starwar_user_ratings.head()



similar_to_starwar=moviemat.corrwith(starwar_user_ratings)
similar_to_liarliar=moviemat.corrwith(liarliar_user_ratings)



corr_starwar=pd.DataFrame(similar_to_starwar,columns=['correlation'])
corr_starwar.dropna(inplace=True)
corr_starwar.head()


corr_starwar.sort_values('correlation',ascending=False).head(10)


corr_starwar=corr_starwar.join(ratings['num of ratings'])
corr_starwar.head()


corr_starwar[corr_starwar['num of ratings']>100].sort_values('correlation',ascending=False).head()


corr_liarliar=pd.DataFrame(similar_to_liarliar,columns=['correlation'])
corr_liarliar.dropna(inplace=True)


corr_liarliar.sort_values('correlation',ascending=False).head()


corr_liarliar=corr_liarliar.join(ratings['num of ratings'])
corr_liarliar.head()


corr_liarliar[corr_liarliar['num of ratings']>100].sort_values('correlation',ascending=False).head()
