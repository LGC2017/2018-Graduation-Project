import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#需要读取的文件有三个
#movies.csv ratings.csv tags.csv
#movies:movieId,title,genres 三列参数，分隔符是',' 内容是电影id，名字，和类别
#ratings:userId,movieId,rating,timestamp 内容是用户评分和时间戳
#tags:userId,movieId,tag,timestamp 内容是用户的评价和时间戳
#注意：部分电影可能有rating但是没有tags
Movies_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\movies.csv','r',encoding='UTF-8')
Ratings_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\ratings.csv','r',encoding='UTF-8')
Tags_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\tags.csv','r',encoding='UTF-8')


#User类总共有三个总要的datamembers，一个是id，一个是记录评分过的电影相关信息的dataframe：电影id，评分，评分日期
#还有一个是用户对电影打的标签的记录，主要是统计各个标签的出现的频次，如果找到高频词，可以在后面进行处理
class User(object):
    num=0 #统计User的人数
    
    def __init__(self,userid): #构造函数
        self.id=userid
        self.rating_movies_inf=pd.DataFrame(columns=['movies_id','movies_rating','rating_time']) #评分过的电影的相关信息，构建一个dataframe
        self.tag_keywords={} #用一个字典统计该用户tags的高频词
        User.num+=1

    def add_movie_inf(self,dataframe):
        self.rating_movies_inf.append(dataframe)

    def add_movie_tags(self,tags): #统计一下各个用户评论的高频词
        for i in tags:
            if i not in self.tag_keywords.keys():
                tag_keywords[i]=1
            else:
                tag_keywords[i]+=1


    def sort_moive_inf(UserList): #给各个用户各自评价过的电影按时间顺序排序
        for i in UserList:
            i.rating_movies_inf=i.rating_movies_inf.sort_values(by='rating_time')

    def sort_tag_inf(UserList): #给tags进行排序，找出高频tag
         for i in UserList:
             i.tag_keywords=sorted(i.items(),key=lambda x:x[1]) #将数据结构从字典转成元组tuple

#电影类的datamembers，主要有id，features，features包括电影的名字，类型和电影的标签,list类型，另外还有一个待用的feature_vector，
#是用来存放训练之后得到的特征向量，方便之后进行距离运算，也是list类型
class Movie(object):
    num=0 #统计总共有多少个电影对象

    def __init__(self,movieid):
        self.id=movieid
        self.feature_vector=[]
        self.feature=[]
        Movie.num+=1

    def construct_vector(MovieList): #暂时空缺
        pass


def LoadData():
    #采用字典将id和对应编号Movie对象进行链接
    userdict={}
    moviedict={}
    for line in Movie_File[1:]:
        movie_text=[line.strip().split(',')]
        genres=[movie_text[-1].strip().split('|')] #将genres里面的词语拆开
        movie_text[-1]=genres
        #movie_title=movie_text[-2][0:-6]
        #movie_text[-2]=movie_title
        movie=Movie(int(movie_text[1][0])) 
        movie.features.extend(f for f in movie_text) #提取数据，数据类型是文本类型
        moviedict[int(movie_text[1][0])]=movie

    for line in Ratings_File[1:]:
        ratings_text=[line.strip().split(',')]
        for i in range(len(ratings_text)): #留意list里存储的数据的类型，并且注意list里面是否包含list
            if i==2:
                ratings_text[i][0]=float(ratings_text[i][0])
            else:
                ratings_text[i][0]=int(ratings_text[i][0])
        #用户id
        uid=ratings_text[0][0]
        #构建movie的dataframe数据结构
        movie_df=pd.DataFrame(data=[i[0] for i in ratings_text[1:]],columns=['movies_id','movies_rating','rating_time']) 
        if uid not in userdict.keys(): #判断该uid的用户对象是否创建了
            user=User(uid) #ratings_text第0项是id
            user.add_movie_inf(movie_df) #往user对象添加它评分过的电影的信息
            userdict[uid]=user
        else:
            userdict[uid].add_movie_inf(movie_df) #往dataframe结构里再加一列


    for line in Tags_File[1:]:
        tags_text=[line.strip().split(',')]
        