import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim
from numpy import linalg as LA

#需要读取的文件有三个
#movies.csv ratings.csv tags.csv
#movies:movieId,title,genres 三列参数，分隔符是',' 内容是电影id，名字，和类别
#ratings:userId,movieId,rating,timestamp 内容是用户评分和时间戳
#tags:userId,movieId,tag,timestamp 内容是用户的评价和时间戳
#注意：部分电影可能有rating但是没有tags
Movies_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\movies.csv','r',encoding='latin1')
Ratings_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\ratings.csv','r',encoding='latin1')
Tags_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\tags.csv','r',encoding='latin1')

case3_seq=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\version1_case3_seq.csv','r',encoding='latin1')
origin_hit_num=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\version2_origin_hit_num.csv','w',encoding='latin1')

#User类总共有三个总要的datamembers，一个是id，一个是记录评分过的电影相关信息的dataframe：电影id，评分，评分日期
#还有一个是用户对电影打的标签的记录，主要是统计各个标签的出现的频次，如果找到高频词，可以在后面进行处理
#注意用户可能只有tags没有rating
class User(object):
    num=0 #统计User的人数
    
    def __init__(self,userid): #构造函数
        self.id=userid
        self.rating_movies_inf=pd.DataFrame(columns=['movies_id','movies_rating','rating_time']) #评分过的电影的相关信息，构建一个dataframe
        self.tag_keywords={} #用一个字典统计该用户tags的高频词
        self.validation=[] #用来保留最后的10%数据
        User.num+=1

    def add_movie_inf(self,dataframe):
        self.rating_movies_inf=self.rating_movies_inf.append(dataframe)
        

    def add_movie_tags(self,tags,movieid): #统计一下各个用户评论的高频词
        Rating=(self.rating_movies_inf[(self.rating_movies_inf.movies_id==movieid)].movies_rating.tolist())
        if len(Rating)==0:
            Rating=3
        else:
            Rating=Rating[0]
        for i in tags:
            if i not in self.tag_keywords.keys():
                self.tag_keywords[i]=float(Rating)*1.5
            else:
                self.tag_keywords[i]+=Rating*1.5


    def sort_moive_inf(self): #给各个用户各自评价过的电影按时间顺序排序
            self.rating_movies_inf=self.rating_movies_inf.sort_values(by='rating_time')

    def add_tag_inf(self,movie,rating): #给tags进行排序，找出高频tag
         for f in movie.features:
             if f not in self.tag_keywords.keys():
                 self.tag_keywords[f]=float(rating)
             else:
                 self.tag_keywords[f]+=float(rating)

#电影类的datamembers，主要有id，features，features包括电影的名字，类型和电影的标签,list类型，另外还有一个待用的feature_vector，
#是用来存放训练之后得到的特征向量，方便之后进行距离运算，也是list类型

class Movie(object):
    num=0 #统计总共有多少个电影对象

    def __init__(self,movieid):
        self.id=movieid
        #self.features_vector=0
        self.features_vector=[]
        self.features=[]
        Movie.num+=1




def LoadData():
    #采用字典将id和对应编号Movie对象进行链接
    userdict={}
    moviedict={}
    flag=True
    for line in Movies_File:
        if flag==True:
            flag=False
            continue

        movie_text=line.strip().split(',')
        genres=movie_text[-1].strip().split('|') #将genres里面的词语拆开
        genres=[i.lower() for i in genres]
        
        #加入名字作为特征
        #movie_title=(movie_text[-2].strip().split(' '))[:-1]
        #filter(None, movie_title)
        #if len(movie_title)!=0 and movie_title[-1][0]=='(':
            #movie_title.pop()
        #movie_title=[i.lower() for i in movie_title]
        
        movie=Movie(int(movie_text[0])) 
        
        #movie.features.extend(f for f in movie_title if f not in stopword) #电影名字作为特征参数

        movie.features.extend(genres) #提取数据，数据类型是文本类型
        moviedict[int(movie_text[0])]=movie

    flag=True
    for line in Ratings_File:
        if flag==True:
            flag=False
            continue

        ratings_text=line.strip().split(',')
        for i in range(len(ratings_text)): #留意list里存储的数据的类型，并且注意list里面是否包含list
            if i==2:
                ratings_text[i]=float(ratings_text[i])
            else:
                ratings_text[i]=int(ratings_text[i])
        #用户id
        uid=ratings_text[0]
        #构建movie的dataframe数据结构
        movie_df=pd.DataFrame(data=[[i for i in ratings_text[1:]]],columns=['movies_id','movies_rating','rating_time']) 
        if uid not in userdict.keys(): #判断该uid的用户对象是否创建了
            user=User(uid) #ratings_text第0项是id
            user.add_movie_inf(movie_df) #往user对象添加它评分过的电影的信息
            userdict[uid]=user
            user.add_tag_inf(moviedict[ratings_text[1]],ratings_text[2])
        else:
            userdict[uid].add_movie_inf(movie_df) #往dataframe结构里再加一列
            userdict[uid].add_tag_inf(moviedict[ratings_text[1]],ratings_text[2])

    
    flag=True
    for line in Tags_File: #提取tags，需要给user一份tag，记录高频词，给movie一份，记录features
        if flag==True:
            flag=False
            continue

        tags_text=line.strip().split(',')
        tags=tags_text[2].strip().split(' ')
        tags=[i.lower() for i in tags] #转成小写
        movid=int(tags_text[1])
        uid=int(tags_text[0])
        if uid in userdict.keys(): #判断该uid的用户对象是否创建了
            userdict[uid].add_movie_tags(tags,movid) 
        else:
            user=User(uid)
            userdict[uid]=user
            userdict[uid].add_movie_tags(tags,movid)
            
        if movid not in moviedict.keys(): 
            moviedict[movid]=Movie(movid)
        moviedict[movid].features.extend(tags) #将tags计入对应movie得features里

    return userdict,moviedict

def sort_rating(userdict): #对用户评分过的电影按时间排序
    for i in userdict.values():
        i.sort_moive_inf()
        


def divide_validation(userdict):
    for user in userdict.values():
        rating_num=user.rating_movies_inf.shape[0]
        validation_num=int(np.round(rating_num*0.15))
        user.validation=[user.rating_movies_inf.iloc[i,0] for i in range((rating_num-validation_num),rating_num)]
        user.rating_movies_inf.index = range(len(user.rating_movies_inf))
        v=[i for i in range((rating_num-validation_num),rating_num)]
        user.rating_movies_inf.drop(v,axis = 0,inplace=True)



def validation(user,RecList):
    hit=0
    hit_list=[]
    for i in RecList:
        if i in user.validation:
            hit+=1
            hit_list.append(i)          
    print(hit/(len(RecList)))
    if user.id in Case3_L:
        print(hit,file=origin_hit_num)
    return hit,len(RecList)


def Test_Recommendation(userdict,moviedict):
        HIT=0
        REC_NUM=0
        for user in userdict.values():
            Rec_Len=len(user.validation)
            if Rec_Len<10:
                Rec_Len=10
            ratinglist=list(user.rating_movies_inf.iloc[:,0])
            Recommend_Point=[]
            for m in moviedict.values():
                temppt=0
                if m.id in ratinglist:
                    continue
                for i in m.features:
                    if i in user.tag_keywords.keys():
                        temppt+=user.tag_keywords[i]
                Recommend_Point.append([temppt,m.id])
            Recommend_Point=sorted(Recommend_Point,key=lambda x:x[0],reverse=True)
            RecList=[i[1] for i in Recommend_Point[0:Rec_Len]]

            H,R=validation(user,RecList)
            HIT+=H
            REC_NUM+=R
        print(HIT/REC_NUM)

def Test():
    userdict,moviedict=LoadData() #载入数据
    sort_rating(userdict) #调用User对象自身的sort_moive_inf函数对rating信息按时间排序
    divide_validation(userdict)
    #给用户推荐作品
    Test_Recommendation(userdict,moviedict)

if __name__=='__main__':
    global Case3_L
    for line in case3_seq:
            TMP=line.strip().split(',')
            Case3_L=[int(i) for i in TMP]
    Test()
    Movies_File
    Ratings_File.close()
    Tags_File.close()
