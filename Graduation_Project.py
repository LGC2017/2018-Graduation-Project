import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gensim

#需要读取的文件有三个
#movies.csv ratings.csv tags.csv
#movies:movieId,title,genres 三列参数，分隔符是',' 内容是电影id，名字，和类别
#ratings:userId,movieId,rating,timestamp 内容是用户评分和时间戳
#tags:userId,movieId,tag,timestamp 内容是用户的评价和时间戳
#注意：部分电影可能有rating但是没有tags
Movies_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\movies_mini.csv','r',encoding='utf-8')
Ratings_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\ratings_mini.csv','r',encoding='utf-8')
Tags_File=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\tags_mini.csv','r',encoding='utf-8')


#User类总共有三个总要的datamembers，一个是id，一个是记录评分过的电影相关信息的dataframe：电影id，评分，评分日期
#还有一个是用户对电影打的标签的记录，主要是统计各个标签的出现的频次，如果找到高频词，可以在后面进行处理
#注意用户可能只有tags没有rating
class User(object):
    num=0 #统计User的人数
    
    def __init__(self,userid): #构造函数
        self.id=userid
        self.rating_movies_inf=pd.DataFrame(columns=['movies_id','movies_rating','rating_time']) #评分过的电影的相关信息，构建一个dataframe
        self.tag_keywords={} #用一个字典统计该用户tags的高频词
        self.tag_movieid=[]
        User.num+=1

    def add_movie_inf(self,dataframe):
        self.rating_movies_inf=self.rating_movies_inf.append(dataframe)
        

    def add_movie_tags(self,tags,movieid): #统计一下各个用户评论的高频词
        for i in tags:
            if i not in self.tag_keywords.keys():
                self.tag_keywords[i]=1
            else:
                self.tag_keywords[i]+=1
        self.tag_movieid.append(movieid)


    def sort_moive_inf(self): #给各个用户各自评价过的电影按时间顺序排序
            self.rating_movies_inf=self.rating_movies_inf.sort_values(by='rating_time')

    def sort_tag_inf(UserList): #给tags进行排序，找出高频tag
         for i in UserList:
             i.tag_keywords=sorted(i.items(),key=lambda x:x[1]) #将数据结构从字典转成元组tuple

#电影类的datamembers，主要有id，features，features包括电影的名字，类型和电影的标签,list类型，另外还有一个待用的feature_vector，
#是用来存放训练之后得到的特征向量，方便之后进行距离运算，也是list类型
class Movie(object):
    num=0 #统计总共有多少个电影对象

    def __init__(self,movieid):
        self.id=movieid
        self.features_vector=[]
        self.features=[]
        Movie.num+=1


def construct_vector(MovieList):  #每部电影的关键字，每个关键字可以训练一个向量，可以计算两个1各向量离另外一个向量组的向量距离最小值作为参考
        Tagsline=[m.features for m in MovieList]
        model=gensim.models.Word2Vec(Tagsline,min_count=1,window=6,sg=0,iter=2) #使用CBow，不排除低频词语，窗口为6，因为语料太少
        #得到了model之后，就可以利用model，得到对应feature，以为一个movie的tags有不少，所以vector不止一个，用相加的方式来表征一个movie
        #也可以计算两组vectors之间的差距，一个vector离另一组vector的差距用最近的距离表示，然后所有vector离另一vector组的距离平均值作为组距
        for i in range(len(Tagsline)):
            MovieList[i].features=np.zeros(len(model.wv(Tagsline[0][0]))) #创建全0的向量
            for j in Tagsline[i]:
                MovieList[i].features+=model.wv(j) #对所有tags的值求和


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
        #movie_title=movie_text[-2][0:-6]
        #movie_text[-2]=movie_title
        movie=Movie(int(movie_text[0])) 
        #movie.features.extend(f for f in movie_text[2:-1]) #电影名字不作为特征参数
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
        else:
            userdict[uid].add_movie_inf(movie_df) #往dataframe结构里再加一列
        print(userdict[uid].rating_movies_inf) #测试用

    
    flag=True
    for line in Tags_File: #提取tags，需要给user一份tag，记录高频词，给movie一份，记录features
        if flag==True:
            flag=False
            continue

        tags_text=line.strip().split(',')
        tags=tags_text[2].strip().split(' ')
        movid=int(tags_text[1])
        uid=int(tags_text[0])
        if uid in userdict.keys(): #判断该uid的用户对象是否创建了
            userdict[uid].add_movie_tags(tags,movid) #给用户统计高频词
        else:
            user=User(uid)
            userdict[uid]=user
            userdict[uid].add_movie_tags(tags,movid)
        if movid not in moviedict.keys(): 
            moviedict[movid]=Movie(movid)
        moviedict[movid].features.extend(tags) #将tags计入对应movie得features里      
    return userdict,moviedict

def sort_rating(userdict): #对用户评分过的电影按时间排序
    for i in userdict:
        i.sort_moive_inf()


def Clustering(user,moviedict): #Denpeak
    rating_movie_num=user.rating_movies_inf.shape[0]
    DisMatrix=np.zeros(rating_movie_num, rating_movie_num)
    DisVec=np.zeros(rating_movie_num*(rating_movie_num-1)/2,1)
    cnt=0
    for i in range(rating_movie_num): #记录下两两movie对象的差距，保存到一个dis矩阵上
        for j in range(i-1): 
            DisMatrix[i][j]=np.norm((moviedict[user.rating_movies_inf.iloc[i,0]].features- \
            moviedict[user.rating_movies_inf.iloc[j,0]].features)) #计算二范数,即欧式距离
            DisMatrix[j][i]=DisMatrix[i][j]
            DisVec[cnt]=DisMatrixMatrix[i][j]
            cnt+=1
    np.sort(DisVec) 
    #排序，方便选取领域参数dc，参考Clustering by fast search and find of density peaks，使平均邻居数为总数1%-2%的dc是合适的dc
    #由于我的数据规模不大，我尝试作调整，调成0.4，
    #也即当评论的电影超过5个（5*5=25），我们才进行聚类操作（小于或等于5，dc直接为最短距离，除了两个点一个类，其余的都是一个类）
    dc=DisVec[np.round(len(DisVec)*0.04)]
    #下一步是求密度p，密度用论文中的离散值，或者博客中用高斯核将其转化为连续值，暂用离散值
    p_array=np.zeros([rating_movie_num,2])
    for i in range(len(p)):
        p_array[i][0]=np.sum(np.array([j for j in DisMatrix[i][:] if j>=dc])) #记录该电影的密度
        p_array[i][1]=user.rating_movies_inf.iloc[i,0] #记录电影id
    p_sort=sorted(p_array.tolist(),key=lambda x: x[0],reverse=True) #p_sort是一个排好序的list，根据第一个参数排序,即根据密度排序，第二项是id,降序排列
    pmax=p_sort[0] #是一个list，第一项是密度第二项是电影id
    #求出个点的密度后,所有点的距离都记录在d向量之中，尤其要注意最大密度的点距离是所有点中的最大距离
    d=np.zeros(rating_movie_num)
    for i in range(1,rating_movie_num): #每个点的距离定义为比它密度大的点集中距离最小那个
        Mindis=DisVec[-1] #定义为最大距离
        for j in range(i):
            if Mindis>DisMatrix[p_sort[i][1]][p_sort[j][1]]: #找到两点的下标，由于密度是从大到小排，每次只要扫描前面的点，找到距离最小的即可
                Mindis=DisMatrix[p_sort[i][1]][p_sort[j][1]]
        d[p_sort[i][1]]=Mindis #根据下标在对应位置填上对应的距离
    d[pmax[1]]=DisVec[-1] #密度最大点填上最大距离
    p=[i[0] for i in p_array]
    y=np.zeros([rating_movie_num,2])
    y[0]=np.array(p)*d #记录d*p
    y[1]=p[1] #记录下标
    sorted(y,key=lambda x:x[0])
    maxdiff=0
    divideindex=0
    for i in range(1,len(y)):
        if y[i]-y[i-1]>=maxdiff:
            maxdiff=y[i]-y[i-1]
            divideindex=i
    clustercen_num=len(y)-divideindex #得到类中心数目
    clustercen_index=[y[i][1] for i in range(divideindex,len(y))]
    cluster_dict={}
    for i in range(rating_movie_num):
        if i in clustercen_index:
            cluster_dict[i]=i
            continue
        else:
            Min=DisMatrix[i][clustercen_index[0]]
            belongto=clustercen_index[0]
            for j in range(len(clustercen_index)):
                if DisMatrix[i][clustercen_index[j]]<Min:
                    belongto=j
                    Min=DisMatrix[i][clustercen_index[j]]
            cluster_dict[i]=belongto
    return cluster_dict,clustercen_index


def Judgement(cluster_dict,clustercen_index,user): 
    #得到了分类之后的数据，数据分成若干个类
    #之后要做的就是根据分类判断，用户口味是否发生变化，由于事先已经排好序，所以cluster_dict里的顺序是按时间排的
    #然后根据滑动窗口，计算窗口内计数最多的类别占比多少，设置阈值，超过认为是有偏向性
    #找到最大的滑窗范围，之后在剩下的区域内再设置窗口，重复到所有元素都被划分到区域内
    #如果前后两个滑窗超过阈值，说明两个时间段内用户的口味有偏向，且发生变化，以类中心作为代表，比较距离，进行推荐
    k=len(cluste
