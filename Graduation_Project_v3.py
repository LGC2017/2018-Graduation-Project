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

Compare_Output=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\op_version3_v1.csv','w',encoding='latin1')
case3_seq=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\version3_case3_seq_v1.csv','r',encoding='latin1')
optimize_hit_num=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\version3_optimize_hit_num.csv','w',encoding='latin1')
origin_hit_num=open('D:\\SYSU课程\大三\毕业论文相关附件\\ml-latest-small\\version3_origin_hit_num.csv','w',encoding='latin1')

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
        movie=Movie(int(movie_text[0])) 
        movie.features.extend(genres) #提取数据，数据类型是文本类型
        moviedict[int(movie_text[0])]=movie


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
        if movid not in moviedict.keys(): 
            moviedict[movid]=Movie(movid)
        #moviedict[movid].features.extend(tags) #将tags计入对应movie得features里
        for i in tags:
            if i not in moviedict[movid].features:
                moviedict[movid].features.append(i)
        '''
        if uid in userdict.keys(): #判断该uid的用户对象是否创建了
            userdict[uid].add_movie_tags(tags,movid) 
        else:
            user=User(uid)
            userdict[uid]=user
            userdict[uid].add_movie_tags(tags,movid)
        '''
    



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

            
        
    return userdict,moviedict

def sort_rating(userdict): #对用户评分过的电影按时间排序
    for i in userdict.values():
        i.sort_moive_inf()
        


def divide_validation(userdict):
    for user in userdict.values():
        rating_num=user.rating_movies_inf.shape[0]
        validation_num=int(np.round(rating_num*0.18))
        user.validation=[user.rating_movies_inf.iloc[i,0] for i in range((rating_num-validation_num),rating_num)]
        user.rating_movies_inf.index = range(len(user.rating_movies_inf))
        v=[i for i in range((rating_num-validation_num),rating_num)]
        user.rating_movies_inf.drop(v,axis = 0,inplace=True)


def Cal_Movie_Distance(user,f_list1,f_list2):
    point=0
    MAX=100
    cnt=0
    for i in f_list1:
            if i in f_list2 and i in user.tag_keywords.keys():
                point+=user.tag_keywords[i]
                cnt+=1
    if point==0:
        return MAX
    else:
        '''
        #设定惩罚系数
        punish=point-(max(len(f_list1),len(f_list2))-cnt) #每多一个多余标签，就对分数减一，作为惩罚
        if punish<=2:
            if point==1:
                return 100/(point)
            else:
                return 100/(point-1)
        else:
            return 100/punish
        '''
        return 100/(point)

def Clustering(user,moviedict): #Denpeak
    rating_movie_num=user.rating_movies_inf.shape[0]
    if rating_movie_num<=2: #数量少于3，没必要进行分类处理
        D={}
        DI=[0]
        D[0]=0
        if rating_movie_num==2:
            D[1]=0
            DI.append(1)
        return D,DI #只有一个变量，那点本身属于自己，中心也是自己
    
    DisMatrix=np.zeros([rating_movie_num, rating_movie_num])
    DisVec=np.zeros(int(rating_movie_num*(rating_movie_num-1)/2))
    cnt=0

    for i in range(rating_movie_num): #记录下两两movie对象的差距，保存到一个dis矩阵上
        for j in range(i): 
            #DisMatrix[i][j]=np.linalg.norm((moviedict[user.rating_movies_inf.iloc[i,0]].features_vector- \
            #moviedict[user.rating_movies_inf.iloc[j,0]].features_vector)) #计算二范数,即欧式距离

            #DisMatrix[i][j]=Cal_Distance(moviedict[user.rating_movies_inf.iloc[i,0]].features_vector, \
            #moviedict[user.rating_movies_inf.iloc[j,0]].features_vector)

            DisMatrix[i][j]=Cal_Movie_Distance(user,moviedict[user.rating_movies_inf.iloc[i,0]].features, \
            moviedict[user.rating_movies_inf.iloc[j,0]].features)

            DisMatrix[j][i]=DisMatrix[i][j]
            DisVec[cnt]=DisMatrix[i][j]
            cnt+=1
    DisVec.sort()
    #排序，方便选取领域参数dc，参考Clustering by fast search and find of density peaks，使平均邻居数为总数1%-2%的dc是合适的dc
    #由于我的数据规模不大，我尝试作调整，调成0.4，
    #也即当评论的电影超过5个（5*5=25），我们才进行聚类操作（小于或等于5，dc直接为最短距离，除了两个点一个类，其余的都是一个类）

    #print(np.round(len(DisVec)*0.3))
    dc=DisVec[int(np.round(len(DisVec)*0.1))] #全连接，总共n*(n-1)/2个连接，DisVec总大小就是这个，选取一个参数，能控制领域数量在一定范围

    #下一步是求密度p，密度用论文中的离散值，或者博客中用高斯核将其转化为连续值，暂用离散值
    p_array=np.zeros([rating_movie_num,3])
    p_max=0
    for i in range(len(p_array)):
        p_array[i][0]=len([j for j in DisMatrix[i][:] if j<dc]) #记录该电影的密度
        p_array[i][1]=user.rating_movies_inf.iloc[i,0] #记录电影id
        p_array[i][2]=i #记录电影当前的编号，后面排序之后根据编号从距离矩阵中找距离
        if p_max<p_array[i][0]: p_max=p_array[i][0]
    
    #找出outlier
    outlier=[int(p[2]) for p in p_array if p[0] <=np.round(rating_movie_num*0.02)]
    
    #对密度p进行归一化
    for i in p_array:
        i[0]/=p_max
    p_sort=sorted(p_array.tolist(),key=lambda x: x[0],reverse=True)    #p_sort是一个排好序的list，根据第一个参数排序,即根据密度排序，第二项是id,降序排列
    pmax=p_sort[0]  #是一个list，第一项是密度，第二项是电影id，第三项是编号
    


    #排除outlier，然后找到点间最大距离，方便归一化
    max_dis=max([np.max(DisMatrix[i][:]) for i in range(rating_movie_num) if i not in outlier])
    DisMatrix=DisMatrix/max_dis

    #求出个点的密度后,所有点的距离都记录在d向量之中，尤其要注意最大密度的点距离是所有点中与密度最大点的最大距离
    #由于密度已经事先从大到小排序，所以只要从最大的密度项遍历到当前项，找到距离最小的项即可
    d=np.zeros(rating_movie_num)
    for i in range(1,rating_movie_num): #每个点的距离定义为比它密度大的点集中距离最小那个
        Mindis=DisVec[-1] #定义为最大距离
        if int(p_sort[i][2]) in outlier:
            d[int(p_sort[i][2])]=0 #离群点的距离设置为0
            continue
        for j in range(i):
            if Mindis>DisMatrix[int(p_sort[i][2])][int(p_sort[j][2])]: #找到两点的下标，由于密度是从大到小排，每次只要扫描前面的点，找到距离最小的即可
                Mindis=DisMatrix[int(p_sort[i][2])][int(p_sort[j][2])]
        d[int(p_sort[i][2])]=Mindis #根据下标在对应位置填上对应的距离
    d[int(pmax[2])]=1 #密度最大点默认一定是类中心


    p=np.zeros(rating_movie_num)
    for i in p_sort:
        p[int(i[2])]=i[0] #按顺序将p的值排好
    y=np.zeros([rating_movie_num,2])
    y[:,0]=np.array(p)*d #记录d*p
    y[:,1]=np.array(range(rating_movie_num))
    y=sorted(y,key=lambda x:x[0],reverse=True)
    

    lastdiv=0
    divideindex=0
    prediv=0
    div_cnt=0
    #当出现2次prediv<=2*div,我们认为之前的点皆为

    #最大差分的一边全是类中心，另一边就是非类中心
    #分割点之后的数据会趋于平稳，也即斜率变化不会很剧烈，所以同时考察连续几个点的斜率的变化的变化程度，二次求导
    #用差分不准确，密度从大到小排序，每次搜索三个点，计算差分的差分

    for i in range(2,len(y)):
        div1=y[i-1][0]-y[i-2][0]
        div2=y[i][0]-y[i-1][0]
        div=abs(div2-div1)

        if prediv<=2*div and i!=2: #如果计数超过2，表明这一段几个点斜率变化区域局部稳定，分界点
            div_cnt+=1
        else:
            div_cnt=0
        if div_cnt==2:
            divideindex=i-3 #记录差距最大的地方divideindex，包括divideindex在内到最后的点的坐标即为中心点坐标
            #print(divideindex)
            break
        prediv=div

    clustercen_num=divideindex+1 #得到类中心数目
    clustercen_index=[int(y[i][1]) for i in range(clustercen_num)] #记录类中心在评论集合中的下标
    cluster_dict={} #该字典是记录点->类中心的映射，key是点下标，item是类中心下标 

    for i in range(rating_movie_num):
        if i in clustercen_index:
            cluster_dict[i]=i
            continue
        if i in outlier:
            cluster_dict[i]=-1 #离群点全划到-1
        else:
            Min=DisMatrix[i][int(clustercen_index[0])] #初始化，点离类中心的最小距离
            belongto=clustercen_index[0] #初始化，点的归属

            for j in range(len(clustercen_index)):
                if DisMatrix[i][int(clustercen_index[j])]<Min:
                    belongto=int(clustercen_index[j])
                    Min=DisMatrix[i][int(clustercen_index[j])]
            cluster_dict[i]=belongto
    #可以考虑重构类中心（取平均值，点不一定实际存在
    clustercen_index.append(-1)
    '''
    for i in clustercen_index:
        for j in cluster_dict.keys():
            if cluster_dict[j]==i:
                print(moviedict[user.rating_movies_inf.iloc[j,0]].features)
        print('\n')
    '''
    return cluster_dict,clustercen_index




def cal_ratio(movlist,start,class_num,min_element_num,cluster_dict): 
    #传入参数有电影列表，窗口开始位置，聚类的个数，最小元素的类别的元素个数
    window_end=int(round(0.3*len(cluster_dict))) #增加一个限定，就是如果发生类型转变，正常不应该超过30%的位置
    countnum=np.zeros(class_num) #记录每种类型的电影的数量
    init_end=start-min_element_num #窗口初始尾部
    
    if init_end<0: #如果发现剩下的散点数量太少了，直接返回0和0
        return -1,0,0,0
    end=init_end
    if init_end>=window_end:
        window_end=init_end-1

    max_ratio=0
    cnt=0 #记录搜索了几个类中心
    cen2index={} #给类中心编号，方便数组记录
    max_cluster=[]

    for i in range(end,-1,-1): #window_end和-1
    #for i in range(end,-1,-1):
        #真正计数的是countnum数组,而这个字典，记录的是countnum下标对应的是哪一个类中心
        tmplist=movlist[i:start] #窗口，不断变长，统计用户观影是否有偏向性
        if i==init_end: #第一步，要扫描序列到当前i的位置，之后只要扫描最新的位置即可
            for j in tmplist:
                if cluster_dict[j] not in cen2index.keys(): 
                    #排除离群点,不进入计数中
                    if cluster_dict[j]==-1:
                        continue
                    cen2index[cluster_dict[j]]=cnt
                    countnum[cnt]+=1
                    cnt+=1
                else:
                    countnum[cen2index[cluster_dict[j]]]+=1
            if np.sum(countnum)!=0:
                max_ratio=np.max(countnum)/np.sum(countnum)
        else:
            if cluster_dict[tmplist[0]] not in cen2index.keys():
                #排除离群点,不进入计数中
                if cluster_dict[tmplist[0]]==-1:
                    continue
                cen2index[cluster_dict[tmplist[0]]]=cnt
                countnum[cnt]+=1
                cnt+=1
            else:
                countnum[cen2index[cluster_dict[tmplist[0]]]]+=1
        if np.sum(countnum)!=0:
            ratio=np.max(countnum)/np.sum(countnum) #设定阈值
            max_cluster= [ind[0] for ind in cen2index.items() if ind[1]==np.argmax(countnum)] #计算最大比例的簇
            if max_ratio<=ratio: #等于也要换，因为可以延长序列
                max_ratio=ratio
                end=i
        if i<=window_end and len(max_cluster)!=0:
                break
    return max_ratio,start,end,max_cluster[0] 
    #返回参数有四个，一个是最大比例ratio，另一个是截至位置，即窗口框住的边缘situation,左右都要返回
    #分别是start和end，而end，区间是[end,start),窗口从右往左数，还有一个是最大占比的类别



def Judgement(cluster_dict,clustercen_index,user,moviedict): 
    #得到了分类之后的数据，数据分成若干个类
    #之后要做的就是根据分类判断，用户口味是否发生变化，由于事先已经排好序，所以cluster_dict里的顺序是按时间排的
    #然后根据滑动窗口，计算窗口内计数最多的类别占比多少，设置阈值，超过认为是有偏向性
    #找到最大的滑窗范围，之后在剩下的区域内再设置窗口，重复到所有元素都被划分到区域内
    #如果前后两个滑窗超过阈值，说明两个时间段内用户的口味有偏向，且发生变化，以类中心作为代表，比较距离，进行推荐

    if len(cluster_dict)<=2: #如果发现评分数量太少，直接通过比较相似度进行推荐
        return 0,0,4 #case 4，直接对比推荐

    if len(clustercen_index)==2: #发现所有点除了离群点都同属一个分类，说明口味没有发生过变化
        return clustercen_index[0],clustercen_index[0],0 #case 0，用中心点作为特征型来对比，进行推荐 



    k=len(clustercen_index)
    movlist=[a for a in cluster_dict.keys()]
    #找到最小簇的元素数量
    min_element_num=int(np.round(len(cluster_dict)*0.1))
    if min_element_num>10:
        min_element_num=10


    if min_element_num<4: #窗口长度至少为3
        min_element_num=4
    
    #设置阈值为0.4
    threshold1=0.6
    #threshold2=0.9

    #边界情况，总共3个点，两个点属于一类，另一个点单独一类
    #只要统计两个区间即可，判断两个区间情况即可知道
    #第一个区间要考虑用户最进情况，所以窗口时固定从最新得评论开始
    ratio1,s1,e1,max_cluster1=cal_ratio(movlist,len(movlist)-1,len(clustercen_index),min_element_num,cluster_dict)
    
    #第二个区间，则是考虑之前是否存在偏向，所以应该用最大连续子数列和的思路找到占比最大的一段序列
    #如果该序列超过某个阈值表明用户之前确对某类型电影有偏好，运用动态规划实现

    #if ratio1>=threshold: #表示最近有偏爱类型，从e1开始往后搜索
        #ratio2,max_cluster2=dp_cal_ratio(movlist,e1-1,clustercen_index,min_element_num,cluster_dict)
    #else: #如果最近没有偏爱类型，那就从头开始用动规的方法找到最比例最大的子串，查看是否曾经有过偏爱类型
         #ratio2,max_cluster2=dp_cal_ratio(movlist,s1-1,clustercen_index,min_element_num,cluster_dict)
    #ratio2,max_cluster2=dp_cal_ratio(movlist,e1-1,clustercen_index,min_element_num,cluster_dict)

    fre_list1=weight_sum(moviedict,e1-1,-1,clustercen_index,cluster_dict,user,0,0)
    fre_list2=weight_sum(moviedict,len(movlist)-1,e1-1,clustercen_index,cluster_dict,user,1,max_cluster1)
    

    print(fre_list1)
    print(fre_list2)

    if ratio1>=threshold1 and len(fre_list2)>0 and len(fre_list1)>0:
        FLAG=0
        for i in fre_list1:
            for j in fre_list2:
                if i[0]==j[0]:
                    FLAG=1
        if FLAG==0:
            return 1,fre_list1,fre_list2
        else:
            return 0,0,0

   
        #sum=0 #计算一下总的频数
        #for i in fre_list2:
            #sum+=i[1]
        #ptr=0
        #for i in fre_list1:
            #for j in fre_list2:
                #if j[0]==i[0]:
                    #ptr+=j[1]
                    #fre_list2.remove(j)
                    #break
        #if (sum-ptr)/sum >=0.8:
            #case=1
        #else:
            #case=0

        #return case,fre_list1,fre_list2

    else:
        return 0,0,0
       
              

def dp_cal_ratio(movlist,start,clustercen_index,min_element_num,cluster_dict): #用动态规划找出连续序列中占比最大的电影类型
    #返回值有两个，比例和类型名字，第二段序列不必知道起始，因为不作后续运算
    init_end=start-min_element_num #窗口初始尾部
    if init_end<=0: #序列太短，直接返回0和0
        return -1,0
    end=init_end
    ratiolist=[]
    for i in clustercen_index: #根据不同的类中心，挑出最大比例的序列
        if i==-1:
            continue #排除离群点
        ratiolist.append(dp_cal_R(movlist,start,min_element_num,i,cluster_dict))
    max_ratio_index=ratiolist.index(max(ratiolist))
    return max(ratiolist),clustercen_index[max_ratio_index]

    



def dp_cal_R(movlist,start,min_element_num,cla,cluster_dict): #cla表示需要计算的是哪一个类的最大连续串数量，有多少个类就要调用多少次函数
    max_ratio=0
    ratio_num=0 #比例分子，即记录下的当前最大比例的序列的分子
    ratio_den=0 #比例的分母，即记录下的当前最大比例的序列的分母
    count=0 #扫描过的同类型目标，每次最大序列更换起点就清零
    count_outlier=0
    arr_start=-1 #记录最大占比序列起点
    arr_end=-1   #记录最大占比序列终点 
    flag=-1 #判断当前序列是否在内循环已经扫描过，用来跳过外循环某些部分

    #最大比例序列=max(之前最大序列，以当前的同类型目标为新起点构建的序列，以当前目标为终点之前最大序列起点不变的序列)
    #返回值：比例ratio
    #循环可能情况：找到指定类型目标后发现后面序列太短，没找到同类型目标，返回0

    for i in range(start,-1,-1):
        if i>flag and flag!=-1: #假设上一步将最大比例序列更换为以i为起点的新序列，直接跳到终点，中间不用扫描，flag记录i应该跳到的位置
            continue
        if flag!=-1 and i==flag: #当i跳到新序列结尾，不用再跳，重置flag
            flag=-1
       

        if cluster_dict[i]==cla: #找到首个同类型的目标
            count=count+1 #最大比例序列的终点开始，到当前位置，总共有几个目标类型
            if i>=min_element_num: #判断后续序列足够长
                cla_count=len([j for j in range(i,i-min_element_num,-1) if cluster_dict[j]==cla]) #以i为起点，计算新序列同类型目标数
                outlier=len([j for j in range(i,i-min_element_num,-1) if cluster_dict[j]==-1]) #默认离群点比min_element_num小
                if max_ratio==0: #初次启动
                    ratio_num=cla_count #最大比例的字段中，属于指定类型的电影有几个
                    ratio_den=min_element_num-outlier #总数
                    max_ratio=ratio_num/ratio_den #初算比例
                    arr_start=i #记录开头
                    arr_end=i-min_element_num+1 #记录结尾
                    count=0
                    count_outlier=0
                    flag=i-min_element_num
                else: 
                    #max_ratio已经有数据，表明之前已经有一个序列，新序列要与旧序列比较
                    new_ratio=cla_count/min_element_num-outlier #新起点序列，该类型电影所占的比例
                    #联合比例，以旧序列的起点为起点，以当前位置为终点，计算比例，首先分子=当前最大序列分子+最大序列结尾到当前位置i还有多少目标
                    #分母=当前最大序列分母+最大序列结尾到当前位置i 
                    
                    combine_ratio=(ratio_num+count)/(ratio_den+(arr_end-i)-count_outlier) 
                    if new_ratio>max_ratio and new_ratio>combine_ratio: #最大序列变为以i为起点的新序列
                        max_ratio=new_ratio
                        arr_start=i
                        arr_end=i-min_element_num+1
                        count=0
                        count_outlier=0
                        ratio_num=cla_count
                        ratio_den=min_element_num-outlier
                        flag=i-min_element_num

                    elif combine_ratio>=max_ratio and combine_ratio>=new_ratio: #最大序列变为以i为结尾，起点不变的新序列
                        max_ratio=combine_ratio
                        ratio_den+=(arr_end-i-count_outlier)
                        arr_end=i #当前最大比例序列起点不变，终点变为当前位置
                        ratio_num+=count
                        count=0 #之前统计的从最大序列到i之中有多少个目标，现在序列包含了这些目标，清零重新统计
                        count_outlier=0 #计算离群点的计数器也要清零
                    #不满足就条件就不更换序列
            else: 
                #序列不够长，不足以从i位置为起点构建min_element_num长度的序列
                combine_ratio=(ratio_num+count)/(ratio_den+(arr_end-i)-count_outlier) 
                if combine_ratio>=max_ratio:
                        max_ratio=combine_ratio
                        ratio_den+=(arr_end-i)-count_outlier
                        arr_end=i
                        ratio_num+=count
                        count=0
                        count_outlier=0
        elif cluster_dict[i]==-1 and max_ratio!=0:
            count_outlier+=1
    
    return max_ratio


def weight_sum(moviedict,start,end,clustercen_index,cluster_dict,user,selection,centre):
    #high_freq[len(clustercen_index)]={}
    high_freq={}
    cnt=0
    if selection==0: #统计以前序列的高频属性
        for i in range(start,end,-1): #统计一下高频属性
            cnt+=1
            for j in moviedict[user.rating_movies_inf.iloc[i,0]].features:
                if j not in high_freq.keys():
                    high_freq[j]=1
                else:
                    high_freq[j]+=1
    else:
        for i in range(start,end,-1): #统计一下高频属性
            if cluster_dict[i]==centre:
                cnt+=1
                for j in  moviedict[user.rating_movies_inf.iloc[i,0]].features:
                    if j not in high_freq.keys():
                        high_freq[j]=1
                    else:
                        high_freq[j]+=1
            
    fre_list=sorted(high_freq.items(),key=lambda x:x[1],reverse=True)
    if selection==0:
        fre_list=[i for i in fre_list if i[1]>=int(cnt*0.7)]
    else:
        fre_list=[i for i in fre_list if (i[1]>=cnt*0.7 and i[1]>2)]
    return fre_list




                        
def MovieRecommendation(userdict,moviedict):
    #遍历，对每个用户进行聚类，判决口味变化
    #userdict里存有评分过电影得列表，再到moviedict寻找电影得相关参数
    global sp_user
    sp_user=[]
    CNT=0
    for u in userdict.values():
        cluster_dict,clustercen_index=Clustering(u,moviedict) 
        #得到多个簇，注意该处返回的dict和index都是下标，即在当前user的评论列表中的第i个数据

        #ratio,max_cluster,case=Judgement(cluster_dict,clustercen_index,u,moviedict) 
        #根据簇进行判决,根据case的不同，max_cluster可能有两个list，也可能是单个list
        case,fre_list1,fre_list2=Judgement(cluster_dict,clustercen_index,u,moviedict) 

        if case==1: #表明前后有明显的偏好变化，增大后面偏向的权重，使新推荐更快的启动
            '''
            sp_user.append(u.id)
            for i in cluster_dict.keys():
                if cluster_dict[i]==max_cluster[0]:
                    for j in moviedict[u.rating_movies_inf.iloc[i,0]].features:
                        if j in u.tag_keywords.keys():
                            u.tag_keywords[j]+=3
                elif cluster_dict[i]==max_cluster[1]:
                    for j in moviedict[u.rating_movies_inf.iloc[i,0]].features:
                        if j in u.tag_keywords.keys():
                            u.tag_keywords[j]-=1
            '''

            sp_user.append(u.id)
            for j in fre_list2:
                if j[0] in u.tag_keywords.keys():
                    u.tag_keywords[j[0]]+=15*j[1]
            for j in fre_list1:
                if j in u.tag_keywords.keys():
                    u.tag_keywords[j]*=1


        print(CNT)
        CNT+=1
    #print(sp_user,file=case3_seq)
        
    #根据判决结果，进行topk比较，如果有口味变化，越接近当前时刻的电影的权重越大，远离越少


def validation(user,RecList):
    hit=0
    hit_list=[]
    for i in RecList:
        if i in user.validation:
            hit+=1
            hit_list.append(i)          
    print(hit/(len(RecList)))
    if user.id in Case3_L:
    #if user.id in sp_user:
        #print(user.validation,'\n',RecList,'\n',hit_list,'\n',file=Compare_Output)
        print(hit,file=origin_hit_num)
        #print(hit,file=optimize_hit_num)
    
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
    #MovieRecommendation(userdict,moviedict)
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
