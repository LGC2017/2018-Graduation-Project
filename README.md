# 2018毕业设计

包含项目描述，参考文献，会议总结的内容

---

## 项目描述

此项目是设计一个推荐系统recommendation system.该推荐系统基于content-based recommendation system，创新点是对时间因素的运用和标签向量化的改进。

有别于传统的推荐系统直接以时间因素作为每个商品的权重来进行评估，我打算运用聚类的方法，来判别用户的口味有无明显的变化，此方法预计能更好地给商品赋予权重。因为一个用户如果在两个不同的连续时间段内购买两种特征有明显区别的商品，我们有理由相信该用户的口味发生变化，此时我们可以给符合最新口味的商品赋予更高的权重;假如一个用户购买的商品种类繁多，但在特定时间点并无倾向于拥有某种特征的商品，简单地根据商品购买时间来赋予权重，会影响推荐系统的推荐质量。

另外，将标签向量化的方法，会直接影响content-based recommendation system的推荐效果，如何修改网络，让生成的向量更好地表征标签是另一个优化方向，之后如果有更详细地优化方法，会补充到文档中。

此项目将运用推荐系统常用的数据集movielens，编写语言为python。

## 项目计划

第一~第二周：阅读相关论文，包括但不仅限于下面提到的参考文献，寻找合适的聚类类型和网络来实现项目。

第三~四周：编码

第五~六周：debug，验证系统效果

## 参考文献

A Content-Based Recommendation System Using Neuro-Fuzzy Approach (2018 IEEE)
https://ieeexplore.ieee.org/ielx7/8466242/8491438/08491543.pdf?tp=&arnumber=8491543&isnumber=8491438&tag=1

A supervised learning approach to build a recommendation system for user-generated content in a casual game (2017 IEEE)
https://ieeexplore.ieee.org/ielx7/8399565/8400303/08400316.pdf?tp=&arnumber=8400316&isnumber=8400303

A review on deep learning for recommender systems: challenges and remedies (2018 Springer Nature)
https://github.com/hongleizhang/RSPapers/blob/master/01-Surveys/2018-Explainable%20Recommendation%20A%20Survey%20and%20New%20Perspectives.pdf

Explainable Recommendation: A Survey and New Perspectives
https://github.com/hongleizhang/RSPapers/blob/master/01-Surveys/2018-A%20review%20on%20deep%20learning%20for%20recommender%20systems-challenges%20and%20remedies.pdf

## 会议总结

10.26第一次会议：

阅读两篇论文，一篇是Image Super-Resolution Using Dense Skip Connections，另一篇是A Survey on Deep Transfer Learning。在前者中我了解到了DenseNets，而在后者中我了解了迁移学习Transfer Learning。可以尝试使用DenseNets来进行特征提取，观察下效果。

导师在听完我的关于项目的设想后，指出特征提取的方法有很多，要做出自己的创意，简单地通过网络进行向量化的想法过于简单。我在反思之后，觉得应该着重两点进行研究，一是尝试新的网络，与常用的网络进行效果对比; 二是对标签的预处理，都是经过相同的网络，一个好的预处理可以得到代表性更强的向量。另外导师也提到了，关于对时间因素的创新运用，要重点去验证其效果，该模型应该与没有运用这个方法的模型和对时间因素有不同用法的模型进行比较。在会议后，我认为我应该从逻辑上分析该方法的优势在何处，增加我的方法的可解释性。


10.30第二次会议：

导师再次建议要多读论文，我也十分同意，需要了解更多研究者的方法，才能更好地完善自己的系统。另外我重新分析我的系统，发现我的系统事实上更接近与content-based rs而不是user-based CF和 item-based CF，因此我寻找论文的方向也作了修改。但CF相关的论文我认为也很有价值，我打算寻找结合这两个模型的混合系统的相关论文，看看能不能将部分CF的方法融入到CBRS中。

## 11/15工作总结
阅读了上述提到的论文，从Explainable Recommendation: A Survey and New Perspectives中了解到offline evaluation的几种常见指标和方法，可以用来分析系统的质量，文中提及大量推荐系统的方法及发展脉络，可以开阔我的思路，并且避免做其他研究者做过的研究。从A Content-Based Recommendation System Using Neuro-Fuzzy Approach我了解到了模糊逻辑的概念，我觉得这个方法可以尝试加入到我的项目中，因为对于user的口味是否发生变化的判决可以用模糊系统进行判决。由于这两周家里有点事，稍微打乱了进度，并且我思考后觉得需要用一周来具体定下系统框架，所以我对整个工作进度进行了调整：下周定下具体可行的系统框架，决定所有算法的选取，给出算法思路，之后的任务顺延。

## 11/30工作总结
梳理整个项目的流程，方便之后的代码编写：
   1. csv文件读入
   2. 数据预处理（学习pandas），区分训练集和验证集，这个按照时间顺序排列，用时间靠后的作为验证集
   3. 特征向量训练（用原生word2vec或者自己用二叉树实现，偏向后者）
   4. 分析用户购买过的商品，用聚类方法进行分类
     （先用密度聚类，寻找更优秀的聚类方法，查CNN的聚类方法，这里细节有很多，第一就是是否选择要实现定义好簇数的聚类方法，还是选择密度聚类这种
      定义密度大小然后决定最终聚类的簇数，聚类完之后，怎么判决口味是否发生变化也是个难点，聚完类之后，我们要去分析同一个类中，时间序
      列的分布，是否具有一致偏向性，算均值是最粗暴的方法，很容受极端值的影响，而且观察也不明显，多组分布的散点，各自标上不同
      的组号，如何分析这几个组在时间有明显差异，这个需要考虑，暂用均值进行判决，在阅读聚类论文，寻找更合适的指标）
   5. 判决是否有明显的类型差异，然后决定权重
   6. top k的推荐
   7. 评价方法我决定用top k推荐方法（可以寻找有无更好地content-base CF验证方法，挑选topk 个推荐，看与用户的匹配程度）
   8. 最后是一个对比，这很重要，我要得出我的优化方法优异，就必须有对比，对比的重点是，普通的content-base CF的效果，和我用w2v的效果，还有就是加不加       时间判决对最终结果的影响。

在明确了流程之后，我学习了pandas相关知识，并进行初步的编码。确定了文本读入的方法，用户user类的结构和电影movie类的结构，代码已上传。
阅读了论文 Daily-Aware Personalized Recommendation based on Feature-Level Time Series Analysis ，了解其他学者在这个方向上的研究成果。

## 12/15工作总结
1. 这两周首先对之前的LoadData函数进行调试，以便实现文档读入功能，并以正确的格式保待利用的数据，修正了User类和Movie类的结构和成员函数。
2. 是对所有的电影的tags和genres数据作为语料库，进行训练，得到可以表征一部电影的向量，参考gensim的用法，其中还阅读了这几篇博客，以便后续可能手动改写w2v：
   https://www.leiphone.com/news/201706/PamWKpfRFEI42McI.html
   https://www.jianshu.com/p/c252a4bf05a5
   https://www.jianshu.com/p/cf51bbaa289a
3. 阅读了聚类的介绍性论文：
   A Comprehensive Survey of Clustering Algorithms (2015 Springer)
来寻找合适的聚类方法。文中提及Clustering Algorithm Based on Density and Distance使我联想起之前了解过的Denpeak函数，之后就确定运用Denpeak聚类算法实现项目，参考的博客：
   https://blog.csdn.net/Leoch007/article/details/80027056#denpeak
并浏览文中提及的Denpeak原出处论文：
   Clustering by fast search and find of density peaks （2014 SCIENCE）
根据资料用Denpeak方法实现了Clustering函数。

## 1/1 工作总结
1. 这两周主要的工作是完善判决函数的逻辑，如何将已经分好类的电影和用户的评分列表相结合，判决用户是否有对特定类型的电影有偏好，并且这个偏好有无改变。
由于聚类考虑的是电影的空间远近，而评分列表则是以评分时间来排列电影，所以判决函数就是分析在空间上像近的电影在时间上的关系如何。如果用户在一段时间内
对某类电影有偏好，那么评分列表的分布应该是在一段时间内某个电影类型的电影出现频率很高。我们认为空间和时间是两个聚类，所以判决函数就是对聚类进行一个
评估。参考了 http://www.cnblogs.com/czhwust/p/mlclusterpre.html 中的Jaccard系数，决定用一个延申窗口，设置阈值，计算各类型电影的占比来进行判决，
之后要参考最大连续子数列和的算法 https://www.cnblogs.com/conw/p/5896155.html ，修改后用来找到某类型电影占比最大的子序列。
2. 修改之前聚类函数，将函数中字典逻辑修改，从类中心作为key，电影列表作为values，修改为电影作为key而类中心作为value。
在判决函数逻辑完成后，工程主要算法已经完成，之后的几天要做的就是对不同用户的评分情况进行分析，调整权重，为用户推荐电影。之后需要进行代码的调试。

## 1/15 工作总结
把最后推荐的权重分配函数Recommendation完成，整个系统各部分函数已经完成。不过程序bug还是有点多，工作重心转移到程序调试，进行了大量了的调试，修复了LoadData函数的bug，聚类函数的bug。

## 2-4月 工作总结
这两个月工作的重点是对代码进行调试和修改，然后进行实验效果的测试。代码很多地方都出现了bug，列举几个一下几个比较大的bug：
1.cal_ratio函数在边界值（用户观影数量太少时）会出错
2.dp_cal_ratio函数在边界值（用户观影数量太少时）会出错
3.Clustering函数里对密度的一个降序排序错写成升序
4.Recommendation函数进行找到top k相似电影时错找成差距最远的top k电影
还有一些边界值的小bug就不一一列举，当模型可以运行时，就开始将代码划分训练集和验证集，进行效果验证，由于我的算法与时间相关，所以我的验证集一定是用户最后观看的一定数量的电影，我选取的比例有12%，15%，18%。我重新打了一份基础的CBRS代码:Graduation_Project_v2.py与我的算法进行比较，但是发现我的优化算法没有起到优化效果，于是我分析了代码情况，得出算法没有效果的原因：
1.参考的DenPeak算法的簇中心划分效果不佳，经常出现所有电影聚成一个簇
2.用户对电影评论及电影本身特性的数据量少，提供给word2vec模型的文本过少，倒值word2vec的效果不好
3.由于word2vec的表征性不足，最后计算电影间距离进行top k推荐就会出现偏差。
根据分析出的原因，我对模型进行优化：
1.在DenPeak算法中创新性地用二次差分法来进行簇心划分
2.不再使用word2vec训练的向量表征电影，转而统计每个用户观看的电影的特征数量，对于高频特征赋予更高的权重，我们定义一部电影的权重是由它拥有特征的权重相加而成，对于一个特定用户，如果一个电影的权重越高，那么就表明用户对它越感兴趣
3.由于修改了电影的表征方式，所以电影间距离要重新定义，我的定义是两电影共有特征的权重和的倒数。对于某个用户而言，相似的电影意味着有很多共有特征；如果两个电影同时拥有几个高权重特征，那么几个特征权重的加和很高，加权和的倒数就会很小。用这个方法表示电影距离，是符合逻辑的。
在对模型进行优化之后，代码又进行一列的修改和调试，最后的得到了优化结果如下图，在图中可以看出，对于目标用户，优化算法能提高总体命中数
![Aaron Swartz](https://raw.githubusercontent.com/LGC2017/2018-Graduation-Project/master/picture/result_cmp1.png)

上述工作大致进行到3月初，之后进行开始进行论文的编写。在四月初向导师交了初稿，导师希望能突出创新点，因此，我多加了一个模型，来验证我的优化角度是可行的。我们称多加的模型为模型二，前面的为模型一。模型二与模型一主要区别在于偏好判决函数不同，模型二用的是加权法进行偏好的判决，对比用户最近的偏好与相对远一点的时间点的偏好特征是否相同来判决用户偏好是否改变。该方法同样能对系统进行优化，优化结果如下图：
![Aaron Swartz](https://raw.githubusercontent.com/LGC2017/2018-Graduation-Project/master/picture/result_cmp2.png)

最后我将代码分成三份，Graduation_Project_v1是模型一，Graduation_Project_v2是基础CBRS代码，二Graduation_Project_v3则是模型二，修改之后的论文顺利提交了。
