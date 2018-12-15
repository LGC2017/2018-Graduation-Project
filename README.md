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
