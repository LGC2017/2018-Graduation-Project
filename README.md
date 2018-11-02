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

A supervised learning approach to build a recommendation system for user-generated content in a casual game (2017 IEEE)

A review on deep learning for recommender systems: challenges and remedies (2018 Springer Nature)

Explainable Recommendation: A Survey and New Perspectives

## 会议总结

10.26第一次会议：

阅读两篇论文，一篇是Image Super-Resolution Using Dense Skip Connections，另一篇是A Survey on Deep Transfer Learning。在前者中我了解到了DenseNets，而在后者中我了解了迁移学习Transfer Learning。可以尝试使用DenseNets来进行特征提取，观察下效果。

导师在听完我的关于项目的设想后，指出特征提取的方法有很多，要做出自己的创意，简单地通过网络进行向量化的想法过于简单。我在反思之后，觉得应该着重两点进行研究，一是尝试新的网络，与常用的网络进行效果对比; 二是对标签的预处理，都是经过相同的网络，一个好的预处理可以得到代表性更强的向量。另外导师也提到了，关于对时间因素的创新运用，要重点去验证其效果，该模型应该与没有运用这个方法的模型和对时间因素有不同用法的模型进行比较。在会议后，我认为我应该从逻辑上分析该方法的优势在何处，增加我的方法的可解释性。

10.30第二次会议：

导师再次建议要多读论文，我也十分同意，需要了解更多研究者的方法，才能更好地完善自己的系统。另外我重新分析我的系统，发现我的系统事实上更接近与content-based rs而不是user-based CF和 item-based CF，因此我寻找论文的方向也作了修改。但CF相关的论文我认为也很有价值，我打算寻找结合这两个模型的混合系统的相关论文，看看能不能将部分CF的方法融入到CBRS中。

