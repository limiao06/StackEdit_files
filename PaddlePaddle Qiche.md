# 如何用PaddlePaddle做摘要

本博客记录了如何使用百度PaddlePaddle框架实现了一个seq2seq模型，并侥幸获得“PaddlePaddle AI 产业应用赛——汽车大师问答摘要与推理”比赛一等奖的过程。 

## 背景
一个偶然的机会，发现了一个AI比赛——[PaddlePaddle AI 产业应用赛——汽车大师问答摘要与推理](https://www.kesci.com/apps/home/competition/5aec0eb10739c42faa203931)。这个比赛要求选手们使用汽车大师所提供的11万条技师与用户的多轮对话与诊断建议报告数据建立模型，从而可基于对话文本、用户问题、车型与车系，输出包含摘要与推断的报告文本，考验模型的归纳总结与推断能力。

看到这个比赛后，“多轮对话”数据第一时间吸引了我的注意，先申请参赛看看数据再说。 这个比赛基于[科赛](https://www.kesci.com/)平台，完成注册后，可以在科赛为你提供的云端虚拟机环境中看到数据，但是没有办法把数据下载到本地，( ⊙ o ⊙ )！ 这么狠，搞数据的想法破灭了，哭。 

既然报名了，就尝试做一下吧，这毕竟是实际商业场景中产生的真实多轮对话数据。 数据产生的场景是这样的，一些车主在自己的爱车遇到问题时，在汽车大师App上发起提问，专业技师会根据问题（Problem）和用户进行一段对话（Conversation），从而帮住用户解决问题，最后技师根据问题和对话生成一个报告(Report)。 如下图所示。
**需要图， 从项目中截图**

这个比赛的任务就是给定问题（Problem）和对话内容（Conversation），自动生成报告（Report）， 通过计算算法生成的report和标准答案的ROUGE分数，评估算法的性能。 这个比赛的数据集由10万多条样本的训练集和各5000条样本的开发集和测试集组成。

## 思考

如何解决这个问题呢？ 那么要把这个问题抽象一下， 有下面几个角度：
1. 从Problem 和 Report的关系看， Report是要给出Problem的解答，所以这是个QA问题啊？
2. 从Conversation 和 Report 的关系看， Report是对Conversation 内容的总结和提炼， 所以这是个文本摘要问题啊？
3. 结合Problem， Conversation 和Report 一起看，好像是根据问题，从Conversation中提取答案的过程， 所以这是个阅读理解问题啊？

到底哪一种理解方式好呢？ 我还真通过简单的验证方法进行了尝试：

### 角度1：QA问题
QA问题的一种解决办法是进行问题匹配， 面对一个问题q，从训练集中选择出最相似的问题q'， 然后把q'的答案a'作为答案返回。 为了简单验证一下， 我写了一个简单的baseline，基于tfidf 计算问题相似度，选择训练集中最相似问题的report作为答案返回。 提交！ 一看结果排行榜， 得分14.5，排名倒数第二位， 不要太惨。。。 仔细思考了一下， 可能存在这种情况：虽然提出的问题很像，但是实际情况各有不同，需要通过对话进一步找到真正的问题所在，对话中的重要信息一点都不用生成的report是好不到哪里去。

### 角度2： 摘要问题
摘要问题我之前没有实际做过啊，怎么快速明确这个思路好不好呢？ 这里感谢Markus同学分享的[开源项目](https://www.kesci.com/apps/home/project/5af51a65cb6ed25ca3279186)， 这里他尝试了一个比较粗暴的摘要方式，直接把Conversation中技师说的第一句话作为结果返回（在自动摘要问题中，段落的第一句话通常是比较重要的话）。这个简单粗暴的方法得分是多少？  32.3286分， 我的天！ 看来这个思路比QA要靠谱。

### 角度3： 阅读理解问题
如果看成是阅读理解问题， 那么就是从Conversation中找出能回答Problem的答案， 由于目前的阅读理解数据集的答案长度通常比较短（一般是几个单词），所以state of the art的作法是根据Problem，从Context中选择一段作为答案，模型只要输出答案的开始和结束位置即可。 但是这个任务的report有点长，常常出现几十个甚至上百个词， 而且report中的词好像并不完全是来自于Conversation。 我写了个程序统计了一下， Report中67.7%的词来自于Conversation， 这个比例虽然不低，但是还是让我放弃了从Conversation中选择一段作为Report的方法。

### 我的思路
我更想把这个问题看成是摘要问题， Report是Conversation的摘要，但是是由Problem指导的摘要。 所以我觉得设计一个seq2seq的网络结构， 根据Problem和Conversation， 以生成式（而非抽取式）的方法，生成Report。 在算法上，我很大程度参考了[Teaching Machines to Read and Comprehend](https://arxiv.org/pdf/1506.03340.pdf) 这篇论文

## 动手 

## 结果

## 心得

> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJwcm9wZXJ0aWVzIjoidGl0bGU6IOWmguS9leeUqFBhZGRsZV
BhZGRsZeWBmuaRmOimgVxuYXV0aG9yOiBNaWFvXG50YWdzOiAn
RGVlcExlYXJuaW5nLFBhZGRsZVBhZGRsZSxTZXEyU2VxJ1xuY2
F0ZWdvcmllczogRExcbiIsImhpc3RvcnkiOlstMTUwNTQ4OTk4
NywyOTA0NjMyMywxNDc2MDg4NDg5LDcxOTI3ODI5MSwtMjAxMz
AwOTEzMywtMjE3MDQ0MTMwLC01ODQ3MTkxMjAsLTEyNDQyMDcw
MjEsMTY2NzgwNzU2LC0yNDM5NTQ1Nl19
-->