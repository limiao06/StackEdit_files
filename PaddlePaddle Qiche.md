# 如何用PaddlePaddle做摘要

本博客记录了如何使用百度PaddlePaddle框架实现了一个seq2seq模型，并侥幸获得“PaddlePaddle AI 产业应用赛——汽车大师问答摘要与推理”比赛一等奖的过程。

## 背景
一个偶然的机会，发现了一个AI比赛——[PaddlePaddle AI 产业应用赛——汽车大师问答摘要与推理](https://www.kesci.com/apps/home/competition/5aec0eb10739c42faa203931)。这个比赛要求选手们使用汽车大师所提供的11万条技师与用户的多轮对话与诊断建议报告数据建立模型，从而可基于对话文本、用户问题、车型与车系，输出包含摘要与推断的报告文本，考验模型的归纳总结与推断能力。

看到这个比赛后，“多轮对话”数据第一时间吸引了我的注意，先申请参赛看看数据再说。 这个比赛基于[科赛](https://www.kesci.com/)平台，完成注册后，可以在科赛为你提供的云端虚拟机环境中看到数据，但是没有办法把数据下载到本地，( ⊙ o ⊙ )！ 这么狠，搞数据的想法破灭了，哭。 

既然报名了，就尝试做一下吧，这毕竟是实际商业场景中产生的真实多轮对话数据。 数据产生的场景是这样的，一些车主在自己的爱车遇到问题后，在汽车大师App上发起提问，专业技师会根据问题（Problem）和用户进行一段对话（Conversation），从而帮住用户解决问题，然后需要根据问题和对话生成一个报告(Report)。 如下图所示。
**需要图**

这个比赛的任务就是给定问题（Problem）和对话内容（Conversation），自动生成报告（Report）。



## 思考

## 动手 

## 结果

## 心得

> Written with [StackEdit](https://stackedit.io/).
<!--stackedit_data:
eyJwcm9wZXJ0aWVzIjoidGl0bGU6IOWmguS9leeUqFBhZGRsZV
BhZGRsZeWBmuaRmOimgVxuYXV0aG9yOiBNaWFvXG50YWdzOiAn
RGVlcExlYXJuaW5nLFBhZGRsZVBhZGRsZSxTZXEyU2VxJ1xuY2
F0ZWdvcmllczogRExcbiIsImhpc3RvcnkiOlstMTI0NDIwNzAy
MSwxNjY3ODA3NTYsLTI0Mzk1NDU2XX0=
-->