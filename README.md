# FAQ
A nlp procession of question and answer
这个程序是根据faq(常见问题回答)为基础来做的demo，思想是将问答对生成字典，并将问题集进行训练成模型，当新问题提出时，通过与模型进行相似度比较，返回一个相似度最大的问题，通过问答字典返回答案提供给用户。
注意：语料是我自己建的一个很小的问答对语料，它所能提供训练的词向量只是很小一部分，因为新问题经分词后在模型中找不到某个词的向量，就会报错。为了使得模型效果更好，建议大家将自己的训练集越大越好，本程序用的只是自建小语料，大家可以下载更大的语料库。


# 1. 数据预处理
   - 分词
   - 去停词
   - 生成问题语料
# 2. 训练模型和加载模型
   - 使用问题语料word2vec训练模型并保存模型
   - 加载模型
# 3. 测试新问题，并输出答案
   - 将新问题分词并与语料库对比，找到相似问题
   - 根据相似问题得到答案
