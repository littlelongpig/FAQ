# !/usr/bin/env python
# _*_ coding utf-8 _*_
# @Time     :2020/6/3 15:24
# @Author   : HL
# @Title    :main
import jieba
from gensim.models import word2vec, KeyedVectors
import logging


def word_process(sentence):
    """
    输出分词和去停用词
    :param sentence: 输入的句子
    :return:
    """
    seglist = jieba.lcut(sentence.strip())
    # print(seglist)
    stopwords = [line.strip() for line in
                 open('D:\workbase\FAQ\SimpleFAQ\dataset\哈工大停用词表.txt',
                      encoding='UTF-8').readlines()]
    # print(len(stopwords))
    output_words = []
    for word in seglist:
        if word not in stopwords:
            output_words.append(word)
    return output_words


def load_dataset():
    dataset = []
    question_corpus = []
    data = open('D:\workbase\FAQ\SimpleFAQ\dataset\自建问答对.txt', encoding='UTF-8').readlines()
    for line in data:
        line = line.strip()
        if not len(line) or line.startswith('#'):
            continue
        dataset.append(line.split())
    dictionary = dict(dataset)
    question_list = list(dictionary.keys())[1:]
    for i in question_list:
        question_corpus.append(word_process(i))
    return dictionary, question_corpus, question_list


def output_corcus(sentences):
    """
    输出问题语料
    :param sentence:
    :return:
    """
    with open('D:\workbase\FAQ\SimpleFAQ\dataset\question_corcus.txt', 'w+',
              encoding='utf-8') as f:
        for i in range(len(sentences)):
            for j in range(len(sentences[i])):
                f.write(str(sentences[i][j]))
                f.write('\t')
            f.write('\n')


def save_model_model():
    """
    保存模型第一种方式：model.save(), .modelg格式
    :return: model
    """
    sentences = list(word2vec.LineSentence('D:\workbase\FAQ\SimpleFAQ\dataset\question_corcus.txt'))
    model = word2vec.Word2Vec(sentences, min_count=1)
    model.save('D:\workbase\FAQ\SimpleFAQ\dataset\question.model')
    return model


def save_model_vector():
    """
    保存模型第一种方式：model.wv.save_word2vec_format() ，.vector 和 .bin
    :return: model
    """
    sentences = list(word2vec.LineSentence('D:\workbase\FAQ\SimpleFAQ\dataset\question_corcus.txt'))
    model = word2vec.Word2Vec(sentences, min_count=1)
    model.wv.save_word2vec_format('D:\workbase\FAQ\SimpleFAQ\dataset\question.vector')
    return model


def qa(new_question, model):
    dictionary, question_corpus, question_list = load_dataset()
    # 分词
    question_cut = word_process(new_question)
    print(question_cut)
    similarity = []
    for index, question in enumerate(question_corpus):
        sim = model.wv.n_similarity(question_cut, question)
        if not sim:
            return '你所提问题找不到答案'
        similarity.append(sim)
    # print(similarity)
    max_similarity = max(similarity)
    # print(max_similarity)
    index = similarity.index(max_similarity)
    question = question_list[index]
    print(question)
    answer = dictionary.get(question)
    return answer


# 增量训练，只有用save()方法保存的.model格式模型才有用
def add_train(new_sentence, model):
    sentences_cut = []
    for i in new_sentence:
        sentences_cut.append(word_process(i))
    model.build_vocab(sentences_cut, update=True)  # 注意update = True 这个参数很重要
    model.train(sentences_cut, total_examples=model.corpus_count, epochs=10)
    model.save('D:\workbase\FAQ\SimpleFAQ\dataset\question.model')
    return model


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # 输出语料
    dictionary, question_corpus, question_list = load_dataset()
    output_corcus(question_corpus)
    # 保存两种模型
    save_model_model()
    save_model_vector()
    # 加载模型
    model1 = KeyedVectors.load_word2vec_format('D:\workbase\FAQ\SimpleFAQ\dataset\question.vector', binary=False)
    print(model1)
    model2 = word2vec.Word2Vec.load("D:\workbase\FAQ\FAQrobot-master\long_faq\dataset\question.model")
    print(model2)
    question = '怎么在百度创建应用？'
    answer = qa(question, model1)
    print(answer)
