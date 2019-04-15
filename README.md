# 多语言巴利三藏翻译模型 Multilang_Translator_For_Pali_Tripitaka


## 巴利三藏背景
巴利文大藏经(Pāli Canon)，又作巴利三藏、南传大藏经，指锡兰上座部所传之佛教大藏经，用巴利文写成，为早期佛教经典的结集。在原始佛教分裂为部派佛教后，很多重要派别都拥有按照自己观点所编集的三藏，但在目前留存下来的藏经中只有南传上座部的巴利文三藏还可看到全貌，其余派别所编集的大部分都遗失了，只能在汉译佛经中可以找出它的蛛丝马迹。

在释迦牟尼佛寂灭之后的第一年， 便由他的上首弟子摩诃迦叶主持，召集了五百长老于摩揭陀国的首都王舍城，在阿阇世王的协助之下，诵集了佛陀生前所说的教法。

佛涅槃一百年后，僧团内部便发生了分裂，主要原因是对于戒律的见解有分歧，七百位长老，两部分比丘在吠舍离举行第二次结集，从人数说名为“七百结集”。

佛灭后二百多年阿育王邀请目犍连子帝须长老来首都华氏城，并召集一千位长老，举行第三次结集，会诵出比较完整的经律论主要成分，据说今日存在的上座部巴利语三藏，就是这次大会最后编定的。

西元前一世纪末叶，锡兰僧团中的长老有鉴于国内曾发生战乱， 担心早期流传下来的教典散失，由以坤德帝沙长老为首的大寺派的五百名长老，于斯里兰卡中部马特列地区的阿卢迦寺举行南传佛教历史上的第四次结集，会诵集结三藏教典，以僧伽罗文字将经典写在贝叶上成书，首次制巴利语三藏辑录成册。

1871 年，缅甸国王敏东在曼德勒召开上座部佛教第五次结集，有二千四百人参加，用三年多时间重新校对巴利文大藏经。同时建立了一片塔林，叫古道陀石经院，每座塔里有一块石碑，每块石碑上刻篇佛经，把这次新校对的巴利文大藏经，全部刻在七二〇座石经塔上，使佛典得到长期保存。

1954至1956年，缅甸政府在首都仰光举行一次上座部佛教史上规模最大的第六次结集，出席者有缅甸、泰国、斯里兰卡、柬埔寨、老挝、印度、巴基斯坦等国的长老比丘二千五百人，根据各国的各种版本，对巴利语的经、律、论三藏典籍，进行了一次非常严密的校勘，并且决定把这次校勘的典籍全部陆续地刊印出来，作为现代世界上最有权威的巴利语大藏经新版本。

## 多语言佛经翻译模型

[巴利语大藏经平行语料](https://github.com/x39826/Pali_Tripitaka)

这里使用的翻译系统是开源的神经网络机器翻译系统 [OpenNMT](http://opennmt.net/), 最初由哈佛大学NLP组开发，它集成许多最新的神经网络机器翻译模型与算法，用于 学术研究和工业开发，在翻译评测中表现非常优异。由于多语言翻译的需要，我们修改了OpenNMT的部分代码，并添加了一些新的模型功能，主要包括:

1. 添加了新的模型输出模块[AdaSoftmaxGenerator](https://github.com/x39826/Multilang_Translator_For_Pali_Tripitaka/blob/master/OpenNMT_py/onmt/modules/AdaSoftmaxGenerator.py)，它是针对多语言翻译超大词表设计的一个树状Softmax输出层，可以节省模型训练时的运行内存和加快翻译时的解码速度。
2. 添加了新的模型优化器[sparseadam](https://github.com/x39826/Multilang_Translator_For_Pali_Tripitaka/blob/master/OpenNMT_py/onmt/utils/optimizers.py)，它是一个多种优化算法的结合体，针对多语言翻译模型大词表问题，在模型不同部位的参数上使用不同的优化算法，以平衡优化算法的时间和物理资源开销。
3. 添加了数据处理脚本[sample.py](https://github.com/x39826/Multilang_Translator_For_Pali_Tripitaka/blob/master/sample.py)，用于生成多语言翻译的训练数据和词表。修改了解码器模块[-trans_to]，实现了指定语言方向的翻译功能。

## 翻译系统训练与测试
**data preprocess and model training**
```
sh shell.sh
```
**model evaluation and local file translation (with/without specified target language)**
```
python test.py
```
**test the online translation API**
```
python test_sever.py
```
