---
layout: post
title: Human Cognition Inspired Word Segmentation
authors: Ghosh, Shinjini (MIT)
tags: [word segmentation, cognitive learning, neural learning]  
---

## Contents

1.  [Introduction to Word Segmentation](#introduction-to-word-segmentation)
2.  [Infant Learning](#infant-learning)
3.  [Neural Word Segmentation](#neural-word-segmentation)
4.  [Probabilistic Modeling](#probabilistic-modeling)
5.  [References](#references)

## Introduction to Word Segmentation

Word segmentation is the process of determining the word boundaries in free-flowing speech or non-segmented text. Language learners of all ages are able to naturally demarcate word boundaries from continuous speech, even without appreciable pauses or other linguistic cues (as mentioned in Saffran et al. 1996). So how is that human cognition allows for word segmentation within such poverty of stimulus? And is there a way we can capture the same computationally for our use in language modeling and beyond?

There have been several attempts and ongoing work for finding different algorithms that will efficiently and accurately segment any given sentence, text, or speech, either in English or in another language. One of the pioneers in this was Saffran et al., who analyzed how children performed this task from a very young age. After her study, different approaches were taken, both in the classical statistics (Brent 1999, Venkataraman 2017), Bayesian realms (Goldwater et al. 2009), and neural learning (Yang et al. 2017, Zheng et al. 2013, Pei et al. 2014, Morita et al. 2015, Chen et al. 2015, Cai and Zhao 2016, Zhang et al. 2016). In this blog, we explore neural learning methods for word segmentation, particularly Yang et al's 2017 paper, and then contrast them with probabilistic modeling. We develop on and implement some probabilistic models, and analyze their performance on the segmentation task, given different unsegmented corpora, especially in relation with human judgements.

### Motivation

We believe that current state-of-the-art language models, which fail on natural language understanding and inference tasks, could benefit with human-inspired augmentations and that an improved word segmentation algorithm would further the current NLP frontiers in the capabilities of neural and non-neural language models, especially because most models currently in place crucially reply on segmenting words correctly. Distilling the knowledge gained from our understanding of human cognition into computational models and human-like intelligent systems could go a long way in overall improved neural learning models.

## Infant Learning

Saffran et al.'s groundbreaking paper delves into statistical learning by 8-month-old infants, and aims to probe one of the very basic human cognitive tasks, a fundamental task accomplished by almost every child in the world---segmentation of words from fluent speech. The authors state that 'successful' word segmentation by the infants, based on only 2 minutes of speech exposure, suggests that they have access to a powerful mechanism for computing the statistical properties of language input. This is a very important observation in building computational models of cognition regarding word segmentation, especially when coupled with the fact that there exists complex and widely varying acoustic structure of speech in different languages and hence, there is no invariant acoustic cue to word boundaries present in all languages.

Saffran et al.'s observations are crucial as we set out to use knowledge of how human intelligence works in order to build more human-like intelligence systems. Neural network models have been exploited due to their strength in non-sparse representation learning and non-linear power in feature combination, resulting in multiple neural word segmentors with comparable accuracies to statistical models. Character embeddings (Zheng et al. 2013), character bigrams (Mansur et al. 2013 and Pei et al. 2014), and words (Morita et al. 2015 and Zhang et al. 2016) leverage non-sparse representations to improve segmentation. On the other hand, non-linear modeling power has been leveraged in multiple neural learning based word segmentation paper---such as multi-layer perceptrons in Zheng et al. 2013, Mansur et al. 2013, Pei et al. 2014, Chen et al. 2015; LSTMs on characters in Chen et al. 2015, Xu and Sun 2016; LSTMs on words in Morita et al. 2015, Cai and Zhao 2015, Zhang et al. 2016. All these models leverage salient features of deep neural networks in word segmentation. On the other hand, as outlined in Goodman et al. 2014, Piantadosi et al. 2012, Piantadosi et al. 2016, and multiple other papers, the probabilistic language of thought hypothesis believes that concepts have a language-like compositionality and encode probabilistic knowledge, thereupon relying on Bayesian inference for production. We also look at how Goldwater et al. 2009 approach the word segmentation problem probabilistically, relying on word context. Finally, we try to revisit Saffran et al.'s experiment, this time computationally.

## Neural Word Segmentation

Word segmentation literature has fairly recented shifted its focus from the statistical methods to deep learning technologies. Various salient aspects of deep learning methods have been exploited in the context of word representation, such as using character embeddings as a foundation of neural word segmentors to reduce sparsity of character n-grams. The non-linear modeling power of neural network structures have been leveraged to represent contexts for segmentation disambiguation. NLP tasks ranging from parsing to sentence generation have uniformly benefited from pre-training, the human-inspired concept of  training a model with one task to help it form parameters that can be used in other tasks. The old knowledge helps new models successfully perform new tasks from experience instead of from scratch. However, there existed a gap in the literature on using pretraining for word segmentation tasks, which Yan et al's 2017 paper aims to fulfil. Their model is conceptually simple and modular, so that the most important sub module, namely a five-character window context, can be pretrained using external data. The authors adopt a multi-task learning strategy (Collobert et al., 2011), casting each external source of information as a auxiliary classification task, sharing a five-character window network. After pretraining, the character window network is used to initialize the corresponding module in the word segmentor. This method outperforms the best statistical and neural segmentation models consistently, giving thebest reported results on 5 datasets in different domains and genres.

### Model Structure

The word segmentor works incrementally from left to right, as the example shown in Table 1. At each step, the state consists of a sequence of words that have been fully recognized, denoted as $W = [w_{-k}, w_{-k+1}, \dots, w_{-1}]$, a current partially recognized word $P$, and a sequence of next incoming characters, denoted as $C = [c_0, c_1, \dots, c_m]$, as shown in Figure 1 below.

<a href="https://imgur.com/81XupE2"><img src="https://i.imgur.com/81XupE2.png" title="source: imgur.com" /></a>

<a href="https://imgur.com/4lstKWR"><img src="https://i.imgur.com/4lstKWR.png" title="source: imgur.com" /></a>

The model starts by initializing $W$ and $P$ as $[]$ and $\Phi$, and $C$ with all the input characters. At each step, the model makes a decision on $c_0$, either deciding to add it as a part of the current word $P$, or demarcating it as the beginning of a new word. This incremental process repeats until $C$, the character bank, is empty, and $P$ is null again. This process can be formally represented as a state-transition process, where a state is a tuple $S = (W, P, C)$ and the transition actions include `Sep` (separate) and `App` (append), as is demonstrated in the deduction system in Figure 2. An end-of-sentence marker is also used. In the figure, $V$ denotes the score of a state, given by a neural network model. The score of the initial state is 0, and the score of other states is the sum of scores of all incremental decisions resulting in the state. The overall score is used for disambiguating states, which correspond to sequences of inter-dependent transition actions.

<a href="https://imgur.com/NURdJqK"><img src="https://i.imgur.com/NURdJqK.png" title="source: imgur.com" /></a>

### Scoring Network
The scoring network used in this model consists of three main layers. The bottommost layer is a representation layer, deriving dense representations $X_W, X_P, X_C$ for $W, P$ and $C$, respectively. The next layer is a hidden layer, used to merge $X_W, X_P, X_C$ into a single vector using $h = tanh(W_{hW} \cdot X_W + W_{hP} \cdot X_P + W{hC} \cdot X_c + b_h)$. This hidden feature $h$ is used to represent the next state $S=(W,P,C)$ for calculating the scores for the next action. A linear output layer with two nodes is used for this with $o=W_o\cdot h+b_o$. The two nodes of $o$ represent the scores of `Sep` and `App`, given $S$, respectively.

### Pretraining
The three basic elements in the neural word segmentor, namely characters, character bigrams and words, can all be pretrained over large unsegmented data. The authors pretrain the five character window network in Figure 3 as an unit, learning the MLP parameter together with character and bigram embeddings. They consider four types of commonly explored external data, all of which have been studied for statistical word segmentation, but not for neural network segmentors.

<a href="https://imgur.com/35Z7n4R"><img src="https://i.imgur.com/35Z7n4R.png" title="source: imgur.com" /></a>

### Decoding and Training
To train the main segmentor, the authors adopt the global transition-based learning and beam-search strategy of Zhang and Clark (2011), as shown in Algorithm 1.

<a href="https://imgur.com/J3vT7hF"><img src="https://i.imgur.com/J3vT7hF.png" title="source: imgur.com" /></a>

For decoding, a standard beam search is used, where the $B$ best partial output hypotheses at each step are maintained in an agenda initialized with the start state. At each step, all hypotheses in the agenda are explored, by applying all the possible actions and the $B$ highest scoring hypotheses are used as the agenda for the next step. For training, the same decoding process is applied to each training example $(x^i, y^i)$, along with an early update strategy. The authors use Adagrad to optimize the model parameters, and use L2 regularization and dropout to reduce overfitting. Character and character bigram embeddings are fine-tuned, but not word embeddings, according to Zhang et al.'s 2016 conclusion. The hyper-parameter values are shown in Table 2.

<a href="https://imgur.com/irhdUAg"><img src="https://i.imgur.com/irhdUAg.png" title="source: imgur.com" /></a>

### Experiments
The authors use Chinese Treebank 6.0 (CTB6) (Xue et al., 2005) as their main dataset, along with the SIGHAN 2005 bake-off (Emerson, 2005) and NLPCC 2016 shared task for Weibo segmentation (Qiu et al., 2016). Chinese gigaword is used for pretraining embedding and it is automatically segmented using ZPar 0.6 (Zhang and Clark, 2007). Standard word precision, recall and F1 are used as evaluation metrics. The authors also perform development experiments to verify the usefulness of various context representations (by varying $X_C$ and $X_W$, network configurations (by measuring the influence of beam size on the baseline segmentor) and different pretraining methods (using punctuation methods, etc.)

The F1 measures are in Tables 4 and 5 below.

<a href="https://imgur.com/k2aQeqV"><img src="https://i.imgur.com/k2aQeqV.png" title="source: imgur.com" /></a>

The final results are as in Tables 7 and 8. The final results compare favorably to existing statistical models, and this is the first time a pure neural network model outperforms all existing methods on this dataset, showcasing the helpfulness of pretraining and deep learning methodologies, as well as being a pioneering work in the area. This model also outperforms the best existing neural models, including a hybrid of a statistical and neural model as in Zhang et al. 2016b.

<a href="https://imgur.com/dyc1a6G"><img src="https://i.imgur.com/dyc1a6G.png" title="source: imgur.com" /></a>

<a href="https://imgur.com/QIlnZDZ"><img src="https://i.imgur.com/QIlnZDZ.png" title="source: imgur.com" /></a>

### Conclusion
The authors have leveraged rich extenral resources to pretrain a deep learning model, thereby yielding a more enhanced word segmentation model which utilises both character and word contexts. The authors use neural multi-task learning to pre-train a set of shared parameters for character contexts. The results indicate a 15.4% relative error reduction and are highly competitive to both statistical and neural state-of-the-art models on six different benchmarks.

## Probabilistic Modeling
To help demonstrate the advantages of this new pretrained deep learning model, we have implemented two statistical methods applied on data obtained from Saffran et al.'s classic 1996 paper.

### Probabilistic Context Free Grammar

We modify the usual PCFG setup as follows. Instead of having an input sentence, we have an input speech stream, segmented into syllables. We assume that the smallest part of speech that infants can discern without external knowledge is syllables (Saffran et al. also take tri-syllabic words and look at probability transitions between word boundaries in infants), and our concern is how they break this syllable stream into words. We then segment the speech stream aka 'Sentence' into words, which further break into more words or a single word, which break into syllable(s). Our PCFG thus looks as below.

```python=
""" 
Sentence -> Words [1.0]
Words -> Word Words [0.8] | Word [0.2]
Word -> Syllables [1.0]
Syllables -> Syllable Syllables [0.8] | Syllable [0.2]
Syllable -> 'tu' [0.083]
Syllable -> 'pi' [0.168]
Syllable -> 'ro' [0.083]
Syllable -> 'go' [0.083]
Syllable -> 'la' [0.168]
Syllable -> 'bu' [0.083]
Syllable -> 'da' [0.083]
Syllable -> 'ko' [0.083]
Syllable -> 'ti' [0.083]
Syllable -> 'du' [0.083] 
"""
```

Just like Saffran et al., we generate speech stream by randomly concatenating words from the input vocabulary (of 2 minutes = 180 words). The syllable probabilities are then inferred from the speech stream, and the word/syllable break probabilities are a parameter that we tweak and see the results with. We then investigate the various word parses (and corresponding) that these PCFGs give us, as well as the probabilities of those parses. A sample parse tree with high probability is shown below---it shows us how given a stream of syllables, our PCFG breaks down the input `da ro pi go la tu` into two words `daropi` and `golatu`. This is one of the "hard" input examples for the vocabulary consisting of the words `pigola`, `golatu`, and `daropi`, because the 'part-word' `pigola` spanned the boundary between `daropi\#golatu`. Below that, we have another parse tree---one with a low probability assigned by the parser, and clearly not adequate. We hypothesize that humans have access to such computing mechanism, and select a high probability parse tree to use in their daily lives.

<a href="https://imgur.com/b8swcsj"><img src="https://i.imgur.com/b8swcsj.png" title="source: imgur.com" /></a>

<a href="https://imgur.com/EaYRI1E"><img src="https://i.imgur.com/EaYRI1E.png" title="source: imgur.com" /></a>

We also try four different parsers and do a comparative analysis of the time taken by them to compute all parses of the input `da ro pi go la tu`, in an attempt to compare with human reaction times. We receive the following values, as shown below. Saffran et al. gave the infants a much higher threshold of 2 seconds to judge word familiarity.

<a href="https://imgur.com/pvysB0v"><img src="https://i.imgur.com/pvysB0v.png" title="source: imgur.com" /></a>

### Dynamic Programming with Probabilistic n-gram Modeling

In addition to the PCFG modeling, we then go on to other probabilistic modeling techniques for word segmentation. We use a model implemented along the lines of Peter Norvig's in the book 'Beautiful Data'. We use Norvig's pre-processed version of the Google Trillion Word Dataset distributed through the Linguistics Data Consortium. This dataset is trimmed of n-grams occurring lower than 40 times, unkified, and sentence demarcations are added. It readily gives us unigram and bigram probabilities, from which we can compute the conditional probabilities as well. A snapshot of the bigram counts data used is below.

<a href="https://imgur.com/aikKO8X"><img src="https://i.imgur.com/aikKO8X.png" title="source: imgur.com" /></a>

We use two probabilistic models---one based on unigrams and the other on bigrams. We recursively split a stream of text, computing the Naive Bayes probability of the sequence of words thus formed, and use dynamic programming to memoize our computation, preventing us from running into exponential times. The Bayes probability is computed using a probability distribution estimated from the counts in the pre-processed data files, and Laplace additive smoothing is used to estimate the probability of unknown words. We also use surprisal values for the bigram model. Finally, the segmentation with the highest probability, or the lowest surprisal, is chosen as our output segmentation.

We create a unit test file, with segmentations of text stream, both straightforward and ambiguous, e.g., 'choosespain' can be segmented both as 'choose spain' or 'chooses pain'. If we believe that humans use statistical inference, then we can assume that the former would be more probable than the latter, based on conditional probability counts of the true. This is a hypothesis we test in our model, and it indeed turns out to be true. Overall, while this model performs well, there remain controversies as to how well such models relate to human cognitive processes.

While English employs word spacing, which makes word segmentation from written corpora fairly easy, Japanese does not (and neither do Mandarin, Cantonese and agglutinative languages). To make our model more comparable to the neural model for Chinese shown above, We trained a bigram model on Wikipedia Japanese data and tested it on Zhang Lang's corpus. The results are much poorer, hovering around 80\% in accuracy metrics. Overall, we note how deep learning methods have resulted in leaps and bounds of progress in fields typically dominated by statistical methods, and how in particular, the word segmentation task in natural language processing has been performed better by neural methods.

## References

1. J. R. Saffran, R. N. Aslin, and E. L. Newport. (1996). Statistical learning by 8-month-old infants. Science, 274(5294):1926 – 1928, 1996.
2. J. Yang, Y. Zhang, and F. Dong. (2017). Neural Word Segmentation with Rich Pretraining. Association for Computational Linguistics.
3. M. Brent. (1999). An Efficient, Probabilistically Sound Algorithm for Segmentation and Word Discovery. Machine Learning, 34. doi: 10.1023/A:1007541817488.
4. S. Goldwater, T. L. Griffiths, and M. Johnson. (2009). A bayesian framework for word segmentation: Exploring the effects of context. Cognition, 112(1):21 – 54. ISSN 0010-0277. doi: 10.1016/j.cognition.2009.03.008.
5. N. D. Goodman, J. Tenenbaum, and T. Gerstenberg. (2014). Concepts in a Probabilistic Language of Thought.
6. S. T. Piantadosi, J. B. Tenenbaum, and N. D. Goodman. (2012). Bootstrapping in a language of thought: a formal model of numerical concept learning. Cognition, 123(2):199–217. doi: doi/10.1016/j.cognition.2011.11.005.
7. A. Venkataraman. (2001). A Statistical Model for Word Discovery in Transcribed Speech. Computational Linguistics, 27(3):352–372, Sept. 2001. ISSN 0891-2017.
8. X. Zheng, H. Chen, and T. Xu. (2013). Deep learning for chinese word segmentation and pos tagging. In EMNLP. Association for Computational Linguistics, pages 647–657.
9. Pei, W., Ge, T., & Chang, B. (2014). Max-Margin Tensor Neural Network for Chinese Word Segmentation. In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Asso ciation for Computational Linguistics. doi:doi.org/10.3115/v1/p14-1028
10. Morita, H., Kawahara, D., & Kurohashi, S. (2015). Morphological Analysis for Unsegmented Languages using Recurrent Neural Network Language Model. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics. doi:doi.org/10.18653/v1/d15-1276
11. Chen, X., Qiu, X., Zhu, C., Liu, P., & Huang, X. (2015). Long Short-Term Memory Neural Networks for Chinese Word Segmentation. In Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics. doi:doi.org/10.18653/v1/d15-1141
12. Cai, D., & Zhao, H. (2016). Neural Word Segmentation Learning for Chinese. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics. doi:doi.org/10.18653/v1/p16-1039
13. Zhang, M., Zhang, Y., & Fu, G. (2016). Transition-Based Neural Word Segmentation. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics. doi:doi.org/10.18653/v1/p16-1040
14. Mansur, M., Pei, W., Change, B. (2013). Feature-based Neural Language Model and Chinese Word Segmentation. In Proceedings of the Sixth International Joint Conference on Natural Language Processing. Association for Computational Linguistics.
15. Chen, X., Qiu, X., Zhu, C., & Huang, X. (2015). Gated Recursive Neural Network for Chinese Word Segmentation. In Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers). Association for Computational Linguistics. doi:doi.org/10.3115/v1/p15-1168
16. Andor, D., Alberti, C., Weiss, D., Severyn, A., Presta, A., Ganchev, K., Petrov, S., & Collins, M. (2016). Globally Normalized Transition-Based Neural Networks. In Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics. doi:doi.org/10.18653/v1/p16-1231
17. Xue, N., Xia, F., Chiou, F., and Palmer, M. (2005). The penn chinese treebank: Phrase structure annotation of a large corpus. Natural language engineering 11(02):207–238.
18. Xue, N. (2003). Chinese word segmentation as character tagging. Computational Linguist.
19. Zhang , Y., & Clark, S. (2007). Chinese Segmentation with a Word-Based Perceptron Algorithm. In Proceedings of the 45th Annual Meeting of the Association of Computational Linguistics. Association for Computational Linguistics. 
20. Zhang , Y., & Clark, S. (2007). Joint Word Segmentation and POS Tagging Using a Single Perceptron. In Proceedings of ACL-08: HLT. Association for Computational Linguistics. 
21. Zhang, Y., & Clark, S. (2011). Syntactic Processing Using the Generalized Perceptron and Beam Search. In Computational Linguistics (Vol. 37, Issue 1, pp. 105–151). MIT Press - Journals. doi:doi.org/10.1162/coli_a_00037
22. Xia, Q., Li, Z., Chao, J., Zhang, M. (2016). Word segmentation on micro-blog texts with external lexicon and heterogeneous data. In International Conference on Computer Processing of Oriental Languages. Springer





