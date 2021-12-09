---
layout: post
title: Human Cognition Inspired Word Segmentation
authors: Ghosh, Shinjini (MIT)
tags: [word segmentation, cognitive learning, neural learning]  
---

## Contents

1.  [Introduction to Word Segmentation](#introduction-to-word-segmentation)
2.  [Infant Statistical Learning](#infant-statistical-learning)
3.  [Neural Word Segmentation](#neural-word-segmentation)
4.  [Probabilistic Modeling](#probabilistic-modeling)
5.  [References](#references)

## Introduction to Word Segmentation

Word segmentation is the process of determining the word boundaries in free-flowing speech or non-segmented text. Language learners of all ages are able to naturally demarcate word boundaries from continuous speech, even without appreciable pauses or other linguistic cues (as mentioned in Saffran et al. 1996). So how is that human cognition allows for word segmentation within such poverty of stimulus? And is there a way we can capture the same computationally for our use in language modeling and beyond?

There have been several attempts and ongoing work for finding different algorithms that will efficiently and accurately segment any given sentence, text, or speech, either in English or in another language. One of the pioneers in this was Saffran et al., who analyzed how children performed this task from a very young age. After her study, different approaches were taken, both in the classical statistics (Brent 1999, Venkataraman 2017), Bayesian realms (Goldwater et al. 2009), and neural learning (Yang et al. 2017, Zheng et al. 2013, Pei et al. 2014, Morita et al. 2015, Chen et al. 2015, Cai and Zhao 2016, Zhang et al. 2016). In this blog, we explore neural learning methods for word segmentation, particularly Yang et al's 2017 paper, and then contrast them with probabilistic modeling. We develop on and implement some probabilistic models, and analyze their performance on the segmentation task, given different unsegmented corpora, especially in relation with human judgements.

<a href="https://imgur.com/81XupE2"><img src="https://i.imgur.com/81XupE2.png" title="source: imgur.com" /></a>

### Motivation

We believe that current state-of-the-art language models, which fail on natural language understanding and inference tasks, could benefit with human-inspired augmentations and that an improved word segmentation algorithm would further the current NLP frontiers in the capabilities of neural and non-neural language models, especially because most models currently in place crucially reply on segmenting words correctly. Distilling the knowledge gained from our understanding of human cognition into computational models and human-like intelligent systems could go a long way in overall improved neural learning models.

## Infant Statistical Learning

Saffran et al.'s groundbreaking paper delves into statistical learning by 8-month-old infants, and aims to probe one of the very basic human cognitive tasks, a fundamental task accomplished by almost every child in the world---segmentation of words from fluent speech. The authors state that 'successful' word segmentation by the infants, based on only 2 minutes of speech exposure, suggests that they have access to a powerful mechanism for computing the statistical properties of language input. This is a very important observation in building computational models of cognition regarding word segmentation, especially when coupled with the fact that there exists complex and widely varying acoustic structure of speech in different languages and hence, there is no invariant acoustic cue to word boundaries present in all languages.

Saffran et al.'s observations are crucial as we set out to use knowledge of how human intelligence works in order to build more human-like intelligence systems. Neural network models have been exploited due to their strength in non-sparse representation learning and non-linear power in feature combination, resulting in multiple neural word segmentors with comparable accuracies to statistical models. Character embeddings (Zheng et al. 2013), character bigrams (Mansur et al. 2013 and Pei et al. 2014), and words (Morita et al. 2015 and Zhang et al. 2016) leverage non-sparse representations to improve segmentation. On the other hand, non-linear modeling power has been leveraged in multiple neural learning based word segmentation paper---such as multi-layer perceptrons in Zheng et al. 2013, Mansur et al. 2013, Pei et al. 2014, Chen et al. 2015; LSTMs on characters in Chen et al. 2015, Xu and Sun 2016; LSTMs on words in Morita et al. 2015, Cai and Zhao 2015, Zhang et al. 2016. All these models leverage salient features of deep neural networks in word segmentation. On the other hand, as outlined in Goodman et al. 2014, Piantadosi et al. 2012, Piantadosi et al. 2016, and multiple other papers, the probabilistic language of thought hypothesis believes that concepts have a language-like compositionality and encode probabilistic knowledge, thereupon relying on Bayesian inference for production. We also look at how Goldwater et al. 2009 approach the word segmentation problem probabilistically, relying on word context. Finally, we try to revisit Saffran et al.'s experiment, this time computationally.

## Neural Word Segmentation

<a href="https://imgur.com/4lstKWR"><img src="https://i.imgur.com/4lstKWR.png" title="source: imgur.com" /></a>

<a href="https://imgur.com/NURdJqK"><img src="https://i.imgur.com/NURdJqK.png" title="source: imgur.com" /></a>

## Probabilistic Modeling

## References
