---
layout: post
title: Human Cognition Inspired Word Segmentation
authors: Ghosh, Shinjini (MIT)
tags: [word segmentation, cognitive learning, neural learning]  
---

1.  [Introduction to Word Segmentation](#introduction-to-word-segmentation)
2.  [Infant Statistical Learning](#infant-statistical-learning)
3.  [Neural Word Segmentation](#neural-word-segmentation)
4.  [Probabilistic Modeling](#probabilistic-modeling)
5.  [References](#references)

## Introduction to Word Segmentation

Word segmentation is the process of determining the word boundaries in free-flowing speech or non-segmented text. Language learners of all ages are able to naturally demarcate word boundaries from continuous speech, even without appreciable pauses or other linguistic cues (as mentioned in Saffran et al. 1996). So how is that human cognition allows for word segmentation within such poverty of stimulus? And is there a way we can capture the same computationally for our use in language modeling and beyond?

There have been several attempts and ongoing work for finding different algorithms that will efficiently and accurately segment any given sentence, text, or speech, either in English or in another language. One of the pioneers in this was Saffran et al., who analyzed how children performed this task from a very young age. After her study, different approaches were taken, both in the classical statistics (Brent 1999, Venkataraman 2017), Bayesian realms (Goldwater et al. 2009), and neural learning (Yang et al. 2017, Zheng et al. 2013, Pei et al. 2014, Morita et al. 2015, Chen et al. 2015, Cai and Zhao 2016, Zhang et al. 2016). In this blog, we explore neural learning methods for word segmentation, particularly Yang et al's 2017 paper, and then contrast them with probabilistic modeling. We develop on and implement some probabilistic models, and analyze their performance on the segmentation task, given different unsegmented corpora, especially in relation with human judgements.

## Infant Statistical Learning

## Neural Word Segmentation

## Probabilistic Modeling

## References
