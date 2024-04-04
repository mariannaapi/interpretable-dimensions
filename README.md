# Interpretable Dimensions in Space

This repository contains data and code for the paper:
Katrin Erk and Marianna Apidianaki (2024). Adjusting Interpretable Dimensions in Embedding Space with Human Judgments. Accepted for publication at NAACL 2024, Mexico City, Mexico.

We share the data and scripts used in our experiments. These are organized in the following directory structure:

**vectors/**

	bert-vectors/
 		bert-large-uncased.Grandetal.top4layers.npz
   		bert-large-uncased.formality.top4layers.pkl
    		bert-large-uncased.complexity.top4layers.pkl
	
 	roberta-large-vectors/
		roberta-large.Grandetal.top4layers.pkl
 		roberta-large.formality.top4layers.pkl
  		roberta-large.complexity.top4layers.pkl

 **sentences/**
 
 	extracted_ukwac_sentences_Grandetal.pkl
  	extracted_ukwac_sentences_complexity.pkl
   	extracted_ukwac_sentences_formality.pkl
 
  **data/**
  
 	style-data/
		filtered_[complexity|formality]_human_scores

	Grand-et-al-data/
 		+++++
      
  **frequency_baseline/**
  
	Grandetal/
 		freq_rank_sorted.[animals|cities|clothing|myth|names|professions|sports|states|weather]
	style/
		freq_rank_sorted.[complexity|formality]
  

** The contents of each file are described below. **

** data/style-data/filtered_complexity_human_scores: 1,160 words with complexity annotations
** data/style-data/filtered_formality_human_scores: 1,274 words with formality annotations

Data with high annotation confidence from the Pavlick and Nenkova (2015) formality and complexity datasets. The filtering is described in Section 3.3 of our paper.   

The file contains 3 columns, as in the Pavlicka and Nenkova (2015) dataset:
- column 1: the mean of the 7 human scores on a scale from 1 to 100: 100 is most formal, 0 is most casual.
- column 2: the phrase
- column 3: the standard deviation of the human scores. A smaller number can be viewed as a higher confidence in the difference.  
        
** sentences/: Sentences from ukWaC (Baroni et al., 2009) which contain instances of words in the Grand et al., formality and complexity datasets.

** frequency baseline/
  sorted

  
=== References ===

Ellie Pavlick, Ani Nenkova (2015) Inducing Lexical Style Properties for Paraphrase and Genre Differentiation, Proceedings of NAACL 2015, Denver, Colorado, pages 218–224. 
Marco Baroni, Silvia Bernardini, Adriano Ferraresi, and Eros Zanchetta (2009) The WaCky wide web: a collection of very large linguistically processed web-crawled corpora. Journal of Language Resources and Evaluation, 43(3):209–226.
