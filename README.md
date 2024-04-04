# Interpretable Dimensions in Space

This repository contains data and code for the paper:
Katrin Erk and Marianna Apidianaki (2024). Adjusting Interpretable Dimensions in Embedding Space with Human Judgments. Accepted for publication at NAACL 2024, Mexico City, Mexico.

We share the data and scripts used in our experiments. These are organized in the following directory structure:

**Vectors/**

  bert-vectors/
      bert-large-uncased.complexity.top4layers.pkl
	    bert-large-uncased.formality.top4layers.pkl
	    bert-large-uncased.top4layers.Grandetal.npz
  
  roberta-large-vectors/
      roberta-large.Grandetal.top4layers.pkl
  	  roberta-large.formality.top4layers.pkl
  	  roberta-large.complexity.top4layers.pkl
 
  filtered_sentenes_ukwac/
      extracted_ukwac_sentences_Grandetal.pkl
    	extracted_ukwac_sentences_complexity.pkl
  	  extracted_ukwac_sentences_formality.pkl
 
  style-data/
      complexity/
          human/
              filtered_complexity_human_scores
      formality/
          human/
              filtered_formality_human_scores
      
  frequency_baseline/
      Grandetal/
          sorted/
      style/
          sorted/
  

** The contents of each file are described below. **

** style-data/complexity/human/filtered_complexity_human_scores
Data with high annotation confidence from the Pavlick and Nenkova (2015) complexity dataset. The filtering is described in Section 3.3 of our paper.   

filtered_complexity_human_scores: contains 1,160 words with complexity annotations
The file contains 3 columns, as in the Pavlicka and Nenkova (2015) dataset:
- column 1: the mean of the 7 human scores on a scale from 1 to 100: 100 is most formal, 0 is most casual.
- column 2: the phrase
- column 3: the standard deviation of the human scores. A smaller number can be viewed as a higher confidence in the difference.  
    
** style-data/formality/human/filtered_formality_human_scores
Data with high annotation confidence from the Pavlick and Nenkova (2015) formality datasets. The filtering is described in Section 3.3 of our paper.   
    
filtered_formality_human_scores: contains 1,274 words with formality annotations
The file contains 3 columns, as in the Pavlicka and Nenkova (2015) dataset:
- column 1: the mean of the 7 human scores on a scale from 1 to 100: 100 is most formal, 0 is most casual.
- column 2: the phrase
- column 3: the standard deviation of the human scores. A smaller number can be viewed as a higher confidence in the difference. 

- filtered_sentences_ukwac/
    Sentences from ukWaC (Baroni et al., 2009) which contain instances of words in the Grand et al., formality and complexity datasets.

- frequency baseline/
  sorted

  
=== References ===

Ellie Pavlick, Ani Nenkova (2015) Inducing Lexical Style Properties for Paraphrase and Genre Differentiation, Proceedings of NAACL 2015, Denver, Colorado, pages 218–224. 
Marco Baroni, Silvia Bernardini, Adriano Ferraresi, and Eros Zanchetta (2009) The WaCky wide web: a collection of very large linguistically processed web-crawled corpora. Journal of Language Resources and Evaluation, 43(3):209–226.
