# Interpretable Dimensions in Space

This repository contains data and code for the paper:

Katrin Erk and Marianna Apidianaki (2024). Adjusting Interpretable Dimensions in Embedding Space with Human Judgments. Accepted for publication at NAACL 2024, Mexico City, Mexico.

If you use our code and data, please make sure to cite this paper:

```
@misc{erk2024adjusting,
      title={Adjusting Interpretable Dimensions in Embedding Space with Human Judgments}, 
      author={Katrin Erk and Marianna Apidianaki},
      year={2024},
      eprint={2404.02619},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

We share the data and scripts used in our experiments. These are organized in the following directory structure:

**Vectors**

 	vectors/
		bert-vectors/
 			bert-large-uncased.Grandetal.top4layers.npz
   			bert-large-uncased.formality.top4layers.pkl
      		bert-large-uncased.complexity.top4layers.pkl
	
 		roberta-large-vectors/
			roberta-large.Grandetal.top4layers.pkl
 			roberta-large.formality.top4layers.pkl
  			roberta-large.complexity.top4layers.pkl

To run the models with GloVe vectors (Pennington et al., 2014), you need to make sure to include glove.42B.300d.zip in a separate 'vectors/glove' folder. 
You can download GloVe embeddings from this page: https://nlp.stanford.edu/projects/glove/

 **Sentences**
 
 	Sentences
  		ukwac_sentences_Grandetal.pkl
  		ukwac_sentences_complexity.pkl
   		ukwac_sentences_formality.pkl

Sentences from ukWaC (Baroni et al., 2009) which contain instances of words in the Grand et al. (2022), formality and complexity datasets.

**Extract contextualized representations**
  	
	extract_contextualized
 		extract_contextualized_representations.py

Extract contextualized BERT or RoBERTa representations from retained ukWaC sentences (in sentences/ folder). Usage example: 

```
python extract-contextualized-representations.py --sentences ukwac_sentences_complexity.pkl --modelname roberta-large --output_dir roberta_embeddings
```

In order to exactly reconstitute the results in the paper, special tokens need to be included when extracting BERT representations but not for RoBERTa. 

  **Data**
  
 	data/
  		style-data/
			filtered_[complexity|formality]_human_scores

		Grand-et-al-data/
 			category_feature.csv files (for all category-feature pairs)
   			features.xlsx


**Frequency_baseline**
  
	frequency_baseline/
 		Grandetal/
 			sorted
   			unsorted
 			freq_rank_sorted.[animals|cities|clothing|myth|names|professions|sports|states|weather]
		style/
 			sorted
   				freq_rank_sorted.[complexity|formality]
   			unsorted
     			freq_rank_unsorted.[complexity|formality]

Log-transformed frequency counts of words in the Google N-gram corpus (Brants and Franz, 2006). 

  
**References**

- Gabriel Grand, Idan Asher Blank, Francisco Pereira and Evelina Fedorenko (2022) Semantic projection recovers rich human knowledge of multiple object features from word embeddings. Nature Human Behaviour.
- Ellie Pavlick and Ani Nenkova (2015) Inducing Lexical Style Properties for Paraphrase and Genre Differentiation. NAACL. 
- Marco Baroni, Silvia Bernardini, Adriano Ferraresi, and Eros Zanchetta (2009) The WaCky wide web: a collection of very large linguistically processed web-crawled corpora. Journal of Language Resources and Evaluation, 43(3):209–226.
- Thorsten Brants and Alex Franz (2006) Web 1T 5-gram Version 1. In LDC2006T13, Philadelphia, Pennsylvania. Linguistic Data Consortium.


**Acknowledgments**

This research is supported in part by the Office of the Director of National Intelligence (ODNI), Intelligence Advanced Research Projects Activity (IARPA), via the HIATUS Program contract #2022-22072200005. The views and conclusions contained herein are those of the authors and should not be interpreted as necessarily representing the official policies, either expressed or implied, of ODNI, IARPA, or the U.S. Government. The U.S. Government is authorized to reproduce and distribute reprints for governmental purposes notwithstanding any copyright annotation therein.




