import torch
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pickle
import sys 
import os
import argparse
import re
import tqdm
import pdb


def careful_tokenization(sentence, tokenizer, model_name, maxlen):
    bert_positions_list = []
    tok_sent = ['[CLS]']
    for orig_token in sentence: 
        number_bert_tokens = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token) 

        if len(tok_sent) + len(bert_token) >= maxlen:
            break

        tok_sent.extend(bert_token)             
        if len(bert_token) > 1:                 
            extra = len(bert_token) - 1
            for i in range(extra):
                number_bert_tokens.append(number_bert_tokens[-1]+1)         
                
        bert_positions_list.append(tuple(number_bert_tokens))

    tok_sent.append('[SEP]')

    return tok_sent, bert_positions_list


def check_correct_token_mapping(bert_tokenized_sentence, positions, word, tokenizer):
    # put together the pieces corresponding to the positions
    tokenized_word = list(tokenizer.tokenize(word))
    # check if they correspond to the word 
    berttoken = []
    bert_positions = positions

    for p in bert_positions:
        berttoken.append(bert_tokenized_sentence[p])
    if berttoken == tokenized_word:
        return True
    else:
        return False



def aggregate_wordpieces(reps_list): 
    reps = torch.zeros([len(reps_list), 1024])              
    for i, wrep in enumerate(reps_list):
        w, rep = wrep
        reps[i] = rep

    if len(reps) > 1:
        reps = torch.mean(reps, dim=0)
    try:
        reps = reps.view(1024)
    except RuntimeError:
        pdb.set_trace()
    return reps



def extract_representations(infos, tokenizer, model_name, maxlen):
    reps = []
    if model_name in ["bert-large-uncased", "bert-large-cased", "bert-base-uncased", "bert-base-cased"]:                       
        config_class, model_class = BertConfig, BertModel

    config = config_class.from_pretrained(model_name, output_hidden_states=True,max_position_embeddings=maxlen)
    model = model_class.from_pretrained(model_name, config=config)

    model.eval()
    with torch.no_grad():
        for info in tqdm.tqdm(infos, total=len(infos)):
            tok_sent = info['bert_tokenized_sentence']
            input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tok_sent)]).to(device)
            inputs = {'input_ids': input_ids}
            outputs = model(**inputs)

            hidden_states = outputs[2]
            bpositions = info["bert_position"]
            reps_for_this_instance = dict()
            
            for i, w in enumerate(info["bert_tokenized_sentence"]):
                if i in bpositions:                                             
                    for l in range(len(hidden_states)): 
                        if l not in reps_for_this_instance:
                            reps_for_this_instance[l] = []

                        reps_for_this_instance[l].append((w, hidden_states[l][0][i].cpu()))

            reps.append(reps_for_this_instance)            

    return reps



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--sentences", type=str, required=True, help="fill with sentences to extract representations from")
    parser.add_argument("--modelname", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True, help="where to save the representations")
    # parser.add_argument("--cased", action="store_true", help="use cased or uncased model?")

    # Usage: python extract-bert-representations.py --sentences extracted_ukwac_sentences.pkl --modelname bert-large-uncased --output_dir bert_embeddings

    args = parser.parse_args()
    sentences = args.sentences
    model_name = args.modelname
    out_reps = args.output_dir

    # if args.cased:
    #     do_lower_case=False
    # else:   
    #     do_lower_case=True

    # if args.cased:  
        # model_name = model + '-cased'
    # else: 
        # model_name = model + '-uncased'

    tokenizer = BertTokenizer.from_pretrained(model_name) 

    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1
    if "large" in model_name:
        toplayer = 13       
    elif "base" in model_name:
        toplayer = 25

    
    maxlen = 512

    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, sentences)

    data = pickle.load(open(file_path, "rb"))

    wordsinsentences = []
    word_count = 0
    sentences_per_word = {}             

    infos = []
    final_representations = {}

    for word in data:
        word_count += 1
        sentence_list = data[word]
        sentences_per_word[word] = len(sentence_list)
        sentenceid = 0

        if sentence_list:
            for sentence in sentence_list:
                info = dict()
                sentenceid += 1
                info["word"] = word
                info["sentence_id"] = sentenceid
                sentence_tokens = sentence['sentence_words']
                info["sentence"] = sentence_tokens
                info["position"] = sentence['position']

                bert_tokenized_sentence, mapp = careful_tokenization(info["sentence"], tokenizer, model_name,maxlen=maxlen)
                            
                info["bert_tokenized_sentence"] = bert_tokenized_sentence
 
                if not len(info["position"]) > 1:
                    try:
                        bert_position = mapp[info["position"][0]]                
                    except IndexError:
                        pdb.set_trace()
                    info["bert_position"] = bert_position

                    if not check_correct_token_mapping(bert_tokenized_sentence, bert_position, info["word"], tokenizer):
                        print("Position mismatch!")
                        pdb.set_trace()
                else:
                    if len(info["position"]) == 2:
                        bert_position1 = mapp[info["position"][0]] 
                        bert_position2 = mapp[info["position"][1]]
                        bert_position = bert_position1 + bert_position2
                        info["bert_position"] = bert_position
                        

                infos.append(info)


    print("EXTRACTING REPRESENTATIONS...")
    reps = extract_representations(infos, tokenizer, model_name, maxlen=maxlen)
    print("...DONE")

    if len(reps) != len(infos):
         print("Serious mismatch")
         pdb.set_trace()
    

    aggregated_vectors = dict()
    
    for rep, instance in zip(reps, infos):
        bposition = instance['bert_position']
        target_word = instance['word']

        selected_layers = []

        for layer_number in rep:
            if layer_number in [21, 22, 23, 24]:            
                k = "rep-"+str(layer_number)

                representation = aggregate_wordpieces(rep[layer_number])
                selected_layers.append(representation)
            
        aggr_keys = aggregated_vectors.keys()
 
        if target_word in aggregated_vectors:
            aggregated_vectors[target_word].append(selected_layers)
        else:
            aggregated_vectors[target_word] = [selected_layers]


    for w in aggregated_vectors:
        tensors_from_examples = aggregated_vectors[w]


        mean_layer_reps_for_each_sentence = {}
        for sentence_tensors in tensors_from_examples:             
            
            stacked_tensors = torch.stack(sentence_tensors, dim=0)

            mean_representation = torch.mean(stacked_tensors, dim=0)
        
            if w in mean_layer_reps_for_each_sentence:
                mean_layer_reps_for_each_sentence[w].append(mean_representation)
            else:
                mean_layer_reps_for_each_sentence[w] = [mean_representation]

        number_mean_reps = len(mean_layer_reps_for_each_sentence[w])
        list_mean_reps = mean_layer_reps_for_each_sentence[w]
        length_list_mean_reps = len(list_mean_reps)

        stacked_tensors_mean_reps = torch.stack(list_mean_reps, dim=0)
        
        final_representation = torch.mean(stacked_tensors_mean_reps, dim=0)
        final_representations[w] = final_representation

output = './bert-embeddings'
os.mkdir(output)

np.savez((os.path.join(output, 'bert-large-uncased.npz')), **final_representations)









