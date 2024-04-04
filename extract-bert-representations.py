import torch
from transformers import BertTokenizer, BertModel, BertConfig
import numpy as np
import pickle
import sys 
import argparse
import re
import tqdm
import pdb


def careful_tokenization(sentence, tokenizer, model_name, maxlen):
    bert_positions_list = []
    tok_sent = ['[CLS]']
    for orig_token in sentence: #.split():
        number_bert_tokens = [len(tok_sent)]
        bert_token = tokenizer.tokenize(orig_token) # tokenize

        ##### check if adding this token will result in >= maxlen (=, because [SEP] goes at the end). If so, stop
        if len(tok_sent) + len(bert_token) >= maxlen:
            break

        tok_sent.extend(bert_token)             # append the new token(s) to the tokenized sentence
        if len(bert_token) > 1:                 # if the word has been split into multiple wordpieces
            extra = len(bert_token) - 1
            for i in range(extra):
                number_bert_tokens.append(number_bert_tokens[-1]+1)         # list of new positions of the target word in the new tokenisation
                
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
        print('berttoken == tokenized_word', berttoken, tokenized_word)
        return True
    else:
        print('berttoken != tokenized_word', berttoken, tokenized_word)
        return False



def aggregate_wordpieces(reps_list): 
    reps = torch.zeros([len(reps_list), 1024])              # len(reps_list) -> how many tokens in the sentence
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

    # ex., python extract-bert-representations-Nov25.py --sentences extracted_ukwac_sentences.pkl --modelname bert-large --output_dir bert_embeddings

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

    tokenizer = BertTokenizer.from_pretrained(model_name) # do_lower_case=do_lower_case)
    # torch.set_default_tensor_type(torch.cuda.FloatTensor)

    torch.manual_seed(0)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    batch_size = 1
    if "large" in model_name:
        toplayer = 13       
    elif "base" in model_name:
        toplayer = 25

    # 25 layers; the input hidden state (final hidden state of the embeddings) is included when using pytorch_transformers (but not for pytorch_pretrained_bert) => give easy access to the embeddings + 
    # facilitate probing work using the output of the embeddings as well as the hidden states
    
    maxlen = 512

    path ='/Users/marianna/Documents/NSF-Katrin/jupyter-notebooks/representations_extracted_Nov26/'
    file_location = path + sentences
    data = pickle.load(open(path + sentences, "rb"))

    wordsinsentences = []
    word_count = 0
    sentences_per_word = {}             # how many sentences per word are available

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
                # mapp is the bert_positions_list returned by careful_tokenization function
                            
                info["bert_tokenized_sentence"] = bert_tokenized_sentence
                #### bert_position_list : the list with all bert positions in tuples [(1,), (2,), (3,), (4,), (5,), (6,), (7,), (8,), (9,), (10,), (11,), (12, 13, 14, 15, 16), (17,)]
 
                if not len(info["position"]) > 1:
                    try:
                        bert_position = mapp[info["position"][0]]                # even if there is 1 position, it's a tuple, e.g., (1,)
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

    print('len reps >> ', len(reps), 'len infos', len(infos))
    if len(reps) != len(infos):
         print("Serious mismatch")
         pdb.set_trace()
    

    aggregated_vectors = dict()
    # get representations of target words using their position 
    
    for rep, instance in zip(reps, infos):
        bposition = instance['bert_position']
        target_word = instance['word']

        selected_layers = []

        for layer_number in rep:
            if layer_number in [16, 21, 22, 23, 24]:            # layers: [16, 21, 22, 23, 24]; in transformers, 25 layers are output, the 24 bert layers + the embedding layer
                k = "rep-"+str(layer_number)

                representation = aggregate_wordpieces(rep[layer_number])
                selected_layers.append(representation)
            
            # mean of representations from layers 16 and top 4 in bert-large (i.e. 16, 21, 22, 23, 24)

        aggr_keys = aggregated_vectors.keys()
 
        if target_word in aggregated_vectors:
            # mean of representations from layers 16 and top 4 in bert-large [16, 21, 22, 23, 24]
            # aggregated_vectors[target_word].append(torch.mean(selected_layers, dim=0))
            aggregated_vectors[target_word].append(selected_layers)
        else:
            aggregated_vectors[target_word] = [selected_layers]


    for w in aggregated_vectors:
        tensors_from_examples = aggregated_vectors[w]
        # print('length tensors_from_examples ===> ', len(tensors_from_examples))      # = number of sentences

        # first average over the 5 retained layers for each sentence; then average over sentences to get the representation for a word

        # mean_representation_across_layers = torch.zeros(1, 1024])
        mean_layer_reps_for_each_sentence = {}
        for sentence_tensors in tensors_from_examples:
            
            # for layer_tensor in sentence_tensors:                   
            # Stack tensors along dimension 0
            # Print shapes before stacking for tensor in l_tensor list
                # print(f"Tensor shape before stacking: {layer_tensor.shape}")

            stacked_tensors = torch.stack(sentence_tensors, dim=0)

            # Calculate the mean along dimension 0
            mean_representation = torch.mean(stacked_tensors, dim=0)
        
            if w in mean_layer_reps_for_each_sentence:
                mean_layer_reps_for_each_sentence[w].append(mean_representation)
            else:
                mean_layer_reps_for_each_sentence[w] = [mean_representation]

        number_mean_reps = len(mean_layer_reps_for_each_sentence[w])
        list_mean_reps = mean_layer_reps_for_each_sentence[w]
        length_list_mean_reps = len(list_mean_reps)

        stacked_tensors_mean_reps = torch.stack(list_mean_reps, dim=0)
        
        # Calculate the mean along dimension 0
        final_representation = torch.mean(stacked_tensors_mean_reps, dim=0)
        final_representations[w] = final_representation

np.savez('/Users/marianna/Documents/NSF-Katrin/jupyter-notebooks/bert_embeddings/bert-large-uncased.npz', **final_representations)



