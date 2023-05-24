def bleu_score(candidate, references):
    """
    Compute the BLEU score given a candidate and references
    :param candidate: list of tokens
    :param references: list of list of tokens
    :return: BLEU score dict
    """
    from nltk.translate.bleu_score import sentence_bleu
    bleu1_score=sentence_bleu(references,candidate,weights=(1,0,0,0))
    bleu2_score=sentence_bleu(references,candidate,weights=(0,1,0,0))
    bleu3_score=sentence_bleu(references,candidate,weights=(0,0,1,0))
    bleu4_score=sentence_bleu(references,candidate,weights=(0,0,0,1))
    return [bleu1_score,bleu2_score,bleu3_score,bleu4_score]


def rouge_score(candidate, references):
    """
    Compute the ROUGE score given a candidate and references
    :param candidate: list of tokens
    :param references: list of list of tokens
    :return: ROUGE score dict
    """


def bert_score(candidate, references):
    """
    Compute the BERT score given a candidate and references
    :param candidate: list of tokens
    :param references: list of list of tokens
    :return: BERT score dict
    """
def bart_score(candidate):
    """
    Compute the BART score given a candidate
    :param candidate: list of tokens
    :return: BART score dict
    """
    from bart_score import BARTScorer
    bart_scorer = BARTScorer(device='cuda', checkpoint='facebook/bart-large-cnn')
    tokens = ' '.join(map(str, candidate)) #make all the word into a sentence
    tokens=tokens+'.' #a sentence need a period
    return bart_scorer.score(['Such as'], [tokens],batch_size=4)


def idf_bleu_score(candidate,reference,token):
    
    """
    Compute the idf_BLEU score given a candidate and references
    :param candidate: list of tokens
    :param references: list of list of tokens
    :param token: number of token in a gram
    :return: BLEU score dict
    """
    import re
    import json
    score=[]
    with open("./raceidf.json",'r',encoding='utf-8')as f:
        idf_lib=json.load(f)
    for one_reference in reference:
        print(reference)
        reference_ngrams = [one_reference[i:i+token] for i in range(len(one_reference)-(token-1))]
        candidate_ngrams = [candidate[i:i+token] for i in range(len(candidate)-(token-1))]
        print(reference_ngrams)
        print(candidate_ngrams)
        candidate_totscore=[]
        for i in candidate_ngrams:
            tmp_score=0
            for j in i:
                if j not in idf_lib.keys():
                    idf_lib[j]=1
                tmp_score+=idf_lib[j]
            candidate_totscore.append(tmp_score)
        count = 0
        for ngram in range(len(candidate_ngrams)):
            count += reference_ngrams.count(candidate_ngrams[ngram])*candidate_totscore[ngram]
            
        try:
            bleu_score = count / sum(candidate_totscore)
        except:
            bleu_score=0
        score.append(bleu_score)
    if len(score)==0:
        return 0
    return max(score)


class UnifiedQA_Scorer:
    def __init__(self, model_config="allenai/unifiedqa-v2-t5-small-1363200"):
        from transformers import T5Tokenizer, T5ForConditionalGeneration
        model_name = model_config
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)

    def option_score(self, qainput, options):
        import torch
        import numpy as np

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum(axis=0)

        def score(model, tokenizer, input_sent, label_sents):
            tensor_input = tokenizer.encode(input_sent, return_tensors='pt')
            scores = []
            for label_sent in label_sents:
                label = tokenizer.encode(label_sent, return_tensors='pt')
                with torch.inference_mode():
                    loss = model(tensor_input, labels=label).loss
                scores.append(-np.exp(loss.item()))
            return softmax(scores)

        return score(self.model, self.tokenizer, qainput, options)
