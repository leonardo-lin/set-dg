def bleu_score(candidate, references):
    """
    Compute the BLEU score given a candidate and references
    :param candidate: list of tokens
    :param references: list of list of tokens
    :return: BLEU score dict
    """


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
