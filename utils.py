from transformers import Trainer
import torch
from torch.nn import CrossEntropyLoss
import torch
from typing import List

def genTrainingExamplesTrans(sentence_pairs, sentencesInSameParagraph):
    negative_examples = []
    for i,s in enumerate(sentencesInSameParagraph):
        start = 0
        pos = s.find(" ", start)
        while(pos != -1):
            left_same_sentence = s[0:pos]
            right_same_sentence = s[pos+1:]
            left_previous_sentences = " ".join( sentencesInSameParagraph[0:i] )
            right_next_sentences = " ".join( sentencesInSameParagraph[min(i+1,len(sentencesInSameParagraph)-1):len(sentencesInSameParagraph)-1])
            start = pos + 1
            negative_examples.append([(left_previous_sentences + " " + left_same_sentence).strip(), (right_same_sentence + " " + right_next_sentences).strip()])
            pos = s.find(" ", start)

    positive_examples = []
    for i,pair in enumerate(sentence_pairs):
        left_same_pair = pair[0]
        right_same_pair = pair[1]
        left_previous_pairs = " ".join( sentencesInSameParagraph[0:i] )
        right_next_pairs = " ".join( sentencesInSameParagraph[min(i+2,len(sentencesInSameParagraph)-1):len(sentencesInSameParagraph)-1])
        positive_examples.append([(left_previous_pairs + " " + left_same_pair).strip(), (right_same_pair + " " + right_next_pairs).strip()])

    return positive_examples, negative_examples


class WeightedTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.class_weight = kwargs["class_weights"]
        del kwargs["class_weights"]
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss_fct = CrossEntropyLoss(weight=torch.tensor(self.class_weight).type_as(logits))
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))


        return (loss, outputs) if return_outputs else loss


def label_cut_noCut(sentences:List[str]) -> list:
    """
    input: `sentences` is list of string (each string is one sentence)
    returns: list of strings '<cut>' or '<no_cut>'
    """
    # label each space as cut/not cut
    cut_notCut_label = []
    for i, s in enumerate(sentences): # each sentence in label
        # the next code line does this for the sentence s:
        # 1. remove leading/trailing space
        # 2. split by the space
        # 3. find the length of the split, the number of spaces is this length-1, this the number of <no_cut> in this sentence
        var1 = ["<no_cut>" for _ in range( len(s.strip().split(" "))-1 ) ]
        if i < len(sentences)-1: # add a <cut> if not the last sentence of paragraph
            cut_notCut_label += var1 + ["<cut>"]
        else: # last sentence of paragraph, no need to cut
            cut_notCut_label += var1

    return cut_notCut_label
