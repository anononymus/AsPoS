# AsPOS: Assamese Parts of Speech Tagger using Deep Learning Approach

This repository contains pre-trained Assamese POS tagging model as well as dataset for Assamese POS. The model is trained using BiLSTM-CRF architecture with MuRIL+Flair embeddings. The model acheived tagging accuracy of F1-score 86.52%.

## Annotated Assamese POS tagged dataset 

The dataset has been annonated by an automatic POS tagger of which the accuracy is 74.62%. After that, it is manually corrected. The dataset is splitted into three parts for model training, those are train.txt, dev.txt and test.txt.

## How to run

Download the pre-trained model from the link- [AsPOS](https://drive.google.com/file/d/1LAi6cZMyRFWoB6uYIWp3CPtTTnfnOCfx/view?usp=sharing)

```
from flair.models import SequenceTagger
from flair.data import  Sentence, Token

# Load the tagger

model = SequenceTagger.load('AsPOS.pt')

#  create example sentence
sen='মাতৃভাষাৰ সমান্তৰালকৈ সংস্কৃত, ইংৰাজী ভাষাৰ চৰ্চা অত্যন্ত জৰুৰী ৷'
sentence = Sentence(sen)

# predict tags and print
model.predict(sentence)
print(sentence.to_tagged_string()
