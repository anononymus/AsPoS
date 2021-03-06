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
sen='ফুকন বসুমতাৰী এজন অধ্য়াপক । তেওঁ বৰ্তমান কোকৰাঝাৰত থাকে ।'
sentence = Sentence(sen)
# predict tags and print
model.predict(sentence)
print(sentence.to_tagged_string())
ফুকন <N_NNP> বসুমতাৰী <N_NN> এজন <QT_QTF> অধ্য়াপক <N_NN> । <RD_PUNC> তেওঁ <PR_PRP> বৰ্তমান <RB> 
কোকৰাঝাৰত <N_NNP> থাকে <V_VM> । <RD_PUNC>

#  create example sentence
sen='মাতৃভাষাৰ সমান্তৰালকৈ সংস্কৃত, ইংৰাজী ভাষাৰ চৰ্চা অত্যন্ত জৰুৰী ৷'
sentence = Sentence(sen)
# predict tags and print
model.predict(sentence)
print(sentence.to_tagged_string()
মাতৃভাষাৰ <N_NN> সমান্তৰালকৈ <N_NN> সংস্কৃত <N_NNP> , <RD_PUNC> ইংৰাজী <N_NNP> ভাষাৰ <N_ANN> চৰ্চা <N_NN> অত্যন্ত <RP_INTF> 
জৰুৰী <N_NN> ৷ <RD_PUNC>

```
