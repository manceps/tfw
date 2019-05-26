# Tensorflow World 2019

At [TensorFlow.world 2019](https://conferences.oreilly.com/tensorflow/tf-ca), on Oct 28-31 Al Kari, Garrett Lander, and Hobson Lane from Manceps will be presenting our experience with NLP at scale for long technical documents, like medical records. We'll also show how to use that pipeline to unredact the Mueller Report at Portland Meetups May 29 & 30:

- [Wed May 29 at Vacasa](http://bit.ly/tfnw-052919)
- [Thurs May 30 at New Relic](https://www.meetup.com/pdxpython/events/gmxlbqyzhbfc)

Many of the techniques we use are explained in detail in [**N**atural **L**anguage **P**rocessing **i**n **A**ction](http://bit.ly/gh-readme-nlpia-book). There's a 50% discount on the book Mon & Tues, May 27 & 28: wm052419au


## Unredacting the Mueller Report

We'll show you how to train an RNN to predict the next words or sentences and use it to predict the text in the redactions (black boxes) of the Mueller Report. Then we'll show you how to use transfer learning in Keras with the state-of-the-art BERT language model to improve the accuracy of this unredaction pipeline.

We built this on the shoulders of a lot of good people contributing code and data to a valiant effort to improve US government transparency:

- [Open Source Mueller Report (machine-readable latex)](http://opensourcemuellerreport.com/)
- [Factbase's human-reviewed text](https://f2.link/mr-sheet)
- [Ian Landis Miller](https://github.com/iandennismiller/mueller-report)
- [Zhao HG's port of BERT to Keras](https://github.com/CyberZHG/keras-bert)
- [Sepehr Sameni's port of BERT to Keras](https://github.com/Separius/BERT-keras)
- [Gaden Buie's improved OCR of the PDF](https://github.com/totalgood/gadenbuie-mueller-report)
- [Manuel Amunategui's RNN for Generating Mueller](http://www.viralml.com/video-content.html?v=_DexQhQB8uI&Title=Generate%20Robert%20Mueller%20with%20TF%202.0,%20Keras,%20GRU,%20TPU,%20For%20Free%20and%20Under%205%20Minutes)
- [Paul Mooney's OCR of the PDF report](https://www.kaggle.com/paultimothymooney/mueller-report)

## Learned Semantic Distance Metrics

## Deep Learning Adversarial Attack and Defense

## Abstractive Summarization of Long Technical Documents
