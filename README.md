# HSG 10,860,1.00 - Introduction to Applied Deep Learning

This repository contains all colabs and links to material for the GSERM Summerschool (Winterschool) of the University of St. Gallen (HSG), Switzerland for 10,860,1.00 - Introduction to Applied Deep Learning.

This course is using [TensorFlow 2.1.0](https://www.tensorflow.org/) and [Google Colab](https://colab.research.google.com/).

## The Course Materials - January 2020

| # | Lecture | Lab |
| --- | --- | --- |
| 1 | **Introduction to Deep Learning** <br> [Lecture Slides](https://docs.google.com/presentation/d/1Uanfl1Y8qOTRXVimUkK__IgL54x1FCfk6HU_qGNCLzI/edit?usp=sharing) <br> Topics: Perceptron, Feed Forward Neural Networks, Deep Fully Connected Neural Networks, Activation and Loss Function, Optimizers, Overfitting, Dropout and other Regularization | Colab [L1 - Deep Learning Intro with mnist](https://colab.research.google.com/github/pA1nD/course-deep-learning/blob/master/L1_Deep_Learning_Intro.ipynb) |
| 2 | **Deep Computer Vision** <br> [Lecture Slides](https://docs.google.com/presentation/d/1V3VK6D_5TS3jUhOh_eiAZni4RcZoYT88chDjqDtNta4/edit#slide=id.g6d193c6e07_0_51) <br> Topics: Convolution, Subsampling, GPUs, Batchnorm, TensorBoard, Spatial Invariance, CNN Applications, Transfer Learning, ImageNet, TF Hub, TF Datasets | Colab [L2 - Convolutional Neural Networks](https://colab.research.google.com/github/pA1nD/course-deep-learning/blob/master/L2_Convolutional_Neural_Networks_v2.ipynb) <br><br> Colab [L3 - Transfer Learning](https://colab.research.google.com/github/pA1nD/course-deep-learning/blob/master/L3_Transfer_Learning.ipynb) |
| 3 | **Deep Sequence Modeling** <br> [Lecture Slides](https://docs.google.com/presentation/d/1csQk968JfM915ouHxUc2VtiHTaUh87XYRCKpZsqH7_Y/edit?usp=sharing) <br> Topics 1: Modern Convolutional Neural Networks, Inception, SqueezeNet, Keras Functional API, Data Processing with tf Datasets, TPUs, TF Production with TFX <br><br> Topics 2: Sequential Data, RNN, LSTM, GRU | Colab [L3 - Data Processing](https://colab.research.google.com/github/pA1nD/course-deep-learning/blob/master/L3_Data_Processing.ipynb) <br><br> Colab [L3 - Recurrent Neural Networks](https://colab.research.google.com/github/pA1nD/course-deep-learning/blob/master/L4_Recurrent_Neural_Networks.ipynb) |
| 4 | **Deep Natural Language Processing** <br> [Lecture Slides](https://docs.google.com/presentation/d/1T5J_mH5JiNKcdaGRY9RV2tW0YS_jtxFIcpIltc25Zp8/edit) (Guest Lecture [by ejoebstl](https://github.com/ejoebstl))<br> Topics: Language in Deep Learning, Word2Vec, Seq2Seq, Neural Machine Translation, Attention, Transformer, BERT | Colab [L4 - Word Embeddings](https://colab.research.google.com/drive/1cuXdxYepBOPcsOgbOpiYWN4Zxm9Q5eSk) with Word2Vec <br><br> Colab [L4 - Seq 2 Seq / Attention](https://colab.research.google.com/drive/1PFBBXYc76Vw158uLTWg5FYNPo_pHp7cr) for translations<br><br>Colab [L4 - Bert and Keras](https://colab.research.google.com/drive/1YLfvycRvLh2leLBZV19OgS52_tt0mZZX) with transfer learning |
| 5 | **Data Bias and Explainable Deep Learning** <br> [Lecture Slides](https://docs.google.com/presentation/d/1ASNzGLIREXHOcs4yEG9rp0Qh8sIK5F8YDgkimdx-wJU/edit?usp=sharing) <br> Topics: Recipe for Training Neural Networks, Data Bias, Deep Learning in Academic Context and for Academic Publications, Explainable Artificial Intelligence (XAI) | Colab [L5 - Interpreting Vision Models with tf-explain](https://colab.research.google.com/github/pA1nD/course-deep-learning/blob/master/L5_Interpreting_Vision_Models_with_tf_explain.ipynb).<br><br> Colab [L5 - Visualize CNN Layer Outputs and Filters](https://colab.research.google.com/github/pA1nD/course-deep-learning/blob/master/L5_Visualize_CNN_Layer_Outputs_and_Filters.ipynb) with Keras Functional API. |

## Recommended Material

**Book**

If there would be a book or resources I'd want share with you they are the following:
[Deep Learning Book](http://www.deeplearningbook.org/). You can find the [full PDF in their GH repo](https://github.com/janishar/mit-deep-learning-book-pdf/blob/master/complete-book-pdf/deeplearningbook.pdf). This is a very comprehensive book that is up to date and good to read with a valuable Bibliography.

**Courses**

MIT's [6.S191](http://introtodeeplearning.com/): Introduction to Deep Learning is a good course covering also Reinforcment Learning and GANs.
Stanford's [CS231n](https://cs231n.github.io/) is a good place to dive deeper into the mechanics of computer vision.

**Other Materials**

The (IMO) didactically best material on deep learning is [Google, Tensorflow and deep learning without a PhD series](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd), by [Martin Görner](https://twitter.com/martin_gorner). 
The [official Tensorflow Resources](https://www.tensorflow.org/overview) are well done and helpful. There are a big number of examples, tutorials and other resources available. Also helpful is their [page on additional learning resources](https://www.tensorflow.org/resources/learn-ml).

## License and Sources

**Main Sources and Credits:**

[Google, Tensorflow and deep learning without a PhD series, Martin Gorner](https://github.com/GoogleCloudPlatform/tensorflow-without-a-phd)

[MIT 6.S191: Introduction to Deep Learning](http://introtodeeplearning.com)

[Tensorflow API Documentation and Resources](http://www.tensorflow.org)

**How to use this material?**

All course materials are copyrighted by their respective author and owner. If you are an instructor and would like to use any materials from this course (slides, labs, code) for educational purposes, you must reference the original source and this repository.

This repository is open sourced under Apache-2.0 License but the license of the original author might apply.
