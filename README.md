# unsupervised-NMT
This is an open source implementation of our framework for unsupervised NMT with weight sharing, which is described in the following papers:

"Unsupervised Neural Machine Translation with Weight Sharing, ACL2018".

Requirements: Tensorflow 1.2.0, python 2.x
  
 Note: 
 
 Our model is based on the state-of-the-art Transformer.
 
 This is a flexible and general framework and it is easy to add any modules to our framework.
 
  Beyond the published paper, we also find that the fixed word embedding is a bottleneck for improving the performance of unsupervised NMT. To handle this problem, we propose the learnable word embeddings beyond the fixed embedding. Experiments show that the learnable word embeddings can achieve up to +1.0 BLEU points improvement on En-De translation. Moreover, it accelerates the model convergence. 
