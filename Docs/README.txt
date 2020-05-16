(A few pointers that I thought you could include)

Data Overview

We have been provided with 5 features, they are as follows:

1. sequence: These are usually the input features to the model. Amino acid sequence for this domain. There are 20 very common amino acids (frequency > 1,000,000), and 4 amino acids that are quite uncommon: X, U, B, O, Z.
2. family_accession: These are usually the labels for the model. Accession number in form PFxxxxx.y (Pfam), where xxxxx is the family accession, and y is the version number. Some values of y are greater than ten, and so 'y' has two digits.
3. sequence_name: Sequence name, in the form "uniprot_accession_id/start_index-end_index".
4. aligned_sequence: Contains a single sequence from the multiple sequence alignment with the rest of the members of the family in seed, with gaps retained.
5. family_id: One word name for family.

Bi-directional LSTM:

LSTM (Long Short-Term Memory) Networks are improved versions of RNN, specialized in remembering information for an extended period using a gating mechanism which makes them selective in what previous information to be remembered, what to forget and how much current input is to be added for building the current cell state. Unidirectional LSTM only preserves information of the past because the inputs it has seen are from the past. Using bidirectional will run the inputs in two ways, one from past to future and one from future to past allowing it to preserve contextual information from both past and future at any point of time. More in-depth explanation for RNN and LSTM can be found here(https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/). 

ProtCNN :

This model uses residual blocks inspired from ResNet architecture which also includes dilated convolutions offering larger receptive field without increasing number of model parameters.

ResNet:
Deeper neural networks are difficult to train because of vanishing gradient problem. ResNets introduces skip connection or identity shortcut connection which is adding initial input to the output of the convolution block. This mitigates the problem of vanishing gradient by allowing the alternate shortcut path for gradient to flow through. 

Model:

1. One hot encoded unaligned sequence of amino acids is passed as the input to the network with zero padding.
2. This network uses residual blocks inspired from ResNet architecture which also includes dilated convolutions offering larger receptive field without increasing number of model parameters.

Cosine Similarity:

A number of recent papers have applied deep learning to achieve accurate protein function annotation using classification schemes such as GO terms and EC numbers, with some experimental validation , and also for DNA-protein interactions . The resulting learned data representations, also known as embeddings, for protein sequences also provide new exploratory tools with the potential for significant impact . To interrogate what ProtCNN learns about the natural amino acids, we add a 5-dimensional trainable representation between the one-hot amino acid input and the embedding network (see Methods for details), and retrain our ProtCNN model on the same unaligned sequence data from Pfam-full, achieving the same performance. Cosine similarity comparison.png (left) shows the cosine similarity matrix of the resulting learned embedding, while Cosine similarity comparison.png (right) shows the BLOSUM62 matrix, created using aligned sequence blocks at roughly 62% identity. The structural similarities between these matrices suggest that ProtCNN has learned known amino acid substitution patterns from the unaligned sequence data.

Sequence Code Frequency:

Amino acid sequences are represented with their corresponding 1 letter code, for example, code for alanine is (A), arginine is (R) and so on. The complete list of amino acids with there code can be found here (http://www.cryst.bbk.ac.uk/education/AminoAcid/the_twenty.html).

1. Most frequent amino acid code is L followed by A, V, G.
2. As we can see, that the uncommon amino acids (i.e., X, U, B, O, Z) are present in very less quantity. Therefore we can consider only 20 common natural amino acids for sequence encoding.


Github Link:

https://github.com/ashwin4glory/PFAM-amino-acid-domain-classification
https://github.com/asafpr/domain_classifier (Not fully convinced with this though)

Research Paper :

https://www.biorxiv.org/content/10.1101/626507v4.full