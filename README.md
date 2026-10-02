# Small World Propensity Informed Channel and Frequency
Band Selection for Spatial Temporal Graph Autoencoders in
Early Language Assessment
Abstract— The early identification of disorders, particularly those related to language development, is
essential for effective interventions and treatments, particularly in preschool children. EEG
signals are captured from multiple electrodes distributed across the scalp and several EEG-based
deep learning approaches often employ all available EEG electrodes in subject assessment.
However, some channels may contain redundant or noisy information, that can negatively affect
the quality of learned representations. In order to address this limitation, this study employs
small world properties to identify the nodes and frequency bands of high significance for a
specific task from an EEG graph input. The proposed approach is leveraged to train a novel
spatial-temporal graph autoencoder model to predict receptive and expressive language scores of
children from 19-channel EEG signals reliably. Here, encoder and decoder modules are both
comprised of long short-term memory as well as sampling and aggregation convolution blocks
that model the temporal properties at each node along with the relationships between nodes. A
multilayer perceptron is used to regress scores from generated embeddings that have been
infused with age, gender and handedness properties. To ensure robust embeddings that
encapsulate both the temporal properties at each node and the relationships between them, the
proposed model is trained using a piece-wise regression loss as well as a reconstruction loss with
structure and connectivity terms. Experimental results show that the proposed approach is
capable of reliably predicting receptive and expressive language scores, achieving a mean
average error consistently under three.
