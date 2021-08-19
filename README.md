# Panoptic-Segmentation - DETR

The Objective implements Object detection and Panoptic Segmentation using DETR on custom dataset


## Object detection and Older approach
The task of object detection consists in finding and localizing the objects visible in a given input image. A common way of characterizing these objects is through a unordered set of tight bounding boxes associated with a category label. 

For the past decades the main approach to object detection was to reduce it to the well-studied problem of image classification by classifying all the bounding boxes. 
However, the set of all possible boxes is infinite so this formulation requires selecting an appropriate subset of candidate boxes to operate on and then these boxes are refined using a regression step. The final pipeline is genrally not end to end differntiable.

## DETR
DETR architecture is built upon a transformer, it incorporates almost no geometric priors (like NMS, anchors etc) and is fully differentiable.

1. We first feed the image to a CNN (a pre-trained convolutional backbone) to get image features. Let’s assume we also add a batch dimension. This means that the input to the backbone is a tensor of shape (batch_size, 3, height, width) , assuming the image has 3 color channels (RGB). The CNN backbone outputs a new lower-resolution feature map, typically of shape (batch_size, 2048, height/32, width/32)
2.  This is then projected to match the hidden dimension of the Transformer of DETR, which is 256 by default, using a nn.Conv2D layer. So now, we have a tensor of shape (batch_size, 256, height/32, width/32).
3. Next, the feature map is flattened and transposed to obtain a tensor of shape (batch_size, seq_len, d_model) = (batch_size, width/32*height/32, 256)
4. Next, this resulting vector is sent through the transformer encoder, outputting encoder_hidden_states of the same shape (you can consider these as image features). 

    The encoder's role is to seperate the object instances. The transformer encoder uses self-attention to globally reason about the image, we can visualize the attention patterns to understand what is going on. for a given point in the image, we compute the attention score wrt all the other pixels in the image and average over the attention head.  We observe high attention scores to the pixels belong to the same object. This process is repeated using other source points.

5. For decoding we pass a fixed set of learnt embeddings called object queries through a transformer decoder. The object queries is a tensor of shape (batch_size, num_queries, d_model), with num_queries typically set to 100 and initialized with zeros. These input embeddings are learnt positional encodings that the authors refer to as object queries, and similarly to the encoder, they are added to the input of each attention layer. Each object query will look for a particular object in the image. 
6. The decoder updates these embeddings through multiple self-attention and encoder-decoder attention layers to output decoder_hidden_states of the same shape: (batch_size, num_queries, d_model)
7. Next, the feature vectors obtained are fed to fully connected layes i.e two heads are added on top for object detection: a linear layer for classifying each object query into one of the objects or “no object”, and a MLP to predict bounding boxes for each query.
8. During training, the model uses bipartite matching loss where  the set of predicted classes + bounding boxes are matched to the ground truth annotations using the hungarian algorithm. The Hungarian matching algorithm is used to find an optimal one-to-one mapping of each of the N queries to each of the N annotations. 
9. Next, standard cross-entropy (for the classes) and a linear combination of the L1 and generalized IoU loss (for the bounding boxes) are used to optimize the parameters of the model.

##  Object queries
- They are given as inputs to the decoder layers.
- They are randomly initialized embeddings that are learnt and refined through the course of training and then fixed for evaliation
- DETR uses 100 object embeddings and this defines the upper bound on the number of objects that this model can detect
- No geometric priors are incorporated in the object queries, instead the model learns it directly from the data
