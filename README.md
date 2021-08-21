# Panoptic-Segmentation - DETR

The Objective implements Object detection and Panoptic Segmentation using DETR on custom dataset


## Object detection - Background
The task of object detection consists in finding and localizing the objects visible in a given input image. A common way of characterizing these objects is through a unordered set of tight bounding boxes associated with a category label. 

For the past decades the main approach to object detection was to reduce it to the well-studied problem of image classification by classifying all the bounding boxes. Modern detectors address this in an indirect way using anchors, non  maximum seperation procedures etc. The set of all possible boxes is infinite so this formulation requires selecting an appropriate subset of candidate boxes to operate on and then these boxes are refined using a regression step. Such methods are significantly influenced by post processing steps and the final pipeline is generally not end to end differntiable and 



## DETR
DETR architecture is built upon a transformer, propose a direct set prediction approach to by pass the surrogate tasks.  It simplifies the detection pipeline by dropping multiple hand-designed componenets that encode prior knowledge like NMS, anchors etc) and is fully differentiable.


### Architecture 

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


## Panoptic Segmentation. 
Its is fusion of instance segmentation which aims at predicting a mask for each distinct instant of a foreground object and semantic segmentation which aims at predicting a class label for each pixel in the background, the resulting task requires that each pixel belongs to exactly one segment. 

DETR can be naturally extend by adding a mask head on top of the decoder outputs for panoptic segmentation. This head can be used to produce panoptic segmentation by treating stuff and thing classes in a unified way. Through panoptic segmentation the authors aim at understanding whether DETR's object embeddings can be used for the downstream tasks.

- First task is to train regular DETR to predict boxes around things (foreground) and stuff (background objects) in a uniform manner
- Once the detection model is trained we freeze the weights, train a mask head for 25 epochs


## Panoptic architecture overview:

![image](https://user-images.githubusercontent.com/42609155/130302393-4363c045-02d8-407e-bbd7-64799966b1d1.png)

- We first feed the image to the CNN and set aside the activations from intermediate layers - Res5, Res4, Res3, Res2.
- These are then passed to transformer encoder, after the encoder we also set aside the encoder version of the image and then proceed to the decoder. 
- we endup with object embedding for the foreground objects and for each segment of the background objects 
- Next, a multi-head attention layer is used that returns the attention scores over the encoded image for each object embedding. 
- we proceed to upsample and clean these masks, using convolutional network that uses the imtermediate activations from the backbone.
- As a result we get high resolution maps where each pixel contains a binary logit of belonging to the mask.
- Finally, the masks are merged by assigning each pixel to the mask with the highest logit using a simple pixel wise argmax


## Part 1 


### We take the encoded image (dxH/32xW/32) and send it to Multi-Head Attention (FROM WHERE DO WE TAKE THIS ENCODED IMAGE?)

- The a new lower-resolution feature map output by CNN backbone, is then passed to the tranformer encoder, the encoder outputs encoder_hidden_states of the shape dxH/32xW/32, you can consider these as image features.

### We do something here to generate NxMxH/32xW/32 maps. (WHAT DO WE DO HERE?)

- Once we endup with object embedding (box embeddings) for the foreground objects and for each segment of the background objects. 
- These box embeddings of shape d x N  and the image features i.e  the encoder outputs encoder_hidden_states of the shape dxH/32xW/32 is fed through a multi-head attention layer 
- Next, a multi-head attention layer for each object computes mutli-head(with M heads) attention scores of this embedding over the output of the encoder, generating M attention heatmaps per object of shape NxMxH/32xW/32 in a small resolution. 

### Then we concatenate these maps with Res5 Block (WHERE IS THIS COMING FROM?)

- The initial image when fed through the CNN backbone architecture (like RESNET50 or RESET101), we set aside the activations from intermediate layers - Res5, Res4, Res3, Res2.
- These imtermediate activations from the backbone are then used to upsample and clean the activation maps.
- As a result we get high resolution maps where each pixel contains a binary logit of belonging to the mask of shape N x H/4 x W/4


## Approach to Solve Panoptic Segmentation on Construction dataset

Below is a high level approach outlines to solve Panoptic Segmentation on Custom Construction dataset, more details would be added as we progress:

### Collect and Prepare data
- Once the dataset is collected, we would examine the dataset and perform some exploratory analysis 
- Next, we have to collect images and annotation for "stuff" from coco dataset.
- Convert the custom dataset to COCO format and combine it with COCO "stuff" dataset 

### Train and Validation Split
- Split the dataset into 80:20 ratio for training and validation

### Object detection
- First we  train DETR to predict boxes around both things and stuff classes for around 275 epoch 
- Once the detection model is trained we freeze the weights

### Panoptic Segmentation 

#### Groundtruth Images for Panoptic Segmentation
- We pass the dataset through some panoptic detector to generate ground truth panoptic images (this needs to be explored further), such that custom masks overlay on binary masks generated by stuff classes.

- We then train a mask head using DETR for 25 epochs for panoptic segmentation
