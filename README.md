# semantic-seg-web
Semantic Segmentation using architecutres like UNET and UNET Attention along with web interface. It trains model based on pre-supplied / user-supplied, training data and provide web UI to tune the hyperparameters.
Semantic Segmentation using architectures like UNET and UNET Attention along with web interface. Models have been trained and are provided for quick prediction. In addition to provided models, it facilitates training of new models based on pre-supplied / user-supplied, training data and provide web UI to tune the hyper-parameters. Web deployment is crucial now a days to complete MLOps lifecycle, hence this repo attempts to provide a web interface to control training and evaluate already trained models with prediction on uploaded unknown image.

## Installation

Download and extract zip in Python 3.9 environment with installed dependencies. Major dependencies are:

```bash
pip install keras shutil tifffile rasterio pillow streamlit
```

## Usage

```bash
streamlit run web.py
```
## About
### Methodology

Models are developed using archictures Unet and Unet-Attention.

#### UNET

UNET is used for performing semantic segmentation of satellite imagery. 
The architecture contains both encoder and decoder paths. 
First path is the contraction path (also called as the encoder) which is used to capture the context in the image. The encoder is  stack of convolutional and max pooling layers. 
The second path is the symmetric expanding path (also called as the decoder) which is used to provide precise localization using transposed convolutions.  It is an end-to-end fully convolutional network (FCN). 
Labelled data is generated using satellite imagery for providing it to model with images for training.

![UNET](//assets/unet.png "Model-UNET")

#### UNET - Attention

While traditional UNET takes skip connections directly as input, UNET-Attention or Attention aware UNET applies attention or ‘weights based on importance’ over skip connections.
The skip connections are later on concatenated with output layers at each depth in decoder path.

![UNET - Attention](//assets/unet-atn.png "Model-UNET-Attention")

### Results & Discussion

Use of DEM channel improved Jaccard Coefficient by ~10%.
Rooftop/Built-Up class is misclassified in ‘road’ and ‘open area’ classes. 
This could be due to improper labelling of pixels in input mask.
Some houses has open roofs exposing underneath floor. Such floors and roads has similar tone and texture which makes it hard to distinguish between them provided RGB and DEM bands. 

#### Results-Unet

Location: Bibipur  
Model: UNET (Depth=5, Filters=32)   
Jaccard Coef (Val) in % = 76.11%

![Unet-Results](//assets/pred_unet.png "UNET-Prediction")

#### Results-Unet-Attention

Location: Bibipur  
Model: UNET-Attention (Depth=4, Filters=32)   
Jaccard Coef (Val) in % = 76.61%

![Unet-Atn-Results](//assets/pred_unet-atn.png "UNET-ATN-Prediction")

### Web User Interface

Customizable & Data independent deployment strategy.
Training
Data Preprocessing
Automatic Data Normalization
Augmentation
Image tiling
Random data splitting in train & validation sets.
Custom data generators
Live feedback from training process via plots
Automatic saving best model
Prediction
Selection of Trained Models with training accuracy.
Facility to upload Dataset for on-the-fly prediction.
Results visualization

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.
