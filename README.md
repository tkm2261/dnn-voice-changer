# dnn-voice-changer
voice changer with DNN &amp; GAN on Keras

I have just made a simple voice change with with DNN &amp; GAN. Although the quality of changing voices is bad, It may be easy to understand.

# References

* https://github.com/r9y9/gantts
* https://arxiv.org/abs/1709.08041

# Usage

### Preprocessing
Run prepare_features_vc.py (please see its options.).

As this code is brought from https://github.com/r9y9/gantts, please check it for more detail.

### Training
Run train.py.

Please change the DIR constant to your Preprocessed folder.

### Prediction
Run predict.py

Please change the input_path constant to your Preprocessed folder.
