# DD2424_Img2Latex

In this project we built an Encoder-Decoder model to convert images of handwritten mathematical expressions to rendable LaTeX-code. The encoder consists of a 6-layered _convolutional neural network_ (CNN) with batch normalization and max-pooling. The decoder consists of a _long short-term memory_ (LSTM) neural network with a soft attention mechanism. For prediction, beam search was used.

The model was trained on the CROHME dataset.

## Some cherry picked results

![Result A](/images/result_a.PNG)
![Result B](/images/result_b.PNG)
![Result C](/images/result_c.PNG)
![Result D](/images/result_d.PNG)



The expressions above were written by the authors themselves. 
