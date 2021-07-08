# Language identification With a transformer and domain adaptation
In this repository there is a solution for the language identification problem using fine-tuned transformer A-ALBERT. <br>
Pretrained model (with f-score of 0.93) can be found here [model weights](https://drive.google.com/file/d/1qKuVL3MGkfjW6c2QFtABwiC6KW1qgXxL/view?usp=sharing). It is <b>important</b> to run 1 batch through the model before downloading the weights in order to make the model use the needed input shape. <br>
Also there is a [CNN](https://drive.google.com/file/d/1nHsAajtTt1Jfpw0UDxfqpSYJ--ZLf5bk/view?usp=sharing) trained to identify language. Implementation can be found in "Torch pipeline file". It works fine, but fails to generalize because the spectrograms from different datasets differ drastically.
