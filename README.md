# TF2 DeepDream
This is a Tensorflow 2.0 implementation of DeepDream.

![img/dreamed/dreamed_horse.png]

[This notebook]() demonstrates how you can start dreaming up images of your own with no more than a few lines of code. 

Enjoy!

## Future Work  
- I intended to dream using a network trained on facial images, but ran into issue making forward passes through the net with images. I hope to fix this because I think it would yield interesting dreams.
- I found [this cool project done by sprawld](https://github.com/sprawld/luciddream) that allows the user to stylistically guide the dreaming process by providing an additional image to the model. It seems to yield super cool dreams, so I hope to get that working here as well.


## Works Cited
- A Google AI blogpost,[_Inceptionism: Going Deeper into Neural Networks_](https://ai.googleblog.com/2015/06/inceptionism-going-deeper-into-neural.html), helped me understand the DeepDream algorithm at a high level.
- [The official Keras implementation of DeepDream by Francois Chollet](https://github.com/keras-team/keras/blob/master/examples/deep_dream.py) guided my programmatic understanding of the DeepDream algorithm.
- [Josh Gordon's TF 2.0 implementation of DeepDream](https://colab.research.google.com/github/random-forests/applied-dl/blob/master/examples/9-deep-dream-minimal.ipynb?fbclid=IwAR2IrpxyA6CmqV5p1UVixzuAh84wh_L61uKOxExaxtMWaP52PF6KeRpndBw#scrollTo=q_Nrh8thxye5), though simplistic and lacking in functionality, helped me get familiar with the changes that came with TF 2.0. 
- I referenced [krasserm's face-recognition project](https://github.com/krasserm/face-recognition) in accessing a trained facial recognition model in TF 2.0.
