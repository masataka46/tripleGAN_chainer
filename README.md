#Sample code for Triple GAN

##Discription

This software is a python implementation of Triple GAN using chainer.  

See the paper.
https://arxiv.org/abs/1703.02291

##Requirement

I only confirm the operation under this environment
1.  python 3.5.3
2.  chainer 3.3.0
3.  pillow 5.0.0
4.  numpy 1.14.0
5.  cupy 2.3.0

##Usage

First of all, prepare mnist data as npy form.  
`python make_mnist_npy.py`
Then, you can get mnist_train_img.npy, mnist_train_label.npy, mnist_test_img.npy and mnist_test_label.npy.  

To train this model, do `python train_tripleGAN_chainer.py`


## License

MIT 
