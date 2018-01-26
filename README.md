# Sample code for Triple GAN

## Discription
This software is a python implementation of Triple GAN using chainer.  

See the paper.
https://arxiv.org/abs/1703.02291

Notice that this code had not implement gaussian noise in discriminator, Because I don't know how to implement it in chainer.  

## Requirement
I only confirm the operation under this environment
1.  python 3.5.3
2.  chainer 3.3.0
3.  pillow 5.0.0
4.  numpy 1.14.0
5.  cupy 2.3.0

## Usage
First of all, prepare mnist data as npy form.  
`python make_mnist_npy.py`
Then, you can get mnist_train_img.npy, mnist_train_label.npy, mnist_test_img.npy and mnist_test_label.npy.  

To train this model, do `python train_tripleGAN_chainer.py`

## Sample image
Sample image generated in generater after about 500 epochs.
![resultimage_580](https://user-images.githubusercontent.com/15444879/35428209-2246aa7c-02b1-11e8-8053-1fbb888d6d7f.png)

## Refference
I also made triple GAN code in tensorflow.  
https://github.com/masataka46/tripleGAN  
But, there are some bugs.  

## License
MIT 
