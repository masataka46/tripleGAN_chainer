import numpy as np
import os
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
# from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
# from chainer.training import extensions
# from PIL import Image
import utility as Utility
from make_mnist_datasets import Make_mnist_datasets

#global variants
batchsize = 100
data_size = 6000
noise_num = 100
class_num = 10
n_epoch = 1000
l2_norm_lambda = 0.001
alpha_P = 0.5
alpha_pseudo = 0.1
alpha_apply_thr = 200

keep_prob_rate = 0.5

mnist_file_name = ["mnist_train_img.npy", "mnist_train_label.npy", "mnist_test_img.npy", "mnist_test_label.npy"]
seed = 1234
np.random.seed(seed=seed)

out_image_dir = './out_images_tripleGAN' #output image file
out_model_dir = './out_models_tripleGAN' #output model file
try:
    os.mkdir(out_image_dir)
    os.mkdir(out_model_dir)
    os.mkdir('./out_images_Debug') #for debug
except:
    print("mkdir error")
    pass

make_mnist = Make_mnist_datasets(mnist_file_name, alpha_P)


def gaussian_noise(h, noise_rate):
    batch, dim = h.data.shape[0],h.data.shape[1]
    ones = Variable(np.ones((batch, dim), dtype=np.float32))
    return F.gaussian(h, noise_rate*ones)

#generator------------------------------------------------------------------
class Generator(chainer.Chain):
    def __init__(self):
        super(Generator, self).__init__(
            fc1 = L.Linear(class_num + noise_num, 500),
            bn2=L.BatchNormalization(500),
            fc3=L.Linear(500, 500),
            bn4=L.BatchNormalization(500),
            fc5=L.Linear(500, 784),
        )

    def __call__(self, y, z, train=True):
        h = F.concat((y, z), axis=1)
        h = self.fc1(h)
        h = F.softplus(h, beta=1.0)
        with chainer.using_config('train', train):
            h = self.bn2(h)
        h = self.fc3(h)
        h = F.softplus(h, beta=1.0)
        with chainer.using_config('train', train):
            h = self.bn4(h)
        h = self.fc5(h)
        h = F.sigmoid(h)
        x = F.reshape(h, (-1, 1, 28, 28))
        return x, y


#discriminator-----------------------------------------------------------------
class Discriminator(chainer.Chain):
    def __init__(self):
        super(Discriminator, self).__init__(
            fc1=L.Linear(class_num + 28 * 28, 1000),
            fc2=L.Linear(1000, 500),
            fc3=L.Linear(500, 250),
            fc4=L.Linear(250, 250),
            fc5=L.Linear(250, 250),
            fc6=L.Linear(250, 1),
        )

    def __call__(self, x, y, train=True):
        x_re = F.reshape(x, (len(x), -1))
        h = F.concat((x_re, y), axis=1)
        h = self.fc1(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.fc2(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.fc3(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.fc4(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.fc5(h)
        h = F.leaky_relu(h, slope=0.2)
        h = self.fc6(h)
        out = F.sigmoid(h)
        return out


#classifier-----------------------------------------------------------------
class Classifier(chainer.Chain):
    def __init__(self):
        super(Classifier, self).__init__(
            c1=L.Convolution2D(1, 32, ksize=5, stride=1, pad=2),  # 28x28 to 28x28
            c2=L.Convolution2D(32, 64, ksize=3, stride=1, pad=1),  # 14x14 to 14x14
            c3=L.Convolution2D(64, 64, ksize=3, stride=1, pad=1),  # 14x14 to 14x14
            c4=L.Convolution2D(64, 128, ksize=3, stride=1, pad=1),  # 7x7 to 7x7
            c5=L.Convolution2D(128, 128  , ksize=3, stride=1, pad=1),  # 7x7 to 7x7
            fc6=L.Linear(128, 10),
        )

    def __call__(self, x, train=True):
        h = self.c1(x)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        with chainer.using_config('train', train):
            h = F.dropout(h, ratio=0.5)
        h = self.c2(h)
        h = F.relu(h)
        h = self.c3(h)
        h = F.relu(h)
        h = F.max_pooling_2d(h, 2)
        with chainer.using_config('train', train):
            h = F.dropout(h, ratio=0.5)
        h = self.c4(h)
        h = F.relu(h)
        h = self.c5(h)
        h = F.relu(h)
        h = F.average_pooling_2d(h, 7)
        h = F.reshape(h, (len(h), -1))
        out = self.fc6(h)
        return x, out


gen = Generator()
dis = Discriminator()
cla = Classifier()

gen.to_gpu()
dis.to_gpu()
cla.to_gpu()

optimizer_gen = optimizers.Adam(alpha=0.0003, beta1=0.5)
optimizer_dis = optimizers.Adam(alpha=0.0003, beta1=0.5)
optimizer_cla = optimizers.Adam(alpha=0.0003, beta1=0.5)

optimizer_gen.setup(gen)
optimizer_dis.setup(dis)
optimizer_cla.setup(cla)

optimizer_dis.add_hook(chainer.optimizer.WeightDecay(0.0005))


#training loop
for epoch in range(0, n_epoch):
    sum_loss_gen = np.float32(0)

    sum_loss_dis = np.float32(0)
    sum_loss_dis_r = np.float32(0)
    sum_loss_dis_c0 = np.float32(0)
    sum_loss_dis_g0 = np.float32(0)

    sum_loss_cla = np.float32(0)
    sum_accu_cla = np.float32(0)
    sum_loss_dis_c1 = np.float32(0)
    sum_loss_RL = np.float32(0)
    sum_loss_RP = np.float32(0)

    len_img_real = make_mnist.make_data_for_1_epoch()

    for i in range(0, len_img_real, batchsize):
        img_real_batch, img_cla_batch, label_real_batch = make_mnist.get_data_for_1_batch(i, batchsize, alpha_P)
        label_real_batch_10c = make_mnist.convert_to_10class_(label_real_batch)
        len_real_batch = len(img_real_batch)
        len_cla_batch = len(img_cla_batch)
        len_gen_batch = int(len(img_real_batch) * alpha_P)
        img_real_batch = Variable(cuda.to_gpu(img_real_batch))
        img_cla_batch = Variable(cuda.to_gpu(img_cla_batch))
        label_real_batch_10c = Variable(cuda.to_gpu(label_real_batch_10c))

        z = np.random.uniform(0, 1, len_gen_batch * noise_num)
        z = z.reshape(-1, noise_num).astype(np.float32)
        z = Variable(cuda.to_gpu(z))

        label_gen_int = np.random.randint(0, class_num, len_gen_batch)
        label_gen = label_gen_int.reshape(-1, 1).astype(np.int32)
        label_gen_int = Variable(cuda.to_gpu(label_gen_int))
        label_gen_float = make_mnist.convert_to_10class_(label_gen)
        label_gen_float = Variable(cuda.to_gpu(label_gen_float))

        d_dis_g_1 = np.ones((len_gen_batch, 1), dtype=np.float32)
        d_dis_g_0 = np.zeros((len_gen_batch, 1), dtype=np.float32)
        d_dis_r_1 = np.ones((len_real_batch, 1), dtype=np.float32)
        d_dis_r_1_I = d_dis_r_1.reshape(len_real_batch).astype(np.int32)
        d_dis_c_1 = np.ones((len_cla_batch, 1), dtype=np.float32)
        d_dis_c_0 = np.zeros((len_cla_batch, 1), dtype=np.float32)
        d_dis_g_1 = Variable(cuda.to_gpu(d_dis_g_1))
        d_dis_g_0 = Variable(cuda.to_gpu(d_dis_g_0))
        d_dis_r_1 = Variable(cuda.to_gpu(d_dis_r_1))
        d_dis_r_1_I = Variable(cuda.to_gpu(d_dis_r_1_I))
        d_dis_c_1 = Variable(cuda.to_gpu(d_dis_c_1))
        d_dis_c_0 = Variable(cuda.to_gpu(d_dis_c_0))

        # stream around generator
        x_gen, y_gen = gen(label_gen_float, z)

        # stream around classifier
        x_cla_0, y_cla_0 = cla(x_gen, train=True)  # from generator
        x_cla_1, y_cla_1 = cla(img_real_batch, train=True)  # real image labeled
        x_cla_2, y_cla_2 = cla(img_cla_batch, train=True)  # real image unlabeled
        loss_RP = F.softmax_cross_entropy(y_cla_0, label_gen_int) #loss in case generated image
        loss_RL = F.softmax_cross_entropy(y_cla_1, d_dis_r_1_I) #loss in case real image

        # stream around discriminator
        out_dis_g = dis(x_gen, y_gen)  # from generator
        out_dis_r = dis(img_real_batch, label_real_batch_10c)  # real image and label
        out_dis_c = dis(x_cla_2, y_cla_2)  # from classifier
        loss_dis_g_D = F.mean_squared_error(out_dis_g, d_dis_g_0) #loss related to generator for D grad
        loss_dis_r_D = F.mean_squared_error(out_dis_r, d_dis_r_1) #loss related to real imaeg for D grad
        loss_dis_c_D = F.mean_squared_error(out_dis_c, d_dis_c_0) #loss related to classifier for D grad
        loss_dis_g_G = F.mean_squared_error(out_dis_g, d_dis_g_1) #loss related to generator for G grad
        loss_dis_c_C = F.mean_squared_error(out_dis_c, d_dis_c_1) #loss related to classifier for C grad

        # total loss of discriminator
        loss_dis_total = loss_dis_r_D + alpha_P * loss_dis_c_D + (1 - alpha_P) * loss_dis_g_D #+ l2_norm_lambda * norm_L2

        # total loss of classifier
        if epoch > alpha_apply_thr:
            loss_cla_total = alpha_P * loss_dis_c_C + loss_RL + alpha_pseudo * loss_RP
        else:
            loss_cla_total = alpha_P * loss_dis_c_C + loss_RL

        # total loss of generator
        loss_gen_total = (1 - alpha_P) * loss_dis_g_G

        #for printout
        sum_loss_gen += loss_gen_total.data

        sum_loss_dis += loss_dis_total.data
        sum_loss_dis_r += loss_dis_r_D.data
        sum_loss_dis_c0 += loss_dis_c_D.data
        sum_loss_dis_g0 += loss_dis_g_D.data

        sum_loss_cla += loss_cla_total.data
        sum_loss_dis_c1 += loss_dis_c_C.data
        sum_loss_RL += loss_RL.data
        sum_loss_RP += loss_RP.data

        #back prop
        dis.cleargrads() #discriminator
        loss_dis_total.backward()
        optimizer_dis.update()

        cla.cleargrads() #classifier
        loss_cla_total.backward()
        optimizer_cla.update()

        gen.cleargrads() #generator
        loss_gen_total.backward()
        optimizer_gen.update()

    print("-----------------------------------------------------")
    print("epoch =", epoch , ", Total Loss of G =", sum_loss_gen, ", Total Loss of D =", sum_loss_dis,
          ", Total Loss of C =", sum_loss_cla)
    print("Discriminator: Loss Real =", sum_loss_dis_r, ", Loss C =", sum_loss_dis_c0, ", Loss D =", sum_loss_dis_g0,)
    print("Classifier: Loss adv =", sum_loss_dis_c1, ", Loss RL =", sum_loss_RL, ", Loss RP =", sum_loss_RP,)

    if epoch % 10 == 0:
        sample_num_h = 10
        sample_num = sample_num_h ** 2

        # z_test = np.random.uniform(0, 1, sample_num_h * noise_num).reshape(sample_num_h, 1, noise_num)
        # z_test = np.tile(z_test, (1, sample_num_h, 1))
        z_test = np.random.uniform(0, 1, sample_num_h * noise_num).reshape(1, sample_num_h, noise_num)
        z_test = np.tile(z_test, (sample_num_h, 1, 1))
        z_test = z_test.reshape(-1, sample_num).astype(np.float32)
        label_gen_int = np.arange(10).reshape(10, 1).astype(np.float32)
        label_gen_int = np.tile(label_gen_int, (1, 10)).reshape(sample_num)
        label_gen_test = make_mnist.convert_to_10class_(label_gen_int)
        label_gen_test = Variable(cuda.to_gpu(label_gen_test))
        z_test = Variable(cuda.to_gpu(z_test))
        x_gen_test, y_gen_test = gen(label_gen_test, z_test, train=False)
        x_gen_test_data = x_gen_test.data
        x_gen_test_reshape = x_gen_test_data.reshape(len(x_gen_test_data), 28, 28, 1)
        x_gen_test_reshape = cuda.to_cpu(x_gen_test_reshape)
        Utility.make_output_img(x_gen_test_reshape, sample_num_h, out_image_dir, epoch)

    if epoch % 100 == 0:
        #serializer
        serializers.save_npz(out_model_dir + '/gen_' + str(epoch) + '.model', gen)
        serializers.save_npz(out_model_dir + '/cla_' + str(epoch) + '.model', cla)
        serializers.save_npz(out_model_dir + '/dis_' + str(epoch) + '.model', dis)

