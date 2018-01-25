import numpy as np
from PIL import Image
import utility as Utility

class Make_mnist_datasets():

    def __init__(self, filename_list, alpha_P):
        #load data from npz file...['test','test_labels','train','train_labels']
        train_img = np.load(filename_list[0])
        train_label = np.load(filename_list[1])
        test_img = np.load(filename_list[2])
        test_label = np.load(filename_list[3])
        
        print("train_img.shape =", train_img.shape)
        print("train_label.shape =", train_label.shape)
        print("test_img.shape =", test_img.shape)
        print("test_label.shape =", test_label.shape)
        #make input data, target data
        x_train = train_img.reshape(60000, 1, 28, 28).astype(np.float32)
        l_train = train_label.reshape(60000, 1).astype(np.int32)
        x_test = test_img.reshape(10000, 1, 28, 28).astype(np.float32)
        l_test = test_label.reshape(10000, 1).astype(np.int32)
        print("x_train.shape = ", x_train.shape)
        print("d_train.shape = ", l_train.shape)
        print("x_test.shape = ", x_test.shape)
        print("d_test.shape = ", l_test.shape)
        print("x_train.shape = ", x_train.shape)
        print("l_train.shape = ", l_train.shape)
        print("x_test.shape = ", x_test.shape)
        print("l_test.shape = ", l_test.shape)
        print("l_train[0] = ", l_train[0])
        print("l_train[1] = ", l_train[1])
        # Utility.make_1_img(x_train)
        
        self.real_num = int(len(x_train) / (1 + alpha_P)) #40,000
        self.else_num = len(x_train) - self.real_num #20,000

        self.img_real = x_train[0:self.real_num]
        self.img_cla = x_train[self.real_num :]
        self.label_real = l_train[0:self.real_num]
        print("self.img_real.shape = ", self.img_real.shape)
        print("self.img_cla.shape = ", self.img_cla.shape)
        print("self.label_real.shape = ", self.label_real.shape)


    def make_data_for_1_epoch(self):

        randInt_real = np.random.permutation(self.real_num)
        randInt_else = np.random.permutation(self.else_num)
        self.img_real_1epoch = self.img_real[randInt_real]
        self.img_cla_1epoch = self.img_cla[randInt_else]
        self.label_real_1epoch = self.label_real[randInt_real]
        return len(self.img_real_1epoch)


    def get_data_for_1_batch(self, i, batchsize, alpha_P):
        img_real_batch = self.img_real_1epoch[i:i + batchsize]
        img_cla_batch = self.img_cla_1epoch[int(i * alpha_P) : int(i * alpha_P) + int(batchsize * alpha_P)]
        label_real_batch = self.label_real_1epoch[i:i + batchsize]
        return img_real_batch, img_cla_batch, label_real_batch


    def convert_to_10class_(self, d): #for tensorflow
        d_mod = np.zeros((len(d), 10), dtype=np.float32)
        for num, contents in enumerate(d):
            d_mod[num][int(contents)] = 1.0
        return d_mod


    def print_img_and_label(self, img_batch, label_batch, int0to9):# for debug

        for num, ele in enumerate(img_batch):
            if num % 10 != int0to9:
                continue

            print("label_batch[", num, "]=", label_batch[num])

            label_num = 0
            for num2, ele2 in enumerate(label_batch[num]):
                if int(ele2) == 1:
                    label_num = num2

            img_tmp = ele
            img_tmp = np.tile(img_tmp, (1, 1, 3)) * 255
            img_tmp = img_tmp.astype(np.uint8)
            image_PIL = Image.fromarray(img_tmp)
            image_PIL.save("./out_images_Debug/debug_img_" + str(num) + "_" + str(label_num) + ".png")
        return

