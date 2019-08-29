# -*- coding: utf-8 -*-

#Librerias
import sys
sys.path.append("..")
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K
from keras.models import Sequential
from keras import layers as L
import tensorflow as tf
from tqdm import tqdm
import tarfile
from IPython import display

#Variables
cascade_face = cv2.CascadeClassifier('frontalface.xml')
raw='raw.tgz'
CODE_SIZE = 256


#funciones
def detect_faces(cascade, test_image):
    image_copy = test_image.copy()
    faces_rect = cascade.detectMultiScale(image_copy, minSize=(50, 50), scaleFactor=1.1, minNeighbors=10)
    if (faces_rect==()): # si no reconoce cara
        return 0 
    else:
        for (x, y, w, h) in faces_rect:
            r = max(w, h) / 1.17
            centerx = x + w / 2
            centery = y + h / 2
            nx = int(centerx - r)
            ny = int(centery - r)
            nr = int(r * 2)
    faceimg = image_copy[ny:ny+nr+10, nx:nx+nr+10]
    return faceimg


def take_images(path):
    res = []
    with tarfile.open(path) as f:
        for m in tqdm(f.getmembers()):
            try:
                img = cv2.imdecode(np.asarray(bytearray(f.extractfile(m).read()), dtype=np.uint8), 1)  
                faces = detect_faces(cascade_face, img)
                img_rgb = cv2.cvtColor(faces, cv2.COLOR_BGR2RGB)
                new_array = cv2.resize(img_rgb, (100, 100))
                res.append(new_array)
                
            except:  
                pass
    print(" {} images".format(len(res)))
    return res

training_dat =take_images(raw)
training_dat = np.float32(training_dat)/255.
IMG_SHAPE =training_dat.shape[1:]

plt.imshow(training_dat[np.random.randint(training_dat.shape[0])], cmap="gray", interpolation="none")

def sample_noise_batch(bsize):
    return np.random.normal(size=(bsize, CODE_SIZE)).astype('float32')


def sample_data_batch(bsize):
    idxs = np.random.choice(np.arange(training_dat.shape[0]), size=bsize)
    return training_dat[idxs]


def sample_images(nrow,ncol, sharp=False):
    images = generator.predict(sample_noise_batch(bsize=nrow*ncol))
    if np.var(images)!=0:
        images = images.clip(np.min(training_dat),np.max(training_dat))
    for i in range(nrow*ncol):
        plt.subplot(nrow,ncol,i+1)
        if sharp:
            plt.imshow(images[i].reshape(IMG_SHAPE),cmap="gray", interpolation="none")
        else:
            plt.imshow(images[i].reshape(IMG_SHAPE),cmap="gray")
    plt.show()


def sample_probas(bsize):
    plt.title('Generated vs real data')
    plt.hist(np.exp(discriminator.predict(sample_data_batch(bsize)))[:,1],
             label='D(x)', alpha=0.5,range=[0,1])
    plt.hist(np.exp(discriminator.predict(generator.predict(sample_noise_batch(bsize))))[:,1],
             label='D(G(z))',alpha=0.5,range=[0,1])
    plt.legend(loc='best')
    plt.show()
############### main ##################


tf.test.gpu_device_name()
gpu_options = tf.GPUOptions(allow_growth=True,per_process_gpu_memory_fraction=1)
s = tf.InteractiveSession(config=tf.ConfigProto(gpu_options=gpu_options))

#red generadora


generator = Sequential()
generator.add(L.InputLayer([CODE_SIZE],name='noise'))
generator.add(L.Dense(128*16*16, activation='elu'))
generator.add(L.Reshape((16,16,128)))
generator.add(L.Deconv2D(128,kernel_size=(2,2),activation='elu'))
generator.add(L.Deconv2D(64,kernel_size=(3,3),activation='elu'))
generator.add(L.UpSampling2D(size=(5,5)))
generator.add(L.Deconv2D(64,kernel_size=4,activation='elu'))
generator.add(L.Deconv2D(64,kernel_size=3,activation='elu'))
generator.add(L.Deconv2D(64,kernel_size=2,activation='elu'))
generator.add(L.Conv2D(3,kernel_size=2,activation=None))
generator.summary()


discriminator = Sequential()

discriminator.add(L.InputLayer(IMG_SHAPE))

discriminator.add(L.Conv2D(8, kernel_size=3))
discriminator.add(L.BatchNormalization())
discriminator.add(L.advanced_activations.LeakyReLU(alpha=.1))

discriminator.add(L.Conv2D(16, kernel_size=3))
discriminator.add(L.BatchNormalization())
discriminator.add(L.advanced_activations.LeakyReLU(alpha=.1))

discriminator.add(L.MaxPooling2D(pool_size=(2, 2)))

discriminator.add(L.Conv2D(32, kernel_size=3))
discriminator.add(L.BatchNormalization())
discriminator.add(L.advanced_activations.LeakyReLU(alpha=.1))

discriminator.add(L.Conv2D(64, kernel_size=3))
discriminator.add(L.BatchNormalization())
discriminator.add(L.advanced_activations.LeakyReLU(alpha=.1))

discriminator.add(L.MaxPooling2D(pool_size=(2, 2)))


# <build discriminator body>

discriminator.add(L.Flatten())
discriminator.add(L.Dense(128,activation='tanh'))
discriminator.add(L.Dense(2,activation=tf.nn.log_softmax))


discriminator.summary()


noise = tf.placeholder('float32',[None,CODE_SIZE])
real_data = tf.placeholder('float32',[None,]+list(IMG_SHAPE))
logp_real = discriminator(real_data)
generated_data = generator(noise) #<gen(noise)>
logp_gen = discriminator(generated_data) #<log P(real | gen(noise))


#discriminator training#
d_loss = -tf.reduce_mean(logp_real[:,1] + logp_gen[:,0])
#regularize
d_loss += tf.reduce_mean(discriminator.layers[-1].kernel**2)
#optimize
disc_optimizer = tf.train.GradientDescentOptimizer(1e-3).minimize(d_loss,var_list=discriminator.trainable_weights)


###generator training###
g_loss = tf.reduce_mean(logp_gen[:,0]) # <generator loss>
gen_optimizer =tf.train.AdamOptimizer(1e-4).minimize(g_loss,var_list=generator.trainable_weights)


s.run(tf.global_variables_initializer())


for epoch in tqdm(range(40000)):
  
    feed_dict = {
        real_data:sample_data_batch(100),
        noise:sample_noise_batch(100)}
  
    for i in range(5):
        s.run(disc_optimizer,feed_dict)
  
    s.run(gen_optimizer,feed_dict)
  
    if epoch %100==0:
        display.clear_output(wait=True)
        sample_images(2,3,True)
        sample_probas(1000)


discriminator.save("discriminator.h5")
generator.save("generator.h5")

