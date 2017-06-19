#!/usr/bin/env python
import argparse
import time

import numpy as np

from chainer import cuda
from chainer import optimizers


parser = argparse.ArgumentParser(
    description=' convnet benchmarks on imagenet')
parser.add_argument('--arch', '-a', default='alexnet',
                    help='Convnet architecture \
                    (alex, googlenet, vgga, overfeat)')
parser.add_argument('--batchsize', '-B', type=int, default=128,
                    help='minibatch size')
parser.add_argument('--gpu', '-g', default=0, type=int,
                    help='GPU ID (negative value indicates CPU)')
parser.add_argument('--insize', '-i', default=224, type=int,
                    help='The size of input images')

args = parser.parse_args()
xp = cuda.cupy if args.gpu >= 0 else np

# Prepare model
print(args.arch)
if args.arch == 'alexnet':
    import alex
    model = alex.Alex()
elif args.arch == 'googlenet':
    import googlenet
    model = googlenet.GoogLeNet()
elif args.arch == 'vgga':
    import vgga
    model = vgga.vgga()
elif args.arch == 'overfeat':
    import overfeat
    model = overfeat.overfeat()
elif args.arch == 'vgg16':
    import vgg16
    model = vgg16.VGG16()
elif args.arch == 'vgg19':
    import vgg19
    model = vgg19.VGG19()
elif args.arch == 'unet':
    import unet
    model = unet.UNET()
elif args.arch == 'resnet50':
    import resnet
    model = resnet.ResNet([3, 4, 6, 3])
elif args.arch == 'resnet101':
    import resnet
    model = resnet.ResNet([3, 4, 23, 3])
elif args.arch == 'resnet152':
    import resnet
    model = resnet.ResNet([3, 8, 36, 3])
else:
    raise ValueError('Invalid architecture name')

if 'resnet' in args.arch:
    model.insize = args.insize

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    model.to_gpu()

# Setup optimizer
optimizer = optimizers.SGD(lr=0.01)
optimizer.setup(model)

workspace_size = int(1 * 2**30)
import chainer

chainer.cuda.set_max_workspace_size(workspace_size)

chainer.config.train = True
chainer.config.use_cudnn = 'always'

import cupy

print('Chainer version:', chainer.__version__)
print('CuPy version:', cupy.__version__)
print('CUDA:', chainer.cuda.available)
if chainer.cuda.available:
    cuda_v = cupy.cuda.compiler._get_nvcc_version().split()[-1].decode('utf-8')
    print('CUDA Version:', cuda_v)
print('cuDNN:', chainer.cuda.cudnn_enabled)
if chainer.cuda.cudnn_enabled:
    cudnn_v = cupy.cudnn._cudnn_version
    print('cuDNN Version:', cudnn_v)


class Timer():
    def preprocess(self):
        if xp == np:
            self.start = time.time()
        else:
            self.start = xp.cuda.Event()
            self.end = xp.cuda.Event()
            self.start.record()

    def postprocess(self):
        if xp == np:
            self.end = time.time()
        else:
            self.end.record()
            self.end.synchronize()

    def getElapseTime(self):
        if xp == np:
            return (self.end - self.start) * 1000
        else:
            return xp.cuda.get_elapsed_time(self.start, self.end)


def train_loop():
    # Trainer
    data = np.ndarray((args.batchsize, 3, model.insize,
                       model.insize), dtype=np.float32)
    print('Input data shape:', data.shape)
    data =  np.random.uniform(-1, 1, data.shape).astype(data.dtype)
    total_forward = 0
    total_backward = 0
    niter = 13
    n_dry = 3

    label = np.ndarray((args.batchsize), dtype=np.int32)
    label.fill(1)
    count = 0
    timer = Timer()
    for i in range(niter):
        x = xp.asarray(data)
        y = xp.asarray(label)

        if args.arch == 'googlenet':
            timer.preprocess()
            out1, out2, out3 = model.forward(x)
            timer.postprocess()
            time_ = timer.getElapseTime()
            if i > n_dry - 1:
                count += 1
                total_forward += time_
            out = out1 + out2 + out3
        else:
            timer.preprocess()
            out = model.forward(x)
            timer.postprocess()
            time_ = time_ = timer.getElapseTime()
            if i > n_dry - 1:
                count += 1
                total_forward += time_

        out.zerograd()
        out.grad = np.random.uniform(-1, 1,out.grad.shape).astype(out.grad.dtype)
        model.cleargrads()
        if xp != np:
            xp.cuda.Stream(null=True)
        timer.preprocess()
        out.backward()
        timer.postprocess()
        time_ = timer.getElapseTime()
        if i > n_dry - 1:
            total_backward += time_
        model.cleargrads()

        del out, x, y
        if args.arch == 'googlenet':
            del out1, out2, out3
    print("Average Forward:  ", total_forward / count, " ms")
    print("Average Backward: ", total_backward / count, " ms")
    print("Average Total:    ", (total_forward + total_backward) / count, " ms")
    print("")


train_loop()
