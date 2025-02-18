import os


if __name__ == '__main__':
    DIR_PATH = '/root/PatchART/mnist'
    os.system(f'python /root/PatchART/mnist/exp_mnist_small.py')
    os.system(f'python /root/PatchART/mnist_stresult.py')