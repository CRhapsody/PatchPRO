import os


if __name__ == '__main__':
    os.system(f'python /root/PatchART/tinyimagenet/exp_tinyimagenet_feature.py')
    os.system(f'python /root/PatchART/tinyimagenet_stresult.py')