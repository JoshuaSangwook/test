try:
    import urllib.request
except ImportError:
    raise ImportError('You should use Python 3.x')

import os.path
import gzip
import pickle
import os
import numpy as np

import ssl

ssl._create_default_https_context = ssl._create_unverified_context
## the absolute path of the current file
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/mnist.pkl"

url_base = 'https://raw.githubusercontent.com/fgnt/mnist/master/'
key_file = {
    'train_img':'train-images-idx3-ubyte.gz',
    'train_lable':'train-labels-idx1-ubyte.gz',
    'test_img':'t10k-images-idx3-ubyte.gz',
    'test_lable':'t10k-labels-idx1-ubyte.gz'
}



train_num = 60000
test_num = 10000
img_dim = (1, 28, 28)
img_size = 784

def _download(file_name):
    file_path = dataset_dir + "/" + file_name
    
    if os.path.exists(file_path):
        print("already exists")
        return

    print("Downloading " + file_name + " ... ")
    urllib.request.urlretrieve(url_base + file_name, file_path)
    print("Done")

def download_mnist():
    for v in key_file.values():
       _download(v)

def _load_data(file_name, offset, reshape=None):
    file_path = dataset_dir + "/" + file_name

    with gzip.open(file_path, 'rb') as f:
        data = np.frombuffer(f.read(), np.uint8, offset=offset)

    if reshape is not None:
        data = data.reshape(reshape)

    return data

def _convert_numpy():
    dataset = {}
    dataset['train_lable'] = _load_data(key_file['train_lable'], offset=8)
    dataset['train_img']  = _load_data(key_file['train_img'], offset=16, reshape=(-1, img_size))
    dataset['test_lable'] = _load_data(key_file['test_lable'], offset=8)
    dataset['test_img']  = _load_data(key_file['test_img'], offset=16, reshape=(-1, img_size))
    
    return dataset

## downloiad the mnist dataset
def init_mnist():
    download_mnist()
    dataset = _convert_numpy()
    print("Creating pickle file ...")
    with open(save_file, 'wb') as f:
        pickle.dump(dataset, f, -1)

    print("Done!")

def _change_one_hot_label(X):
    T = np.zeros((X.size, 10))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

def load_mnist(normalize=True,one_hot_label=False): #, flatten=True, ):
    if not os.path.exists(save_file):
        init_mnist()

    with open(save_file, 'rb') as f:
        dataset = pickle.load(f)

    if normalize:
        for key in ('train_img', 'test_img'):
            dataset[key] = dataset[key].astype(np.float32)
            dataset[key] /= 255.0
    
    if one_hot_label:
        dataset['train_label'] = _change_one_hot_label(dataset['train_label'])
        dataset['test_label'] = _change_one_hot_label(dataset['test_label'])    
    

    return dataset['train_img'], dataset['train_lable'] #, (dataset['test_img'], dataset['test_lable']


if __name__ == '__main__':
    if not os.path.exists(save_file):
        init_mnist()

    train_img, train_lable = load_mnist()
    print('train_img:',train_img.shape)
    print('train_img:',train_img[0:])
    tmp = train_img[0:]
    print('tmp:',tmp.shape)
    