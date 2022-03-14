from glob import glob

data_dir = '../data/dvs_test/data_processed/obj'
from sklearn.model_selection import train_test_split

filenames = glob(f'{data_dir}/*.jpg')
files = ['data/obj/'+filename.split('/')[-1]+'\n' for filename in filenames]
X_train, X_valid = train_test_split(files, test_size=0.15)
train_file = '../data/dvs_test/data/train.txt'
valid_file = '../data/dvs_test/data/valid.txt'
with open(train_file, 'w') as f:
    f.writelines(X_train)
with open(valid_file, 'w') as f:
    f.writelines(X_valid)