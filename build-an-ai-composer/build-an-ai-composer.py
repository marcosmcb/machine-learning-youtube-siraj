# Code for video https://www.youtube.com/watch?v=S_f2qV2_U00

'''

Machine Learning: Data + Model = Pattern Recognition

ML in 4 steps

1- Collect Dataset
2- Build the model
3- Train the model
4- Test the model

'''

import urllib
import zipfile
import nottingham_util
import rnn

url = "www-etud.iro.umontreal.ca/~boulanni/NOttingham.zip"

zip = zipfile.ZipFile(r'dataset.zip')
zip.extractall('data')

nottingham_util.create_model()

rnn.train_model()