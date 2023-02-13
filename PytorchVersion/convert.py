import os

i = 0
for root, dirs, filenames in os.walk(os.getcwd() + "/pool/"):
	for dir in dirs:
		i += 1
print(i)
"""
DATA_ITEM_LINES = 16 + 1 + 1 + 1
root = os.getcwd()+"/pool_/"
data_files = [x.path for x in os.scandir(root) if x.name.endswith(".gz")]
for file in data_files:
  with gzip.open(file, 'r') as chunk_file:
    print(file)
    print(file.split('/')[-1][:-3].split('.')[-1])
    if os.path.exists(os.getcwd()+"/pool/"+file.split('/')[-1][:-3].split('.')[-1]+"/") :
      continue
    lines = chunk_file.readlines()
    sample_count = len(lines) // DATA_ITEM_LINES
    os.makedirs(os.getcwd()+"/pool/"+file.split('/')[-1][:-3].split('.')[-1]+"/")
    for index in range(sample_count):
      sample_index = index * DATA_ITEM_LINES
      sample = lines[sample_index:sample_index+DATA_ITEM_LINES]
      with gzip.open(os.getcwd()+"/pool/"+file.split('/')[-1][:-3].split('.')[-1]+"/"+str(index)+".gz", 'w') as f:
        f.writelines(sample)
"""
"""
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow
import os
import scipy.io as scio
from tensorflow.python.platform import gfile

save_path = os.path.join(os.getcwd(), "models/aurora-model-0")
#首先，使用tensorflow自带的python打包库读取模型
model_reader = pywrap_tensorflow.NewCheckpointReader(save_path)

#然后，使reader变换成类似于dict形式的数据
var_dict = model_reader.get_variable_to_shape_map()
dicit = {}
#最后，循环打印输出
for key in var_dict:
  dicit[key] = model_reader.get_tensor(key)
  print("variable name: ", key)
  print(model_reader.get_tensor(key))

scio.savemat('weight.mat', dicit)


graph = tf.get_default_graph()
graphdef = graph.as_graph_def()
_ = tf.train.import_meta_graph(os.getcwd() + "/models/aurora-model-0.meta")
summary_write = tf.summary.FileWriter(os.getcwd() + "/models/" , graph)

"""
