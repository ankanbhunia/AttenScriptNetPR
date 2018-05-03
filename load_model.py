from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf

def load_graph(model_dir,sess):  
    
    ckpt_file_path  = model_dir + [i for i in os.listdir(model_dir) if i.endswith('meta')][0]
    
    print ('loading tensorflow model ...')
    
    loader = tf.train.import_meta_graph(ckpt_file_path)
    
    loader.restore(sess, tf.train.latest_checkpoint(model_dir))
    
    graph = tf.get_default_graph()
    
    return graph


def load_model(model_path,sess):
    
    model = load_graph(model_path,sess)
    
    print ('creating tensorboard logfiles...')
    
    tf.summary.FileWriter('logfiles',sess.graph)
    
    return [model.get_tensor_by_name('Input_script:0'),
            model.get_tensor_by_name('keep_prob:0'),
            model.get_tensor_by_name('prediction:0')]

