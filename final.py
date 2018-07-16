import sys
sys.path.insert(0, '/home/anshuman/Documents/Repos/insightface/deploy')#to ensure face_model
import face_model
import argparse
import cv2
import numpy as np
import logging
import mxnet as mx
head = '%(asctime)-15s %(message)s'
logging.basicConfig(level=logging.DEBUG, format=head)

def get_iterators(batch_size, data_shape=(3, 112, 112)):
    train = mx.io.ImageRecordIter(
        path_imgrec         = '../datasets/img/saved_faces/orahiimagedb_train.rec',#training sets
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape
        )
    val = mx.io.ImageRecordIter(
        path_imgrec         = '../datasets/img/saved_faces/orahiimagedb_val.rec',#load the training dataset heres
        data_name           = 'data',
        label_name          = 'softmax_label',
        batch_size          = batch_size,
        data_shape          = data_shape,
        )
    return (train, val)
def get_model(ctx, image_size, model_str, layer):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('Loading the Model... Please Wait \n',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym, context=ctx, label_names = None)
  #model.bind(data_shapes=[('data', (args.batch_size, 3, image_size[0], image_size[1]))], label_shapes=[('softmax_label', (args.batch_size,))])
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1]))])
  model.set_params(arg_params, aux_params)
  return model,sym,arg_params,aux_params
def fit(symbol, arg_params, aux_params, train, val, batch_size, num_gpus):
    devs = [mx.gpu(i) for i in range(num_gpus)]
    mod = mx.mod.Module(symbol=symbol, context=devs)
    mod.fit(train, val,
        num_epoch=19,
        arg_params=arg_params,
        aux_params=aux_params,
        allow_missing=True,
        batch_end_callback = mx.callback.Speedometer(batch_size, 10),
        kvstore='device',
        optimizer='sgd',
        optimizer_params={'learning_rate':0.01},
        initializer=mx.init.Xavier(rnd_type='gaussian', factor_type="in", magnitude=2),
        eval_metric='acc')
    metric = mx.metric.Accuracy()
    return mod.score(val, metric)
def get_fine_tune_model(symbol, arg_params, num_classes, layer_name='dropout0'):#need to figure out the last layer name
    """
    symbol: the pretrained network symbol
    arg_params: the argument parameters of the pretrained model
    num_classes: the number of classes for the fine-tune datasets
    layer_name: the layer name before the last fully-connected layer
    """
    all_layers = symbol.get_internals()
    net = all_layers[layer_name+'_output']
    net = mx.symbol.FullyConnected(data=net, num_hidden=num_classes, name='fc1')
    net = mx.symbol.SoftmaxOutput(data=net, name='softmax')
    new_args = dict({k:arg_params[k] for k in arg_params if 'fc1' not in k})
    return (net, new_args)

num_classes = 20
batch_per_gpu = 8
num_gpus = 1
ctx=mx.gpu(0)# to set the context
mod_epoch='../models/model-r34-amf/model/model,0'#cause we are gonna split this by the comma will look into actual epoch and actual values mentioned in default
img_size=(112,112)
layer='fc1'
model,sym, arg_params, aux_params=get_model(ctx,img_size,mod_epoch,layer)
# = mx.model.load_checkpoint(model, 0) improvised check pointing up top
(new_sym, new_args) = get_fine_tune_model(sym, arg_params, num_classes)
batch_size = batch_per_gpu * num_gpus
(train, val) = get_iterators(batch_size)
mod_score = fit(new_sym, new_args, aux_params, train, val, batch_size, num_gpus)
mx.model.save_checkpoint('orahiimages',0,new_sym,new_args,aux_params)
#assert mod_score > 0.77, "Low training accuracy."
