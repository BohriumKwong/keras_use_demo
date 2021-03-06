# 使用Keras进行模型训练和预测的代码示范

##  Installation

```pip install -r requirements.txt```



## Run
训练使用main.py，测试使用predict_demo.py。

```
python main.py
python predict_demo.py
```

具体可参看main.py和predict_demo.py。


## 程序文件说明

### main.py
整个训练过程的完整代码，包括建立模型、数据生成器设置以及模型编译和训练

### predict_demo.py
加载模型进行预测的代码示范，包括针对单张图片进行各种预测(包括只求proba)，以及一个完整目录(二级目录为类型标签)的图片分batch进行预测的方法。此外，还提供使用模型评估、调用sklearn输出classification_report和confusion_matrix并画出美化后的confusion_matrix的方法。

### metrics.py
这个是第三方的acc函数定义脚本，提供许多keras所没有的指标统计，如recall 和F-Score，有需要的请自行在项目中import。比如在model.compile时定义``metrics=[metrics.precision,metrics.recall,metrics.fscore]``，模型开始训练后输出会如下：

Epoch 10/10
33497/33497 [==============================] - 15992s 477ms/step - loss: 0.6079 - precision: 0.7972 - recall: 0.7972 - fmeasure: 0.7972 - val_loss: 0.5196 - val_precision: 0.8090 - val_recall: 0.8090 - val_fmeasure: 0.8090

### generators.py
这个是经我改善后的第三方图像生成器,对比自带的生成器多了颜色转换增强的方法stain_transformation(基于`opencv`),按需使用，具体代码如下：
```python
class DataGenerator(object):
    def __init__(self,
                # ……
                # 前略
                 stain_transformation=False
                 ):

        if data_format is None:
            data_format = K.image_data_format()

    	# ……
    	# 前略
        self.stain_transformation = stain_transformation

	def random_transform(self, x, seed=None):
    	# ……
    	# 前略
        if self.stain_transformation:
            if np.random.random() > 0.5: #也可以根据实际情况，把这个条件改为True
                x = cv2.cvtColor(np.uint8(x),cv2.COLOR_RGB2HSV)
                scale = np.random.uniform(low = 0.98, high = 1.18)
                x[:,:,0] = x[:,:,0] * scale
                x = cv2.cvtColor(np.uint8(x),cv2.COLOR_HSV2RGB)
            else:
                pass
```
同样地，有需要的话请自行在项目中import。


### grad_cam.py
这个是用于生成指定网络的指定layer的卷积和梯度输出特征可视化的脚本,核心方法如下：
```python
def grad_cam(model, x, category_index, layer_name):
    """
    Args:
       model: model
       x: image input
       category_index: category index
       layer_name: last convolution layer name
    """
    # get category loss
    class_output = model.output[:, category_index]

    # layer output
    convolution_output = model.get_layer(layer_name).output
    # get gradients
    grads = K.gradients(class_output, convolution_output)[0]
    # get convolution output and gradients for input
    gradient_function = K.function([model.input], [convolution_output, grads])

    output, grads_val = gradient_function([x])
    output, grads_val = output[0], grads_val[0]

    # avg
    if np.sum(np.unique(grads_val) == np.array([0])) != 1:
        weights = np.mean(grads_val, axis=2)
        cam = np.dot(np.mean(output,axis=2), weights.T)
    else:
        cam = np.mean(output,axis=2)
    # create heat map
    cam = cv2.resize(cam, (x.shape[2], x.shape[1]), cv2.INTER_CUBIC)
    cam = np.maximum(cam, 0)
    heatmap = cam / np.max(cam)

    # Return to BGR [0..255] from the preprocessed image
    image_rgb = x[0, :]
    image_rgb -= np.min(image_rgb)
    image_rgb = np.minimum(image_rgb, 255)

    cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_COOL)
    cam = np.float32(cam) + np.float32(image_rgb)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam), heatmap
```
该方法一次只能接受 **(1,h,w,3)** 这样的numpy array进行预测,仅支持指定非`flatten`后的layer,最终生成的cam图是通过所指定的feature map在axis=2维度上进行求均值而得出的。由于部分图像提取出来的梯度有时候是0,直接融合的话会使得整个cam图清零,因此只有在非0的时候才会将梯度融合到输出结果中。其中配色方案**cv2.COLORMAP_COOL**可酌情选择其他的,具体可以参考这个链接:
https://blog.csdn.net/u013381011/article/details/78341861


### losses.py
根据ICIAR2018_BACH_Challenge乳腺癌分类比赛项目而新写的loss函数，假设你的场景也是这样：分类标签的数字大小能代表错分类的容忍度关系，假设有0,1,2,3四类(简而言之，0是绝对正常，此外越往上越严重)，可以相对容忍类别高的预测成类别低的(如1类预测为0类，3类预测成1类等)，但不能容忍类别低的预测成类别高的，对此情况需要进行额外的惩罚，(如1类预测成2类，0类预测成3类)，其中低类别预测为高类别的，惩罚要比低列表预测为次高类别的更甚(以两者的差的绝对值作为惩罚标准)。
主要是在这个文件增加一个新的loss函数定义：
```python
def weight_categorical_crossentropy(y_true, y_pred):
    return K.weight_categorical_crossentropy(y_true, y_pred)
```
由于只是新增，没有原来的代码进行任何修改，有需要的话可以将这个文件直接替换python安装目录下keras包的根目录的loss文件，即：
**本机python安装目录/site-packages/keras/losses.py**

### tensorflow_backend.py
接续上修改，需要在本脚本中新增``weight_categorical_crossentropy(y_true, y_pred)``方法的具体定义，如下：
```python
def weight_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """weighted Categorical crossentropy between an output tensor and a target tensor.

    # Arguments
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1`
            corresponds to data format `channels_last`,
            and `axis=1` corresponds to data format
            `channels_first`.

    # Returns
        Output tensor.

    # Raises
        ValueError: if `axis` is neither -1 nor one of
            the axes of `output`.
    """
    output_dimensions = list(range(len(output.get_shape())))
    if axis != -1 and axis not in output_dimensions:
        raise ValueError(
            '{}{}{}'.format(
                'Unexpected channels axis {}. '.format(axis),
                'Expected to be -1 or one of the axes of `output`, ',
                'which has {} dimensions.'.format(len(output.get_shape()))))
    # Note: tf.nn.softmax_cross_entropy_with_logits
    # expects logits, Keras expects probabilities.
    if not from_logits:
        # scale preds so that the class probas of each sample sum to 1
        output /= tf.reduce_sum(output, axis, True)
        # manual computation of crossentropy
        _epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
        output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
        return - tf.reduce_sum(target * tf.log(output)*(1 + tf.cast(tf.greater(tf.argmax(output),tf.argmax(target)),tf.float64) \
                                               * tf.abs(tf.argmax(output) - tf.argmax(target))), axis)
    else:
        return tf.nn.softmax_cross_entropy_with_logits(labels=target,
                                                       logits=output)
```

由于只是新增，没有原来的代码进行任何修改，有需要的话可以将这个文件直接替换python安装目录下keras包的根目录的tensorflow_backend.py文件，即：
**本机python安装目录/site-packages/keras/backend/tensorflow_backend.py**
