import json
import utils
import models
import metrics
import numpy as np
from validation import KFoldCustom
from utils import PlotGraph
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix


#loading data

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

inputs = np.concatenate((x_train, x_test), axis=0)
targets = np.concatenate((y_train, y_test), axis=0)


hold_path = './states/resnet/state_resnetcifar10holdout102030.json'


#loading dataset and config train set and test set
with open(hold_path, mode='r') as f:
    hold_params = json.loads(f.read())


indexs_hold = hold_params['dataset']


inputs_hold = np.array(list(map(lambda i: inputs[i], indexs_hold)))
targets_hold =  np.array(list(map(lambda i: targets[i], indexs_hold)))


sizes = [0.1, 0.2, 0.3]

for test_size in sizes:
    # test_size = 0.1

    train_x, test_x, train_y, test_y  = train_test_split(inputs_hold,
                                                        targets_hold, 
                                                        test_size=test_size,
                                                        random_state=0, 
                                                        shuffle=False)

    #loading model

    alexnet_hold = models.Resnet34(input_shape=(32,32,3))

    alexnet_hold().compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=[
            "categorical_accuracy",
            "Precision",
            "Recall",
            "AUC",
            metrics.f1_score
        ]
    )

    alexnet_hold().load_weights('./weights/resnet/cifar10_resnet_holdout-{}.h5'.format(test_size))

    alexnet_hold().summary()



    cm = confusion_matrix(test_y, np.argmax(alexnet_hold().predict(test_x), axis=1))
    (fpr, tpr, auc) = metrics.get_roc_curve(utils.to_categorical(test_y), alexnet_hold().predict(test_x))


    cm = cm.astype('float32')
    cm_norm = np.zeros(cm.shape, dtype='float32')
    for i, n in enumerate(cm):
        soma = n.sum()
        for j, m in enumerate(n):
            cm_norm[i][j] = cm[i][j]/soma

    print(cm_norm)

    plot = PlotGraph([0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], path='./graphs/', save=True)

    plot.plot_cm(cm_norm, 'cm_cifar10_resnet_holdout{}.png'.format(int(test_size*100)))
    plot.plot_roc(fpr, tpr, auc, 'roccurve_cifar10_resnet_holdout{}.png'.format(int(test_size*100)))



#Kfold 10
kfold_path = './states/resnet/state_resnetcifar10kfold10.json'
with open(kfold_path, mode='r') as f:
    kfold_params = json.loads(f.read())

indexs_kfold = kfold_params['dataset']

kfold = KFoldCustom(k=10)

inputs_kfold = np.array(list(map(lambda i: inputs[i], indexs_kfold)))
targets_kfold =  np.array(list(map(lambda i: targets[i], indexs_kfold)))

alexnet_kfold = []
models_path = './weights/resnet/cifar10_resnet_kfold-10_{}.h5'

for i in range(10):
    model =  models.Resnet34(input_shape=(32,32,3))
    model().compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=[
            "categorical_accuracy",
            "Precision",
            "Recall",
            "AUC",
            metrics.f1_score]
    )
    model().load_weights(models_path.format(i+1))
    model().summary()
    alexnet_kfold.append(model)
    
models_cm = []
fprs = []
tprs = []
aucs = []

for i, (train, test) in enumerate(kfold.split(inputs_kfold)):
    model = alexnet_kfold[i]()
    models_cm.append(confusion_matrix(targets_kfold[test], 
                                    np.argmax(model.predict(inputs_kfold[test]), axis=1)))
    
    (fpr, tpr, auc) = metrics.get_roc_curve(utils.to_categorical(targets_kfold[test]), model.predict(inputs_kfold[test]))
    
    fprs.append(fpr)
    tprs.append(tpr)
    aucs.append(auc)


cm_mean = np.zeros(models_cm[0].shape)
for cm in models_cm:
    cm_mean += (cm/models_cm[0].shape[0]).astype('int32')


cm_norm = np.zeros(cm.shape, dtype='float32')
for i, n in enumerate(cm):
    soma = n.sum()
    for j, m in enumerate(n):
        cm_norm[i][j] = cm[i][j]/soma


roc_max = (fprs[0], tprs[0], aucs[0])
for i in range(len(aucs)):
    if roc_max[2] > aucs[i]:
        roc_max = (fprs[i], tprs[i], aucs[i])

print(cm_norm)

plot = PlotGraph([0,1,2,3,4,5,6,7,8,9], [0,1,2,3,4,5,6,7,8,9], path='./graphs/', save=True)
plot.plot_cm(cm_norm, 'cm_cifar10_resnet_kfold10.png')
(fpr, tpr, auc) = roc_max
plot.plot_roc(fpr, tpr, auc, 'roccurve_cifar10_resnet_kfold10.png')
