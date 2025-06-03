import numpy as np
from tensorflow.keras.backend import gradients
import keras
from keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.image as mpimg
from IPython.display import Image, display
#import IPython
from tensorflow import data as tf_data
import tensorflow as tf
import os
from PIL import Image
from keras import backend as K
from tensorflow.python.framework.ops import disable_eager_execution
import tensorflow.compat.v1 as tf1
from tensorflow.python.framework import ops
#from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input, decode_predictions
#from tensorflow.keras.applications.mobilenet_v3 import preprocess_input, decode_predictions
from tensorflow.keras.applications.efficientnet_v2 import preprocess_input, decode_predictions
#import cv2
from keras.layers import Lambda
from tensorflow.python.framework.ops import disable_eager_execution
import random
import datetime
from matplotlib import colormaps
from tensorflow.keras.optimizers.schedules import ExponentialDecay
import shap
from sklearn.metrics import ConfusionMatrixDisplay
from tensorflow.python.ops.numpy_ops import np_config
from sklearn.manifold import TSNE
#from openTSNE import TSNE
from sklearn.preprocessing import LabelBinarizer
#from scipy import interp
from sklearn.preprocessing import label_binarize
from sklearn.decomposition import PCA
from random import uniform
from tensorflow import keras
from sklearn.metrics import RocCurveDisplay
from sklearn.metrics import roc_curve
import lime
from lime import lime_image
from lime.wrappers.scikit_image import SegmentationAlgorithm
from skimage.segmentation import mark_boundaries
from collections import Counter
from tensorflow.keras import Model

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def prediction_ex(image_batch,predictions,save_path):

    print("Beginning prediction examples eval...")
    plt.figure(figsize=(15, 10))
    r=9
    if BATCH_SIZE < 10:
        r=BATCH_SIZE
    for i in range(r):
        ax = plt.subplot(3, 3, i + 1)
        #print(image_batch[i])
        plt.imshow(image_batch[i].numpy().astype("uint8"))
        plt.title("Pred: "+str(predictions[i])+" True: " + str(label_batch[i]),fontsize='10')
        #plt.title(label_batch[i],y=1.0,color='b')
        plt.axis("off")
    plt.tight_layout()
    plt.savefig(os.path.join(save_path,"fig")+"/Prediction_"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg")

def SHAP_explainer(IMG_HW,images,class_names,test_dataset,BATCH_SIZE,save_path,model):
    tf.keras.backend.clear_session()
    
    print("Beginning SHAP eval explainer....")

    def f(x):
        tmp = x.copy()
        preprocess_input(tmp)
        return model(tmp)
    image_batch, label_batch = test_dataset.as_numpy_iterator().next()
    #print(image_batch.shape)
    blur="blur("+str(IMG_HW)+","+str(IMG_HW)+")"
    masker_blur = shap.maskers.Image(str(blur), image_batch[0].shape)
    #masker_blur = shap.maskers.Image("blur(512,512)", image_batch.shape)
    explainer = shap.Explainer(f, masker_blur, output_names=class_names)
    
    #print(image_batch.shape)
    shap_values_ = explainer(image_batch[0:4].astype("uint8"), max_evals=5000, batch_size=BATCH_SIZE,outputs=shap.Explanation.argsort.flip[:4])   
    shap.image_plot(shap_values_)
    plt.show()
    plt.savefig(os.path.join(save_path,"fig")+"/SHAP_"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg")
    
def LIME(image_batch,predictions_lime,label_lime,score,model,save_path,nb_classe):
    tf.keras.backend.clear_session()
    print("Evaluating LIME....")



    segmenter = SegmentationAlgorithm('slic', n_segments=50, compactness=5, sigma=1)
    img_lime=image_batch
    #print(img_lime)
    explainer = lime_image.LimeImageExplainer()

    facteur=4
    fig ,axs= plt.subplots(nb_classe+1, 4,figsize=[5*facteur, (nb_classe+1)*facteur])
    axs[0,0].text(0.45,0.1, 'First',fontsize=3*facteur)
    axs[0,0].axis('off')
    axs[0,1].text(0.4,0.1, 'Second',fontsize=3*facteur)
    axs[0,1].axis('off')
    axs[0,2].text(0.35,0.1, 'Penultimate',fontsize=3*facteur)
    axs[0,2].axis('off')
    axs[0,3].text(0.45,0.1, 'Least',fontsize=3*facteur)
    axs[0,3].axis('off')

    
    npa=np.asarray(score,dtype=np.float32)
    score=(npa*10000).astype(int)
    score_list=score.tolist()
    
    
    for i in range(len(label_lime)):

        true_label=label_lime[i-1]
        score=score_list.copy()
        
        print("Evaluating for "+str(true_label))

        #print(score[i-1])
        
        label_test=label_lime.copy()
        
        score_min=min(score[i-1])
        label_min=label_test[score[i-1].index(score_min)]
        label_test.pop(score[i-1].index(score_min))
        score[i-1].pop(score[i-1].index(score_min))
        
        score_max=max(score[i-1])
        label_max=label_test[score[i-1].index(score_max)]
        label_test.pop(score[i-1].index(score_max))
        score[i-1].pop(score[i-1].index(score_max))

        score_second=max(score[i-1])
        label_second=label_test[score[i-1].index(score_second)]
        label_test.pop(score[i-1].index(score_second))
        score[i-1].pop(score[i-1].index(score_second))

        score_penul=min(score[i-1])
        label_penul=label_test[score[i-1].index(score_penul)]
        #label_test.pop(score[i-1].index(score_penul))
        #score[i-1].pop(score[i-1].index(score_penul))          ####!!!!! be careful with equal or less than 4 classes   

        for j in range(4):
            #print('j:'+str(j))
            if j==2:
                l=nb_classe-1
                #print('l2:'+str(l))
                print('Evaluating least probable....')
                axs[i+1,j].set_title("Label: "+str(true_label)+' Pred: '+str(label_min)+'\nScore: '+str((score_min)/100),fontsize = 3*facteur)
            if j==3:
                l=nb_classe-2
                #print('l3:'+str(l))
                print('Evaluating penultimate probable....')
                axs[i+1,j].set_title("Label: "+str(true_label)+' Pred: '+str(label_penul)+'\nScore: '+str((score_penul)/100),fontsize = 3*facteur)
            if j==1:
                l=1
                #print('l1:'+str(l))
                print('Evaluating second most probable....')
                axs[i+1,j].set_title("Label: "+str(true_label)+' Pred: '+str(label_second)+'\nScore: '+str((score_second)/100),fontsize = 3*facteur)
            if j==0:
                l=0
                #print('l0:'+str(l))
                print('Evaluating most probable....')
                
                axs[i+1,j].set_title("Label: "+str(true_label)+' Pred: '+str(label_max)+'\nScore: '+str((score_max)/100),fontsize = 3*facteur)
                
            explanation = explainer.explain_instance(img_lime[i-1],
                                            classifier_fn = model.predict,
                                            top_labels=nb_classe, hide_color=0, num_samples=1000, segmentation_fn=segmenter)
            #print(l)
            #print(nb_classe)
            image, mask = explanation.get_image_and_mask(explanation.top_labels[l],
                                    positive_only=False,
                                    num_features=10,
                                    hide_rest=False,
                                    min_weight=0
                                    )
            #plt.subplot(5, 7, i+1)
            axs[i+1,j].imshow(mark_boundaries(image, mask))
            axs[i+1,j].axis('off')
            #axs[i,j].set_title("Label: "+str(label_lime[i])+' Pred: '+str(predictions_lime[i]),fontsize = 10)
            #axs[i,j].set_title("Label: "+str(label_lime[i])+' Pred: '+str(l),fontsize = 10)
    fig.tight_layout()
    #fig.suptitle("LIME prediction", fontsize=7*facteur)
    fig.savefig(os.path.join(save_path,"fig")+"/LIME_"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg")
    plt.close()

def T_sne(NB_TEST,BATCH_SIZE,model,tsne_dataset,perplex,class_names,save_path,nb_threshold):

    print("Beginning T-SNE eval ....")

    tf.keras.backend.clear_session()
    if NB_TEST > nb_threshold:
        batch_num=nb_threshold
    else:
        batch_num=NB_TEST
    print("Testing with "+str(batch_num)+" batch of "+str(BATCH_SIZE)+" samples ( for a total of "+str(batch_num * BATCH_SIZE)+" samples)...")
    model2 = tf.keras.Model(inputs=model.input, outputs=model.layers[-2].output)
    test_ds = np.concatenate(list(tsne_dataset.take(batch_num).map(lambda x, y : x))) # get five batches of images and convert to numpy array
    features = model2(test_ds)
    labels = np.argmax(model(test_ds), axis=-1)

    
    

    def scale_to_01_range(x):

        value_range = (np.max(x) - np.min(x))
        starts_from_zero = x - np.min(x)
        return starts_from_zero / value_range
    
    for p in perplex:

        if (batch_num * BATCH_SIZE) < p:
            p = (batch_num * BATCH_SIZE)-1
    
        print("Evaluating for perplexity: "+str(p))
        #with tf.device('/CPU:0'):
        #print("Evaluating PCA....")
        #pca = PCA(n_components=100)
        #features = pca.fit_transform(features)
        #print(features)

        tsne = TSNE(n_components=2,perplexity=p,verbose = 1,n_jobs=-1,n_iter=10000,method='exact').fit_transform(features)

        tx = tsne[:, 0]
        ty = tsne[:, 1]

        tx = scale_to_01_range(tx)
        ty = scale_to_01_range(ty)


        classes = class_names
        colors=[]
        cmap=colormaps['hsv']
        colors = cmap(np.linspace(0, 1, len(classes)))

        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(111)
        m=0
        marker=[".","v","^","<",">","8","s","p","P","*","h","X","D"]
        for idx, c in enumerate(colors):
            indices = [i for i, l in enumerate(labels) if idx == l]
            current_tx = np.take(tx, indices)
            current_ty = np.take(ty, indices)
            if m > 12:
                m=0
            ax.scatter(current_tx, current_ty, c=c, s=50, marker=marker[m],label=classes[idx])
            m=m+1

        ax.legend(loc='best')
        fig.suptitle("t-SNE with perplexity of "+str(p))
        fig.tight_layout() 
        fig.savefig(os.path.join(save_path,"fig")+"/t-SNE_"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+str(random.random())+".jpg")
        plt.close()

def basic_result(train_dataset,validation_dataset,test_dataset,save_path):

    tf.keras.backend.clear_session()
    print("Beginning basic test ....")
    loss, accuracy, auc, top_3_categorical_accuracy = model.evaluate(train_dataset)
    print('Train accuracy :', accuracy)
    print('Train loss:', loss)
    print('Train auc:', auc)
    print('Train top 3 accuracy:', top_3_categorical_accuracy)
    txt_train=' Train accuracy :'+str(accuracy)+'\nTrain loss:'+str(loss)+' \nTrain auc:'+str(auc)+'\nTrain top 3 accuracy:'+str(top_3_categorical_accuracy)

    loss, accuracy, auc, top_3_categorical_accuracy = model.evaluate(validation_dataset)
    print('Validation accuracy :', accuracy)
    print('Validation loss:', loss)
    print('Validation auc:', auc)
    print('Validation top 3 accuracy:', top_3_categorical_accuracy)
    txt_valid='\nValidation accuracy :'+str(accuracy)+'\nValidation loss:'+str(loss)+' \nValidation auc:'+str(auc)+'\nValidation top 3 accuracy:'+str(top_3_categorical_accuracy)

    loss, accuracy, auc, top_3_categorical_accuracy = model.evaluate(test_dataset)
    print('Test accuracy :', accuracy)
    print('Test loss :', loss)
    print('Test auc :', auc)
    print('Test top 3 accuracy:', top_3_categorical_accuracy)
    txt_test='\nTest accuracy :'+str(accuracy)+'\nTest loss:'+str(loss)+' \nTest auc:'+str(auc)+'\nTest top 3 accuracy:'+str(top_3_categorical_accuracy)

    print('Saving results in basic_results.txt')

    if not os.path.exists(os.path.join(save_path,'fig','basic_results.txt')):
        with open(os.path.join(save_path,'fig','basic_results.txt'), "x") as file:
                    file.write(txt_train+txt_valid+txt_test)

def ROC_AUC(roc_auc_pred,label_batch,class_names,save_path):
    
    tf.keras.backend.clear_session()
    print("Beginning ROC-AUC eval ....")

    roc_auc_pred=np.array(roc_auc_pred).reshape((len(label_batch),len(class_names)))
    roc_auc_pred_np=tf.math.argmax(roc_auc_pred, 1)
    unique_label=Counter(label_batch)
    print(unique_label)
    len_labels=len(unique_label)
    print("Total of unique label: "+str(len_labels))
    label_batch_np=np.array(label_batch)
    label_binarizer = LabelBinarizer().fit(label_batch_np)
    #print(label_binarizer)

    j=0

    #print(label_batch_np)

    for i in label_batch_np:
        #print(i)
        if j==0:
            true_label=np.array(label_binarizer.transform([str(i)]))
            #print(true_label)
            j=1
        else:
            true_label = tf.concat([true_label, label_binarizer.transform([str(i)])],0)
            #print(true_label)


    true_label=np.array(true_label)
    roc_auc_pred=np.array(roc_auc_pred)


    cmap=colormaps['Set1']
    colors = cmap(np.linspace(0, 1, len_labels))

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111)
    for class_id, color in enumerate(colors):
        display = RocCurveDisplay.from_predictions(
            true_label[:, class_id],
            roc_auc_pred[:, class_id],
            name=class_names[class_id],
            color=color,
            ax=ax,
            plot_chance_level=(class_id == 2)
            #despine=True,
        )
        _ = display.line_.set(
        linestyle='-'
        )
    _ = display.ax_.set(
        xlabel="False Positive Rate",
        ylabel="True Positive Rate",
        title="One-vs-Rest ROC curves",
    )
    
    plt.savefig(os.path.join(save_path,"fig")+"/ROC-AUC_"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg")
    plt.close()

def Conf_matrix(label_batch, predictions,save_path):
    tf.keras.backend.clear_session()
    figsize = (9,8)
    ConfusionMatrixDisplay.from_predictions(label_batch, predictions, normalize='true', cmap='Reds', ax=plt.subplots(figsize=figsize)[1])
    plt.xticks(rotation=45, ha='right')
    plt.savefig(os.path.join(save_path,"fig")+"/Conf_matrix_"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg")
    plt.close()


def Grad_CAM(img_size,last_conv_layer_name,img_path,model,save_path,inner_model):
    class Gradcam:
        """
            Class to rapresent heatmap of output of neural network based on Grad-CAM algorithm.
        
            ...

            Attributes
            __________
            model: HD5
            model to analyze
            layer_name: str
            name of the layer of the model to analyze
            img_path: str
            Path of the image
            size: (int, int)
            size of the image that the model takes as input 
            pred_index (opt): int
            index of target label of image
            inner_model (opt): str
            name of the inner model, used for transfer learning

            Methods
            _______
            get_img_array():
            Load an image and convert it into numpy array.
            set_index_from_path():
            Predict the label of the image passed.
            make_gradcam_heatmap(img_array):
            Calculate the heatmap.
            overlay_heatmap(heatmap, campath, alpha, cmap):
            Superimposed original image and heatmap and return another image.
            generate_stack_img(figsize, save_name, superimposed_name, alpha, cmap):
            Mix together original image, heatmap and superimposed image and save them
            into a single compose image.
        
        """

        def __init__(self, model, layer_name, img_path, size, pred_index=None, inner_model=None):
            """Gradcam constructor to inizialize the object
            
            Parameters
            __________
            model: HD5
                model to analyze
            layer_name: str
                name of the layer of the model to analyze
            img_path: str
                Path of the image
            size: (int, int)
                size of the image that the model takes as input 
            pred_index (opt): int
                index of target label of image
            inner_model (opt): str
                name of the inner model, used for transfer learning
            """

            self.model = model
            self.layer_name = layer_name
            self.img_path = img_path
            self.size = size
            self.pred_index = pred_index
            self.inner_model = inner_model
            
            if self.inner_model == None:
                self.inner_model = model

            if self.pred_index == None:
                self.pred_index = self.set_index_from_path()

        def get_img_array(self):
            """Load an image and convert it into numpy array.

            Returns
            _______
            (ndarray) = Image converted into array
            """

            img = tf.keras.utils.load_img(self.img_path, target_size=self.size)
            array = tf.keras.utils.img_to_array(img)
            array = np.expand_dims(array, axis=0)

            return array

        def set_index_from_path(self):
            """Predict the label of the image passed.

            Returns
            _______
            (int) = Predicted index
            """

            #array = self.get_img_array()
            array = self.load_image(image_path= self.img_path)
            self.model.layers[-1].activation = None
            preds = self.model.predict(array)
            i = np.argmax(preds, axis=1)

            self.pred_index = i[0]

            return self.pred_index
        '''
        def build_guided_model(self):
            if "GuidedBackProp" not in ops._gradient_registry._registry: #avoid over-write
                @ops.RegisterGradient("GuidedBackProp")
                def _GuidedBackProp(op, grad):
                    dtype = op.inputs[0].dtype
                    return grad * tf1.cast(grad > 0., dtype) * \
                        tf1.cast(op.inputs[0] > 0., dtype)
        
            g = tf1.get_default_graph()       #guidedbackdrop in another copy
            with g.gradient_override_map({'Relu': 'GuidedBackProp'}):
                new_model = self.model
            return new_model
        '''
        def make_gradcam_heatmap(self, img_array):
            """Calculate the heatmap.

            Paramenters
            ___________
            img_array: ndarray
                Image converted into array

            Returns
            _______
            (ndarray) = Heatmap of image converted into array
            """
            grad_inner_model=self.inner_model
            grad_model = tf.keras.Model(
                inputs=grad_inner_model.inputs,
                outputs=[grad_inner_model.get_layer(self.layer_name).output, grad_inner_model.output]
            )

            with tf.GradientTape() as tape:
                inputs = tf.cast(img_array, tf.float32)
                last_conv_layer_output, preds = grad_model(inputs)
                #class_channel = preds[:, self.pred_index]
                class_channel = preds[:, len(preds)]

            grads = tape.gradient(class_channel, last_conv_layer_output)
            pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

            last_conv_layer_output = last_conv_layer_output[0]
            heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
            heatmap = tf.squeeze(heatmap)

            heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

            return heatmap.numpy()

        def overlay_heatmap(self, heatmap, cam_path, alpha, cmap):
            """Superimposed original image and heatmap and return another image.

            Parameters
            __________
            heatmap: ndarray
                Heatmap of the image
            cam_path: str
                Path where to save only superimposed image generated
            alpha: float
                Parameter used as weight for color map when overlapping is applied
            cmap: str
                Type of color map used for heatmap

            Returns
            _______
            (Image) = Image of superimposed image of original and heatmap
            """

            img = tf.keras.utils.load_img(self.img_path, target_size=self.size)
            img = tf.keras.utils.img_to_array(img)

            heatmap = np.uint8(255 * heatmap)

            color_map = colormaps.get_cmap(cmap)
            color_map = color_map(np.arange(256))[:, :3]
            color_map = color_map[heatmap]
            color_map = tf.keras.utils.array_to_img(color_map)
            color_map = color_map.resize((img.shape[1], img.shape[0]))
            color_map = tf.keras.utils.img_to_array(color_map)

            # Superimpose the heatmap on original image
            superimposed_img = color_map * alpha + img
            superimposed_img = tf.keras.utils.array_to_img(superimposed_img)

            if cam_path != None:
                
                superimposed_img.save(cam_path)

            return superimposed_img
        
        def load_image(self,image_path, preprocess=True):
            """Load and preprocess image."""
            x = tf.keras.preprocessing.image.load_img(image_path, target_size=self.size)
            if preprocess:
                x = tf.keras.preprocessing.image.img_to_array(x)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
            return x

        def generate_stack_img(self, figsize=(17,9), save_name=None, superimposed_name=None, alpha=0.4, cmap="jet"):
            """Mix together original image, heatmap and superimposed image and save them
            into a single compose image.

            Parameters
            __________
            figsize (opt): (int, int)
                Figsize of final image
                default=(17,9)
            save_name (opt): str
                Name of image to save.
                default={original_image_name}_gradcam_{current_time} 
            superimposed_name (opt): str
                Name of superimposed image to save.
                default=None, superimposed image will not be saved
            alpha (opt): float
                Parameter used as weight for color map when overlapping is applied
                default=0.4
            cmap (opt): str
                Type of color map used for heatmap
                default="jet"

            Returns
            _______
            None
            """

            #img_array = self.get_img_array()
            img_array = self.load_image(image_path= self.img_path)
            heatmap = self.make_gradcam_heatmap(img_array)
            superimposed_gradcam = self.overlay_heatmap(heatmap, cam_path=superimposed_name, alpha=alpha, cmap=cmap)
            

            if save_name == None:
                current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
                extract_name = (self.img_path.split("/")[-1]).rsplit(".",1)[0]
                save_name = "./Grad_CAM/" + extract_name + "_gradcam_" + current_time 

            img_original = Image.open(self.img_path)
            img_original = img_original.resize(self.size)

            #fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
            #suptitle="Grad-CAM | Prediction: " + str(class_names[self.pred_index])+" | Path: "+str(self.img_path)
            
            #slash=[i for i, n in enumerate(self.img_path) if n == '/']
            dir_path=os.path.dirname(self.img_path)
            c_name=os.path.basename(os.path.basename(dir_path))
            label="Pred: "+str(class_names[self.pred_index])+"|Label: "+str(c_name)
            print(label)

            return img_original,heatmap,superimposed_gradcam,label,save_name
        
    def load_image(size,image_path, preprocess=True):
            """Load and preprocess image."""
            x = tf.keras.preprocessing.image.load_img(image_path, target_size=size)
            if preprocess:
                x = tf.keras.preprocessing.image.img_to_array(x)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)
            return x
    
    def deprocess_image(x):
            x = x.copy()
            x -= x.mean()
            x /= (x.std() + K.epsilon())
            x *= 0.25

            # clip to [0, 1]
            x += 0.5
            x = np.clip(x, 0, 1)

            # convert to RGB array
            x *= 255
            if K.image_data_format() == 'channels_first':
                x = x.transpose((1, 2, 0))
            x = np.clip(x, 0, 255).astype('uint8')
            return x
    
    def guided_backprop(inner_model,img_path,layer_name,size):
        @tf.custom_gradient
        def guidedRelu(x):
            def grad(dy):
                    return tf.cast(dy>0,"float32") * tf.cast(x>0, "float32") * dy
                #return tf.nn.relu(x), grad
            return tf.nn.silu(x), grad
        guided_inner_model=inner_model
        guided_grad_model = tf.keras.models.Model(
                inputs=guided_inner_model.inputs,
                outputs=[guided_inner_model.get_layer(layer_name).output,
                        guided_inner_model.output]
            )
        
        guided_layer_dict = [guided_layer for guided_layer in guided_grad_model.layers[1:] if hasattr(guided_layer,'activation')]
        for guided_layer in guided_layer_dict:
            #if guided_layer.activation == tf.keras.activations.relu:
            if guided_layer.activation == tf.keras.activations.swish:
                guided_layer.activation = guidedRelu
        guided_image_prepro=load_image(size,img_path)
        with tf.GradientTape() as guided_tape:
                guided_inputs = tf.cast(guided_image_prepro, tf.float32)
                guided_tape.watch(guided_inputs)
                guided_outputs = guided_grad_model(guided_inputs)

        guided_grads = guided_tape.gradient(guided_outputs,guided_inputs)[0]
        guided_grads = np.flip(deprocess_image(np.array(guided_grads)), -1)
        return guided_grads
    
    print("Beginning Grad-CAM eval ....")
    print('Nb img : '+str(len(img_path)))
    tf.keras.backend.clear_session()
    #model.load_weights(checkpoint_path)
    img_size=(img_size,img_size)

    #last_conv_layer_name = 'conv_7b'
    facteur=3
    fig, axs= plt.subplots(len(img_path), 4,figsize=[5*facteur, len(img_path)*facteur])
    i=0
    for img in img_path:
        print('Img Grad-CAM: '+str(i))
        gc = Gradcam(model, 
                layer_name=last_conv_layer_name,
                img_path=img,
                size=img_size,
                inner_model=model.get_layer(inner_model))
        img_original,heatmap,superimposed_gradcam,label,save_name =gc.generate_stack_img(save_name=os.path.join(save_path,"fig")+"/Grad_Cam_"+str(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))+".jpg")
        
        

        axs[i,1].imshow(heatmap)
        axs[i,1].axis('off')
        axs[i,1].set_title("Heatmap",fontsize=5*facteur)

        axs[i,2].imshow(superimposed_gradcam)
        axs[i,2].axis('off')
        axs[i,2].set_title("Grad-CAM",fontsize=5*facteur)

        axs[i,3].set_title(str(label),fontsize=5*facteur)
        axs[i,3].imshow(img_original)
        axs[i,3].axis('off')

        i=i+1
    i=0
    for img in img_path:
        print('Img Guided-Grad-CAM: '+str(i))
        guided_grad_cam = guided_backprop(model.get_layer(inner_model), img,last_conv_layer_name,img_size)
        
        axs[i,0].imshow(guided_grad_cam)
        axs[i,0].set_title("Guided Grad-CAM",fontsize=5*facteur)
        axs[i,0].axis('off')

        i=i+1
    #fig.suptitle("Grad-CAM and Guided Grad-CAM evaluation",fontsize=30)
    fig.tight_layout()   
    fig.savefig(save_name)
    plt.close()
        

############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################
##############################################################      Main programm       ####################################################################
############################################################################################################################################################
############################################################################################################################################################
############################################################################################################################################################


COLOR_MODE_IMG = 'grayscale'
IMG_HW=350
BATCH_SIZE = 8
IMG_SIZE = (IMG_HW, IMG_HW)
fine_tune_epochs =10
initial_epochs = 10
test_true=0
fine_tune_at = 0
validation_split=0.15

PATH="./dataset_arnaud/Dataset_croped/Dataset_dorso_ventral"
save_path="./save/350_Dataset_dorso_ventral_efficientnetB4"
checkpoint_path = os.path.join(save_path,"cp_350.weights.h5")


aPath = '--xla_gpu_cuda_data_dir=/usr/lib/cuda'
os.environ['XLA_FLAGS'] = aPath
#os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'

'''
path_grad_cam=os.path.join(PATH,'Grad_Cam')
img_path=[]

for (dirpath, dirnames, filenames) in os.walk(path_grad_cam):
                for img in filenames:
                    img_path.append(os.path.join(path_grad_cam,img))
'''

try:
    os.mkdir(os.path.join(save_path,"fig"))
except OSError as error:
    print("Path already existing: ", error)

cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1)


train_dir =os.path.join(PATH,'train')


train_dataset, validation_dataset = keras.utils.image_dataset_from_directory(train_dir,
                                                            shuffle=True,
                                                            batch_size=BATCH_SIZE,
                                                            image_size=IMG_SIZE,
                                                            label_mode="categorical",
                                                            validation_split=validation_split,
                                                            subset="both",
                                                            seed=5465189,
                                                            color_mode='rgb')
                                                            #pad_to_aspect_ratio=True)
test_file_paths = validation_dataset.file_paths
if test_true == 1:
    test_dir = os.path.join(PATH, 'test')
    #test_dir='./dataset_arnaud/Dataset_croped/Dataset_lateral/train/'
    test_dataset = tf.keras.utils.image_dataset_from_directory(test_dir,
                                                                 shuffle=True,
                                                                 batch_size=BATCH_SIZE,
                                                                 image_size=IMG_SIZE,
                                                                 label_mode="categorical",
                                                                 #pad_to_aspect_ratio=True,
                                                                 color_mode='rgb',
                                                                 seed=5465189)
    test_file_paths = test_dataset.file_paths

class_names = train_dataset.class_names
r=9
if BATCH_SIZE < 10:
    r=BATCH_SIZE
print(r)
plt.figure(figsize=(10, 10))
for images, labels in train_dataset.take(1):
    for i in range(r):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(images[i].numpy().astype("uint8"))
        #plt.title(class_names[labels[i]])
        plt.axis("off")
val_batches = tf.data.experimental.cardinality(validation_dataset)

if test_true == 0:
    test_dataset = validation_dataset.take(val_batches // 5)
    validation_dataset = validation_dataset.skip(val_batches // 5)
    #test_file_paths = file_paths[:val_batches // 5]

path_grad_cam=[]
for name in class_names:
    for path in test_file_paths:
        dir_path=os.path.dirname(path)
        c_name=os.path.basename(os.path.basename(dir_path))
        #print(c_name)
        if c_name == name:
            path_grad_cam.append(path)
            break
img_path=path_grad_cam
test_shape = test_dataset
print('Number of validation batches: %d' % tf.data.experimental.cardinality(validation_dataset))
NB_TEST=tf.data.experimental.cardinality(test_dataset)
print('Number of test batches: %d' % NB_TEST)
#NB_TEST= int(NB_TEST) - 1 
NB_TEST=int(NB_TEST)
AUTOTUNE = tf.data.AUTOTUNE

train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

layer=[tf.keras.layers.RandomFlip('horizontal'),
  tf.keras.layers.RandomRotation(0.2),
  tf.keras.layers.RandomContrast(0.2),
  tf.keras.layers.RandomBrightness(0.2)]
if COLOR_MODE_IMG == 'grayscale':
   layer.append(tf.keras.layers.Lambda(lambda x: tf.repeat(tf.reduce_sum(x*tf.constant([0.21, 0.72, 0.07]), axis=-1, keepdims=True),3,axis=-1)))
data_augmentation = tf.keras.Sequential(
  layer
)

for image, _ in train_dataset.take(1):
    plt.figure(figsize=(10, 10))
    first_image = image[0]
    for i in range(r):
        ax = plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255)
        plt.axis('off')
#preprocess_input = tf.keras.applications.efficientnet.preprocess_input
# Create the base model from the pre-trained model
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.EfficientNetB4(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
image_batch, label_batch = next(iter(train_dataset))
feature_batch = base_model(image_batch)
print(feature_batch.shape)
base_model.trainable = False
base_model.summary()
global_average_layer = tf.keras.layers.GlobalMaxPool2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)
prediction_layer = tf.keras.layers.Dense(len(class_names),activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)
inputs = tf.keras.Input(shape=(IMG_HW, IMG_HW, 3))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)
base_learning_rate = 0.0005

METRICS=['categorical_accuracy', 
        tf.keras.metrics.AUC(name='auc',multi_label=False),
        tf.keras.metrics.TopKCategoricalAccuracy(name="top_3_categorical_accuracy",k=3)]
model.compile(optimizer=tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate),
              loss=tf.keras.losses.CategoricalCrossentropy(),
              metrics=METRICS)
#model.summary()
len(model.trainable_variables)

base_model.trainable = True
print("Number of layers in the base model: ", len(base_model.layers))

# Fine-tune from this layer onwards


# Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False
model.compile(loss=tf.keras.losses.CategoricalCrossentropy(),
              optimizer = tf.keras.optimizers.RMSprop(learning_rate=base_learning_rate/10),
              metrics=METRICS)


#model.summary()
len(model.trainable_variables)
fine_tune_epochs =1
total_epochs =  initial_epochs + fine_tune_epochs

model.load_weights(checkpoint_path)


label_matrix = 0
predictions = 0
i=0
img_list=[]



for image_batch, label_batch in test_dataset.take(NB_TEST):
    if i==0:
       label_matrix=label_batch
       predictions=model.predict_on_batch(image_batch).flatten() 
       i=1
    else:
        label_matrix = tf.concat([label_matrix, label_batch],0)
        img_pred=model.predict_on_batch(image_batch)
        predictions = tf.concat([predictions, img_pred.flatten()],0)
    if BATCH_SIZE == len(image_batch):
        img_list.append(image_batch)
label_batch = label_matrix  
roc_auc_pred= predictions
print(len(img_list)*BATCH_SIZE)
img_list=tf.reshape(img_list,[len(img_list)*BATCH_SIZE,IMG_HW,IMG_HW,3])


label_names=[]
for i in label_batch:
   e=0
   for j in i:
      if j == 1:
         label_names.append(class_names[e])
      e=e+1
   e=0
label_batch=label_names
# Apply a sigmoid since our model returns logits

#predictions = tf.nn.sigmoid(predictions)
prediction_matrix=predictions
#predictions = tf.where(predictions < 0.5, 0, 1)
div=len(class_names)
div_pred=int(len(predictions)/div)
e=0
pred_true=0.0
label_predict=[]
for i in range(div_pred):
    for j in range(div):
        it=(len(class_names)*i)+j
        pred=predictions[it].numpy()
        if float(pred_true) < pred.item(0):
            pred_lab=class_names[j]
            pred_true=pred.item(0)
    #print(pred_true)
    pred_true=0.0
    label_predict.append(pred_lab)
    e=e+1
predictions=label_predict
print('Predictions:\n', predictions[0:9])
print('Labels:\n', label_batch[0:9])


# preping data for lime

prediction_matrix=tf.reshape(prediction_matrix, [int(len(prediction_matrix)/len(class_names)),int(len(class_names))])
img_lime=[]
label_lime=[]
predictions_lime=[]
score=[]
for i in class_names:
    for j in range(len(label_batch)):
        if i == label_batch[j-1]:
            img_lime.append(img_list[j-1].numpy().astype("uint8"))
            label_lime.append(label_batch[j-1])
            predictions_lime.append(predictions[j-1])
            score.append(prediction_matrix[j-1])
            #print(prediction_matrix[j])
            break

combined_dataset = validation_dataset.concatenate(test_dataset)
NB_TEST=tf.data.experimental.cardinality(combined_dataset)

############################################################################################################################################
#######################################################        Figure       ######################################################################
############################################################################################################################################
############################################################################################################################################

###############prediction example for 9 img

#prediction_ex(img_list,predictions,save_path)

##############basic result saved in txt format

#basic_result(train_dataset,validation_dataset,test_dataset,save_path)

#############Grad-CAM 

#Grad_CAM(IMG_HW,'top_conv',img_path,model,save_path,"efficientnetv2-m")
#img_path=test_file_paths[51:100]
Grad_CAM(IMG_HW,'top_conv',img_path,model,save_path,"efficientnetb4")
#Grad_CAM(IMG_HW,'conv_7b',img_path,model,save_path,"inception_resnet_v2")

#############conf_matrix

#Conf_matrix(label_batch, predictions,save_path)

############ ROC -AUC implementation

#ROC_AUC(roc_auc_pred,label_batch,class_names,save_path)

###############LIME implementation

#LIME(img_lime,predictions_lime,class_names,score,model,save_path,len(class_names))

############t-SNE implementation

#T_sne(int(NB_TEST),BATCH_SIZE,model,test_dataset,[5,10,15,20,25,30,35,40,45,50,75],class_names,save_path,15)
#T_sne(int(NB_TEST),BATCH_SIZE,model,combined_dataset,[5,10,15,20,25,30,35,40,45,50,75],class_names,save_path,10)  # work with 180 img max for now

################SHAP implementation
################Deep Explainer

#SHAP_explainer(IMG_HW,img_lime,class_names,test_dataset,BATCH_SIZE,save_path,model)


