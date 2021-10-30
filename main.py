from utils.utils import generate_results_csv
from utils.utils import create_directory
from utils.utils import read_dataset
from utils.utils import transform_mts_to_ucr_format
from utils.utils import visualize_filter
from utils.utils import viz_for_survey_paper
from utils.utils import viz_cam
import os
import numpy as np
import sys
import sklearn
import utils
from utils.constants import CLASSIFIERS
from utils.constants import ARCHIVE_NAMES
from utils.constants import ITERATIONS
from utils.utils import read_all_datasets
import random
from matplotlib import pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sfc.create_sfc import *

func_dict = {"sweep":sweep,"scan":scan,"hilbert":gilbert_curve,"random":random}



def calc_dims(ishape):
    m_row = int(math.sqrt(ishape)+0.5)
    m_col = m_row

    m_row = m_row + (m_row%2)
    m_col = m_col + (m_col%2)
    return m_row,m_col
    
def train_a_ga(X):
    global x_train,y_train,x_test,y_test,y_true, m_row,m_col,nb_classes,input_shape,classifier

    path = create_path_from_shape(m_row,m_col,X)
    if path is None:
        print("ERROR",X)
        return 7000
    
    mx_train = data_generator(path,x_train,m_row,m_col,verbose=False)
    mx_test = data_generator(path,x_test,m_row,m_col,verbose=False)   

    classifier.reset_model()
    pred = classifier.fit(mx_train, y_train, mx_test, y_test, y_true)
    #pred = classifier.predict(mx_test, y_true,mx_train,y_train,y_test,return_df_metrics=True)
    return -pred
  
def train_with_ga():
    global model,x_train,y_train,x_test,y_test,y_true, m_row,m_col,nb_classes,input_shape, classifier
    from geneticalgorithm2 import geneticalgorithm2 as ga2 # for creating and running optimization model
    from geneticalgorithm2 import Crossover, Mutations, Selection # classes for specific mutation and crossover behavior
    from geneticalgorithm2 import Population_initializer # for creating better start population
    from geneticalgorithm2 import np_lru_cache # for cache function (if u want)
    from geneticalgorithm2 import plot_pop_scores # for plotting population scores, if u want
    from geneticalgorithm2 import Callbacks # simple callbacks
    from geneticalgorithm2 import Actions, ActionConditions, MiddleCallbacks # middle callbacks

    alg_params = {'max_num_iteration': 100,
                                           'population_size':20,
                                           'mutation_probability':0.1,
                                           'elit_ratio': 0.01,
                                           'crossover_probability': 0.5,
                                           'parents_portion': 0.3,
                                           'crossover_type':'uniform',
                                           'mutation_type': 'uniform_by_center',
                                           'selection_type': 'roulette',
                                           'max_iteration_without_improv':20}
    
    
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]
    m_row, m_col = calc_dims(x_train.shape[1])
    
    input_shape = (m_row,m_col)
    print(input_shape)
    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)
    print(m_row,m_col)
    classifier = create_classifier(classifier_name, (m_row+1,m_col+1), nb_classes, output_directory,verbose=False)

    classifier.epoch_num = 1
    
    m_row, m_col = calc_dims(x_train.shape[1])
    m_cir_row = m_row // 2
    m_cir_col = m_col // 2
    m_total_weights = ((m_cir_col-1)*(m_cir_row-1)*2)+(m_cir_col-1)+(m_cir_row-1)
    
    varbound = np.array([[0,m_total_weights+2]]*m_total_weights)
    vartype = np.array([['int']]*m_total_weights)
    
    model = ga2(train_a_ga, dimension = len(varbound),   
                variable_boundaries = varbound,
                variable_type='int', 
                variable_type_mixed = None, 
                function_timeout = 100000, 
                algorithm_parameters= alg_params)
    model.run(save_last_generation_as = output_directory+"res1.npy")
    plt.show()
    print(f"Iteration 1 {model.output_dict['function']}")
    path = create_path_from_shape(m_row,m_col,model.output_dict["variable"])
    model.plot_results(save_as=output_directory+"iter1.png")
    model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= output_directory+"plot_scores_end1.png")
    with open(output_directory+"path",'w') as mfile:
        mfiel.write(str(path))
    classifier.verbose = True
    classifier.epoch_num = 5000
    train_a_ga(model.output_dict['last_generation'])
    plt.show()

    print(f"Iteration 2 {model.output_dict['function']}")
    path = create_path_from_shape(m_row,m_col,model.output_dict["variable"])
    model.plot_results(save_as=output_directory+"iter2.png") 
    model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= output_directory+"plot_scores_end2.png")

    #classifier.epoch_num = 1000
    #model.run(start_generation=model.output_dict['last_generation'],save_last_generation_as = "res1.npy")

    #print(f"Iteration 3 {model.output_dict['function']}")
    #path = create_path_from_shape(m_row,m_col,model.output_dict["variable"])
    #model.plot_results(save_as="images/iter3.png") 
    #model.plot_generation_scores(title = 'Population scores after ending of searching', save_as= 'images/plot_scores_end3.png')

    
def fit_classifier(sfc_name,verbose=False):
    global x_train,y_train,x_test,y_test, y_true,nb_classes
    x_train = datasets_dict[dataset_name][0]
    y_train = datasets_dict[dataset_name][1]
    x_test = datasets_dict[dataset_name][2]
    y_test = datasets_dict[dataset_name][3]

    nb_classes = len(np.unique(np.concatenate((y_train, y_test), axis=0)))

    # transform the labels from integers to one hot vectors
    enc = sklearn.preprocessing.OneHotEncoder(categories='auto')
    enc.fit(np.concatenate((y_train, y_test), axis=0).reshape(-1, 1))
    y_train = enc.transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.transform(y_test.reshape(-1, 1)).toarray()

    # save orignal y because later we will use binary
    y_true = np.argmax(y_test, axis=1)
    if sfc_name == 'none':
        if len(x_train.shape) == 2:  # if univariate
            # add a dimension to make it multivariate with one dimension 
            x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
            x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
            
        input_shape = x_train.shape[1:]

        classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory,verbose=verbose)

        print("Shape", x_train.shape,y_train.shape)
        classifier.fit(x_train, y_train, x_test, y_test, y_true)
        print("Regular result",classifier.predict(x_test, y_true,x_train,y_train,y_test,return_df_metrics=True)["accuracy"])

        return
    
    if sfc_name == 'genetic':
        train_with_ga()
        return 

    m_row, m_col = calc_dims(x_train.shape[1])
    x,y = func_dict[sfc_name](m_row,m_col)

    x_train = data_generator((x,y),x_train,m_row,m_col,verbose=verbose)
    x_test = data_generator((x,y),x_test,m_row,m_col,verbose=verbose)

    input_shape = x_train.shape[1:]
    img_width, img_height = input_shape[0], input_shape[1]

    x_train = x_train.reshape(x_train.shape[0], img_width, img_height, 1)
    x_test = x_test.reshape(x_test.shape[0], img_width, img_height, 1)

    print(x_train.shape)

    classifier = create_classifier(classifier_name, input_shape, nb_classes, output_directory,verbose=verbose)
    classifier.epoch_num = 3000
    classifier.fit(x_train, y_train, x_test, y_test, y_true)
    print(sfc_name,"result",classifier.predict(x_test, y_true,x_train,y_train,y_test,return_df_metrics=True)["accuracy"])

    return



def create_classifier(classifier_name, input_shape, nb_classes, output_directory, verbose=False):
    if classifier_name == 'fcn':
        from classifiers import fcn
        return fcn.Classifier_FCN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mlp':
        from classifiers import mlp
        return mlp.Classifier_MLP(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet':
        from classifiers import resnet
        return resnet.Classifier_RESNET(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcnn':
        from classifiers import mcnn
        return mcnn.Classifier_MCNN(output_directory, verbose)
    if classifier_name == 'tlenet':
        from classifiers import tlenet
        return tlenet.Classifier_TLENET(output_directory, verbose)
    if classifier_name == 'twiesn':
        from classifiers import twiesn
        return twiesn.Classifier_TWIESN(output_directory, verbose)
    if classifier_name == 'encoder':
        from classifiers import encoder
        return encoder.Classifier_ENCODER(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'mcdcnn':
        from classifiers import mcdcnn
        return mcdcnn.Classifier_MCDCNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'cnn':  # Time-CNN
        from classifiers import cnn
        return cnn.Classifier_CNN(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'inception':
        from classifiers import inception
        return inception.Classifier_INCEPTION(output_directory, input_shape, nb_classes, verbose)
    if classifier_name == 'resnet2d':
        import classifiers.resnet2d
        return classifiers.resnet2d.Classifier_2DRESNET(output_directory, input_shape, nb_classes, verbose)

############################################### main

# change this directory for your machine
root_dir = '.'

if sys.argv[1] == 'run_all':
    for classifier_name in CLASSIFIERS:
        print('classifier_name', classifier_name)

        for archive_name in ARCHIVE_NAMES:
            print('\tarchive_name', archive_name)
            datasets_dict = read_all_datasets(root_dir, archive_name)

            for iter in range(ITERATIONS):
                print('\t\titer', iter)
                for sfc_name in ['genetic']:
                #for sfc_name in ['random','sweep','scan','hilbert']:

                    tmp_output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name + '/'+sfc_name+'/'
                    for dataset_name in utils.constants.dataset_names_for_archive[archive_name][::]:

                        try:
                            print('\t\t\tdataset_name: ', dataset_name)

                            output_directory = tmp_output_directory + dataset_name + '/'

                            create_directory(output_directory)
                            fit_classifier(sfc_name,verbose=True)

                            print('\t\t\t\tDONE')

                            # the creation of this directory means
                            create_directory(output_directory + '/DONE')
                        except ValueError:
                            with open('not_compatible.txt','a') as mfile:
                                mfile.write(f"{dataset_name} not compatible\n")


elif sys.argv[1] == 'transform_mts_to_ucr_format':
    transform_mts_to_ucr_format()
elif sys.argv[1] == 'visualize_filter':
    visualize_filter(root_dir)
elif sys.argv[1] == 'viz_for_survey_paper':
    viz_for_survey_paper(root_dir)
elif sys.argv[1] == 'viz_cam':
    viz_cam(root_dir)
elif sys.argv[1] == 'generate_results_csv':
    res = generate_results_csv('results.csv', root_dir)
    print(res.to_string())
else:
    # this is the code used to launch an experiment on a dataset
    archive_name = sys.argv[1]
    dataset_name = sys.argv[2]
    classifier_name = sys.argv[3]
    itr = sys.argv[4]
    sfc_name = itr
    
    verbose = False
    if len(sys.argv)>5:
        verbose = sys.argv[5]=='-v'

    if itr == '_itr_0':
        itr = ''

    output_directory = root_dir + '/results/' + classifier_name + '/' + archive_name +'/'+ itr + '/' + \
                       dataset_name + '/'

    test_dir_df_metrics = output_directory + 'df_metrics.csv'

    print('Method: ', archive_name, dataset_name, classifier_name, itr)



    create_directory(output_directory)
    datasets_dict = read_dataset(root_dir, archive_name, dataset_name)

    fit_classifier(sfc_name,verbose=verbose)

    print('DONE')

    # the creation of this directory means
    create_directory(output_directory + '/DONE')
