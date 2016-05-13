import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T

def load_data(batch1_name="../data/cifar-10-batches-py/data_batch_1", 
              batch2_name="../data/cifar-10-batches-py/data_batch_2",
              batch3_name="../data/cifar-10-batches-py/data_batch_3",
              batch4_name="../data/cifar-10-batches-py/data_batch_4",
              batch5_name="../data/cifar-10-batches-py/data_batch_5",
              test_batch_name="../data/cifar-10-batches-py/test_batch"):
    b1 = open(batch1_name, 'rb')
    b2 = open(batch2_name, 'rb')
    b3 = open(batch3_name, 'rb')
    b4 = open(batch4_name, 'rb')
    b5 = open(batch5_name, 'rb')
    tb = open(test_batch_name, 'rb')
    
    batch1 = cPickle.load(b1)
    batch2 = cPickle.load(b2)
    batch3 = cPickle.load(b3)
    batch4 = cPickle.load(b4)
    batch5 = cPickle.load(b5)
    test_batch = cPickle.load(tb)
    
    training_data_array = np.vstack([batch1['data'],batch2['data'],batch3['data'],batch4['data']])
    training_data_array = training_data_array.reshape(40000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
    training_data_array = np.dot(training_data_array[..., :3], [0.299, 0.587, 0.114])
    training_data_array = training_data_array.reshape(40000, 1024)
    training_data_array = (training_data_array/255.0)-0.5
    training_labels_array = np.hstack([batch1['labels'],batch2['labels'],batch3['labels'],batch4['labels']])
    training_data = (training_data_array, training_labels_array)
    
    validation_data_array = np.vstack(batch5['data'])
    validation_data_array = validation_data_array.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
    validation_data_array = np.dot(validation_data_array[..., :3], [0.299, 0.587, 0.114])
    validation_data_array = validation_data_array.reshape(10000, 1024)
    validation_data_array = (validation_data_array/255.0)-0.5
    validation_labels_array = np.hstack(batch5['labels'])
    validation_data = (validation_data_array, validation_labels_array)
    
    test_data_array = np.vstack(test_batch['data'])
    test_data_array = test_data_array.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype('float32')
    test_data_array = np.dot(test_data_array[..., :3], [0.299, 0.587, 0.114])
    test_data_array = test_data_array.reshape(10000, 1024)
    test_data_array = (test_data_array/255.0)-0.5
    test_labels_array = np.hstack(test_batch['labels'])
    test_data = (test_data_array, test_labels_array)
    
    b1.close()
    b2.close()
    b3.close()
    b4.close()
    b5.close()
    tb.close()
    
    def shared(data):
        """Place the data into shared variables.  This allows Theano to copy
        the data to the GPU, if one is available.

        """
        shared_x = theano.shared(
            np.asarray(data[0], dtype=theano.config.floatX), borrow=True)
        shared_y = theano.shared(
            np.asarray(data[1], dtype=theano.config.floatX), borrow=True)
        return shared_x, T.cast(shared_y, "int32")
    return [shared(training_data), shared(validation_data), shared(test_data)]
