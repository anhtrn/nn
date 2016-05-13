import cPickle
import gzip
import numpy as np
import theano
import theano.tensor as T

batch1_name="../data/cifar-10-batches-py/data_batch_1" 
batch2_name="../data/cifar-10-batches-py/data_batch_2"
batch3_name="../data/cifar-10-batches-py/data_batch_3"
batch4_name="../data/cifar-10-batches-py/data_batch_4"
batch5_name="../data/cifar-10-batches-py/data_batch_5"
test_batch_name="../data/cifar-10-batches-py/test_batch"
     
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

#----------------------------- TRAINING DATA ---------------------------------#

# Combine all data batches together
training_data_array = np.vstack([batch1['data'],batch2['data'],batch3['data'],batch4['data']])
training_labels_array = np.hstack([batch1['labels'],batch2['labels'],batch3['labels'],batch4['labels']])

# Extract images of each label from the training set
training_data = []
for i in range(0, 10):
    training_data.append(np.take(training_data_array, (training_labels_array == i).nonzero(), axis=0)[0])

# Create corresponding labels
# training_labels = []
# for j in range(0, 10):
#     temp = np.empty(training_data[j].shape[0])
#     temp.fill(1)
#     training_labels.append(temp)

# Create airplane/automobile data set
airplane_automobile_data = np.vstack((training_data[0], training_data[1]))
airplane_automobile_labels = np.hstack((np.zeros(training_data[0].shape[0]), np.ones(training_data[1].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_automobile_data.shape[0])
airplane_automobile_data = airplane_automobile_data[p]
airplane_automobile_labels = airplane_automobile_labels[p]
# save to file
dict = {'data': airplane_automobile_data, 'labels': airplane_automobile_labels}
output = open('airplane_automobile', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create airplane/ship data set
airplane_ship_data = np.vstack((training_data[0], training_data[8]))
airplane_ship_labels = np.hstack((np.zeros(training_data[0].shape[0]), np.ones(training_data[8].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_ship_data.shape[0])
airplane_ship_data = airplane_ship_data[p]
airplane_ship_labels = airplane_ship_labels[p]
# save to file
dict = {'data': airplane_ship_data, 'labels': airplane_ship_labels}
output = open('airplane_ship', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create airplane/truck data set
airplane_truck_data = np.vstack((training_data[0], training_data[9]))
airplane_truck_labels = np.hstack((np.zeros(training_data[0].shape[0]), np.ones(training_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_truck_data.shape[0])
airplane_truck_data = airplane_truck_data[p]
airplane_truck_labels = airplane_truck_labels[p]
# save to file
dict = {'data': airplane_truck_data, 'labels': airplane_truck_labels}
output = open('airplane_truck', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create automobile/ship data set
automobile_ship_data = np.vstack((training_data[1], training_data[8]))
automobile_ship_labels = np.hstack((np.zeros(training_data[1].shape[0]), np.ones(training_data[8].shape[0])))
# shuffle all images
p = np.random.permutation(automobile_ship_data.shape[0])
automobile_ship_data = automobile_ship_data[p]
automobile_ship_labels = automobile_ship_labels[p]
# save to file
dict = {'data': automobile_ship_data, 'labels': automobile_ship_labels}
output = open('automobile_ship', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create automobile/truck data set
automobile_truck_data = np.vstack((training_data[1], training_data[9]))
automobile_truck_labels = np.hstack((np.zeros(training_data[1].shape[0]), np.ones(training_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(automobile_truck_data.shape[0])
automobile_truck_data = automobile_truck_data[p]
automobile_truck_labels = automobile_truck_labels[p]
# save to file
dict = {'data': automobile_truck_data, 'labels': automobile_truck_labels}
output = open('automobile_truck', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create ship/truck data set
ship_truck_data = np.vstack((training_data[8], training_data[9]))
ship_truck_labels = np.hstack((np.zeros(training_data[8].shape[0]), np.ones(training_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(ship_truck_data.shape[0])
ship_truck_data = ship_truck_data[p]
ship_truck_labels = ship_truck_labels[p]
# save to file
dict = {'data': ship_truck_data, 'labels': ship_truck_labels}
output = open('ship_truck', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/cat data set
bird_cat_data = np.vstack((training_data[2], training_data[3]))
bird_cat_labels = np.hstack((np.zeros(training_data[2].shape[0]), np.ones(training_data[3].shape[0])))
# shuffle all images
p = np.random.permutation(bird_cat_data.shape[0])
bird_cat_data = bird_cat_data[p]
bird_cat_labels = bird_cat_labels[p]
# save to file
dict = {'data': bird_cat_data, 'labels': bird_cat_labels}
output = open('bird_cat', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/frog data set
bird_frog_data = np.vstack((training_data[2], training_data[6]))
bird_frog_labels = np.hstack((np.zeros(training_data[2].shape[0]), np.ones(training_data[6].shape[0])))
# shuffle all images
p = np.random.permutation(bird_frog_data.shape[0])
bird_frog_data = bird_frog_data[p]
bird_frog_labels = bird_frog_labels[p]
# save to file
dict = {'data': bird_frog_data, 'labels': bird_frog_labels}
output = open('bird_frog', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/horse data set
bird_horse_data = np.vstack((training_data[2], training_data[7]))
bird_horse_labels = np.hstack((np.zeros(training_data[2].shape[0]), np.ones(training_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(bird_horse_data.shape[0])
bird_horse_data = bird_horse_data[p]
bird_horse_labels = bird_horse_labels[p]
# save to file
dict = {'data': bird_horse_data, 'labels': bird_horse_labels}
output = open('bird_horse', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create cat/frog data set
cat_frog_data = np.vstack((training_data[3], training_data[6]))
cat_frog_labels = np.hstack((np.zeros(training_data[3].shape[0]), np.ones(training_data[6].shape[0])))
# shuffle all images
p = np.random.permutation(cat_frog_data.shape[0])
cat_frog_data = cat_frog_data[p]
cat_frog_labels = cat_frog_labels[p]
# save to file
dict = {'data': cat_frog_data, 'labels': cat_frog_labels}
output = open('cat_frog', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create cat/horse data set
cat_horse_data = np.vstack((training_data[3], training_data[7]))
cat_horse_labels = np.hstack((np.zeros(training_data[3].shape[0]), np.ones(training_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(cat_horse_data.shape[0])
cat_horse_data = cat_horse_data[p]
cat_horse_labels = cat_horse_labels[p]
# save to file
dict = {'data': cat_horse_data, 'labels': cat_horse_labels}
output = open('cat_horse', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create frog/horse data set
frog_horse_data = np.vstack((training_data[6], training_data[7]))
frog_horse_labels = np.hstack((np.zeros(training_data[6].shape[0]), np.ones(training_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(frog_horse_data.shape[0])
frog_horse_data = frog_horse_data[p]
frog_horse_labels = frog_horse_labels[p]
# save to file
dict = {'data': frog_horse_data, 'labels': frog_horse_labels}
output = open('frog_horse', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

#---------------------------- VALIDATION DATA --------------------------------#

validation_data_array = np.vstack(batch5['data'])
validation_labels_array = np.hstack(batch5['labels'])

# Extract images of each label from the validation set
validation_data = []
for i in range(0, 10):
    validation_data.append(np.take(validation_data_array, (validation_labels_array == i).nonzero(), axis=0)[0])

# Create airplane/automobile data set
airplane_automobile_data = np.vstack((validation_data[0], validation_data[1]))
airplane_automobile_labels = np.hstack((np.zeros(validation_data[0].shape[0]), np.ones(validation_data[1].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_automobile_data.shape[0])
airplane_automobile_data = airplane_automobile_data[p]
airplane_automobile_labels = airplane_automobile_labels[p]
# save to file
dict = {'data': airplane_automobile_data, 'labels': airplane_automobile_labels}
output = open('airplane_automobile_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create airplane/ship data set
airplane_ship_data = np.vstack((validation_data[0], validation_data[8]))
airplane_ship_labels = np.hstack((np.zeros(validation_data[0].shape[0]), np.ones(validation_data[8].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_ship_data.shape[0])
airplane_ship_data = airplane_ship_data[p]
airplane_ship_labels = airplane_ship_labels[p]
# save to file
dict = {'data': airplane_ship_data, 'labels': airplane_ship_labels}
output = open('airplane_ship_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create airplane/truck data set
airplane_truck_data = np.vstack((validation_data[0], validation_data[9]))
airplane_truck_labels = np.hstack((np.zeros(validation_data[0].shape[0]), np.ones(validation_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_truck_data.shape[0])
airplane_truck_data = airplane_truck_data[p]
airplane_truck_labels = airplane_truck_labels[p]
# save to file
dict = {'data': airplane_truck_data, 'labels': airplane_truck_labels}
output = open('airplane_truck_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create automobile/ship data set
automobile_ship_data = np.vstack((validation_data[1], validation_data[8]))
automobile_ship_labels = np.hstack((np.zeros(validation_data[1].shape[0]), np.ones(validation_data[8].shape[0])))
# shuffle all images
p = np.random.permutation(automobile_ship_data.shape[0])
automobile_ship_data = automobile_ship_data[p]
automobile_ship_labels = automobile_ship_labels[p]
# save to file
dict = {'data': automobile_ship_data, 'labels': automobile_ship_labels}
output = open('automobile_ship_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create automobile/truck data set
automobile_truck_data = np.vstack((validation_data[1], validation_data[9]))
automobile_truck_labels = np.hstack((np.zeros(validation_data[1].shape[0]), np.ones(validation_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(automobile_truck_data.shape[0])
automobile_truck_data = automobile_truck_data[p]
automobile_truck_labels = automobile_truck_labels[p]
# save to file
dict = {'data': automobile_truck_data, 'labels': automobile_truck_labels}
output = open('automobile_truck_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create ship/truck data set
ship_truck_data = np.vstack((validation_data[8], validation_data[9]))
ship_truck_labels = np.hstack((np.zeros(validation_data[8].shape[0]), np.ones(validation_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(ship_truck_data.shape[0])
ship_truck_data = ship_truck_data[p]
ship_truck_labels = ship_truck_labels[p]
# save to file
dict = {'data': ship_truck_data, 'labels': ship_truck_labels}
output = open('ship_truck_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/cat data set
bird_cat_data = np.vstack((validation_data[2], validation_data[3]))
bird_cat_labels = np.hstack((np.zeros(validation_data[2].shape[0]), np.ones(validation_data[3].shape[0])))
# shuffle all images
p = np.random.permutation(bird_cat_data.shape[0])
bird_cat_data = bird_cat_data[p]
bird_cat_labels = bird_cat_labels[p]
# save to file
dict = {'data': bird_cat_data, 'labels': bird_cat_labels}
output = open('bird_cat_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/frog data set
bird_frog_data = np.vstack((validation_data[2], validation_data[6]))
bird_frog_labels = np.hstack((np.zeros(validation_data[2].shape[0]), np.ones(validation_data[6].shape[0])))
# shuffle all images
p = np.random.permutation(bird_frog_data.shape[0])
bird_frog_data = bird_frog_data[p]
bird_frog_labels = bird_frog_labels[p]
# save to file
dict = {'data': bird_frog_data, 'labels': bird_frog_labels}
output = open('bird_frog_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/horse data set
bird_horse_data = np.vstack((validation_data[2], validation_data[7]))
bird_horse_labels = np.hstack((np.zeros(validation_data[2].shape[0]), np.ones(validation_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(bird_horse_data.shape[0])
bird_horse_data = bird_horse_data[p]
bird_horse_labels = bird_horse_labels[p]
# save to file
dict = {'data': bird_horse_data, 'labels': bird_horse_labels}
output = open('bird_horse_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create cat/frog data set
cat_frog_data = np.vstack((validation_data[3], validation_data[6]))
cat_frog_labels = np.hstack((np.zeros(validation_data[3].shape[0]), np.ones(validation_data[6].shape[0])))
# shuffle all images
p = np.random.permutation(cat_frog_data.shape[0])
cat_frog_data = cat_frog_data[p]
cat_frog_labels = cat_frog_labels[p]
# save to file
dict = {'data': cat_frog_data, 'labels': cat_frog_labels}
output = open('cat_frog_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create cat/horse data set
cat_horse_data = np.vstack((validation_data[3], validation_data[7]))
cat_horse_labels = np.hstack((np.zeros(validation_data[3].shape[0]), np.ones(validation_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(cat_horse_data.shape[0])
cat_horse_data = cat_horse_data[p]
cat_horse_labels = cat_horse_labels[p]
# save to file
dict = {'data': cat_horse_data, 'labels': cat_horse_labels}
output = open('cat_horse_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create frog/horse data set
frog_horse_data = np.vstack((validation_data[6], validation_data[7]))
frog_horse_labels = np.hstack((np.zeros(validation_data[6].shape[0]), np.ones(validation_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(frog_horse_data.shape[0])
frog_horse_data = frog_horse_data[p]
frog_horse_labels = frog_horse_labels[p]
# save to file
dict = {'data': frog_horse_data, 'labels': frog_horse_labels}
output = open('frog_horse_validation', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

#------------------------------- TEST DATA -----------------------------------#

test_data_array = np.vstack(test_batch['data'])
test_labels_array = np.hstack(test_batch['labels'])

# Extract images of each label from the test set
test_data = []
for i in range(0, 10):
    test_data.append(np.take(test_data_array, (test_labels_array == i).nonzero(), axis=0)[0])

# Create airplane/automobile data set
airplane_automobile_data = np.vstack((test_data[0], test_data[1]))
airplane_automobile_labels = np.hstack((np.zeros(test_data[0].shape[0]), np.ones(test_data[1].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_automobile_data.shape[0])
airplane_automobile_data = airplane_automobile_data[p]
airplane_automobile_labels = airplane_automobile_labels[p]
# save to file
dict = {'data': airplane_automobile_data, 'labels': airplane_automobile_labels}
output = open('airplane_automobile_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create airplane/ship data set
airplane_ship_data = np.vstack((test_data[0], test_data[8]))
airplane_ship_labels = np.hstack((np.zeros(test_data[0].shape[0]), np.ones(test_data[8].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_ship_data.shape[0])
airplane_ship_data = airplane_ship_data[p]
airplane_ship_labels = airplane_ship_labels[p]
# save to file
dict = {'data': airplane_ship_data, 'labels': airplane_ship_labels}
output = open('airplane_ship_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create airplane/truck data set
airplane_truck_data = np.vstack((test_data[0], test_data[9]))
airplane_truck_labels = np.hstack((np.zeros(test_data[0].shape[0]), np.ones(test_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(airplane_truck_data.shape[0])
airplane_truck_data = airplane_truck_data[p]
airplane_truck_labels = airplane_truck_labels[p]
# save to file
dict = {'data': airplane_truck_data, 'labels': airplane_truck_labels}
output = open('airplane_truck_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create automobile/ship data set
automobile_ship_data = np.vstack((test_data[1], test_data[8]))
automobile_ship_labels = np.hstack((np.zeros(test_data[1].shape[0]), np.ones(test_data[8].shape[0])))
# shuffle all images
p = np.random.permutation(automobile_ship_data.shape[0])
automobile_ship_data = automobile_ship_data[p]
automobile_ship_labels = automobile_ship_labels[p]
# save to file
dict = {'data': automobile_ship_data, 'labels': automobile_ship_labels}
output = open('automobile_ship_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create automobile/truck data set
automobile_truck_data = np.vstack((test_data[1], test_data[9]))
automobile_truck_labels = np.hstack((np.zeros(test_data[1].shape[0]), np.ones(test_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(automobile_truck_data.shape[0])
automobile_truck_data = automobile_truck_data[p]
automobile_truck_labels = automobile_truck_labels[p]
# save to file
dict = {'data': automobile_truck_data, 'labels': automobile_truck_labels}
output = open('automobile_truck_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create ship/truck data set
ship_truck_data = np.vstack((test_data[8], test_data[9]))
ship_truck_labels = np.hstack((np.zeros(test_data[8].shape[0]), np.ones(test_data[9].shape[0])))
# shuffle all images
p = np.random.permutation(ship_truck_data.shape[0])
ship_truck_data = ship_truck_data[p]
ship_truck_labels = ship_truck_labels[p]
# save to file
dict = {'data': ship_truck_data, 'labels': ship_truck_labels}
output = open('ship_truck_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/cat data set
bird_cat_data = np.vstack((test_data[2], test_data[3]))
bird_cat_labels = np.hstack((np.zeros(test_data[2].shape[0]), np.ones(test_data[3].shape[0])))
# shuffle all images
p = np.random.permutation(bird_cat_data.shape[0])
bird_cat_data = bird_cat_data[p]
bird_cat_labels = bird_cat_labels[p]
# save to file
dict = {'data': bird_cat_data, 'labels': bird_cat_labels}
output = open('bird_cat_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/frog data set
bird_frog_data = np.vstack((test_data[2], test_data[6]))
bird_frog_labels = np.hstack((np.zeros(test_data[2].shape[0]), np.ones(test_data[6].shape[0])))
# shuffle all images
p = np.random.permutation(bird_frog_data.shape[0])
bird_frog_data = bird_frog_data[p]
bird_frog_labels = bird_frog_labels[p]
# save to file
dict = {'data': bird_frog_data, 'labels': bird_frog_labels}
output = open('bird_frog_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create bird/horse data set
bird_horse_data = np.vstack((test_data[2], test_data[7]))
bird_horse_labels = np.hstack((np.zeros(test_data[2].shape[0]), np.ones(test_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(bird_horse_data.shape[0])
bird_horse_data = bird_horse_data[p]
bird_horse_labels = bird_horse_labels[p]
# save to file
dict = {'data': bird_horse_data, 'labels': bird_horse_labels}
output = open('bird_horse_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create cat/frog data set
cat_frog_data = np.vstack((test_data[3], test_data[6]))
cat_frog_labels = np.hstack((np.zeros(test_data[3].shape[0]), np.ones(test_data[6].shape[0])))
# shuffle all images
p = np.random.permutation(cat_frog_data.shape[0])
cat_frog_data = cat_frog_data[p]
cat_frog_labels = cat_frog_labels[p]
# save to file
dict = {'data': cat_frog_data, 'labels': cat_frog_labels}
output = open('cat_frog_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create cat/horse data set
cat_horse_data = np.vstack((test_data[3], test_data[7]))
cat_horse_labels = np.hstack((np.zeros(test_data[3].shape[0]), np.ones(test_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(cat_horse_data.shape[0])
cat_horse_data = cat_horse_data[p]
cat_horse_labels = cat_horse_labels[p]
# save to file
dict = {'data': cat_horse_data, 'labels': cat_horse_labels}
output = open('cat_horse_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

# Create frog/horse data set
frog_horse_data = np.vstack((test_data[6], test_data[7]))
frog_horse_labels = np.hstack((np.zeros(test_data[6].shape[0]), np.ones(test_data[7].shape[0])))
# shuffle all images
p = np.random.permutation(frog_horse_data.shape[0])
frog_horse_data = frog_horse_data[p]
frog_horse_labels = frog_horse_labels[p]
# save to file
dict = {'data': frog_horse_data, 'labels': frog_horse_labels}
output = open('frog_horse_test', 'ab+')
cPickle.dump(dict, output, protocol=cPickle.HIGHEST_PROTOCOL)
output.close()

b1.close()
b2.close()
b3.close()
b4.close()
b5.close()
tb.close()

