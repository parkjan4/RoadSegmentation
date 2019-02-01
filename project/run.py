import numpy as np
from helpers import *
from cnn_model1 import *
from cnn_model2 import *
from cnn_model3 import *


# If set to False, train the networks
charge_model = True
# If set to False, do the predictions
charge_predic = True

# Load the training set
root_dir = "data/training/"

image_dir = root_dir + "images/"
files = os.listdir(image_dir)
N = min(100, len(files))
imgs = np.asarray([load_image(image_dir + files[i]) for i in range(N)])

gt_dir = root_dir + "groundtruth/"
gt_imgs = np.asarray([load_image(gt_dir + files[i]) for i in range(N)])

# # Model Architecture
if not charge_model:
    model1 = create_model1(padding=16)
    model2 = create_model2(padding=24)
    model3 = create_model3(padding=16)
    
    # Define number of images to use for training
    n = 100

    # Extract patches from input images
    patch_size = 16 # each image is now 16 x 16 pixels

    # training set
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(n)]
    gt_patches = [img_crop(gt_imgs[i], patch_size, patch_size) for i in range(n)]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])
    gt_patches =  np.asarray([gt_patches[i][j] for i in range(len(gt_patches)) for j in range(len(gt_patches[i]))])

    # Convert groundtruth image pixel to labels {0, 1}
    labels_tr = np.asarray([value_to_class(np.mean(gt_patches[i])) for i in range(len(gt_patches))])
    labels_tr0 = [i for i, j in enumerate(labels_tr) if j == 0]
    labels_tr1 = [i for i, j in enumerate(labels_tr) if j == 1]

    # Train image data
    train = img_patches

    # Convert labels to "onehot" format
    num_classes = 2
    onehot_tr = to_onehot(labels_tr, num_classes)


# # Data Pre-Processing
if not charge_model:
    # Image padding by extending the 16x16 patch image by reflection
    train16 = image_padding(train, padding = 16)
    train24 = image_padding(train, padding = 24)


# # Train Network
if not charge_model:
    model1, history1 = train_network1(train16, onehot_tr, padding=16)
    model2, history2 = train_network2(train24, onehot_tr, padding=24)
    model3, history3 = train_network3(train16, onehot_tr, padding=16)
    save_model1(model1, "")
    save_model2(model2, "")
    save_model3(model3, "")


# # Create Submission

# Loaded a set of images
root_dir = "data/test_set_images/"

imgs = []
num_test_imgs = 50
for i in range(1, num_test_imgs + 1):
    image_dir = root_dir + "test_" + str(i) + "/"
    img = os.listdir(image_dir)
    imgs.append(load_image(image_dir + img[0]))

# Process Test Images
if not charge_predic:
    # Extract patches from input images
    patch_size = 16 
    img_patches = [img_crop(imgs[i], patch_size, patch_size) for i in range(len(imgs))]

    # Linearize list of patches
    img_patches = np.asarray([img_patches[i][j] for i in range(len(img_patches)) for j in range(len(img_patches[i]))])

    # Image padding
    test_set16 = image_padding(img_patches, padding = 16)
    test_set24 = image_padding(img_patches, padding = 24)


# Generate Predictions
if not charge_predic:
    model1 = create_model1(padding=16)
    model2 = create_model2(padding=24)
    model3 = create_model3(padding=16)

    model1.load_weights("weights1.hdf5")
    model2.load_weights("weights2.hdf5")
    model3.load_weights("weights3.hdf5")

    predictions1 = model1.predict(test_set16, batch_size=None, verbose=1, steps=None)
    predictions2 = model2.predict(test_set24, batch_size=None, verbose=1, steps=None)
    predictions3 = model3.predict(test_set16, batch_size=None, verbose=1, steps=None)


    # Force class probabilities into labels 1 or 0
    if predictions1.shape[1] == 1:
        temp = (predictions1 >= 0.5) * 1
        pred_array = np.empty(len(temp))
        for i in range(len(temp)):
            pred_array[i] = temp[i][0]
    else:
        pred_array = (predictions1[:,0] < predictions1[:,1]) * 1

    np.save("predictions_cnn1.npy", pred_array)
    
    # Same for predictions2
    if predictions2.shape[1] == 1:
        temp = (predictions2 >= 0.5) * 1
        pred_array = np.empty(len(temp))
        for i in range(len(temp)):
            pred_array[i] = temp[i][0]
    else:
        pred_array = (predictions2[:,0] < predictions2[:,1]) * 1

    np.save("predictions_cnn2.npy", pred_array)
    
    # Same for predictions3
    if predictions3.shape[1] == 1:
        temp = (predictions3 >= 0.5) * 1
        pred_array = np.empty(len(temp))
        for i in range(len(temp)):
            pred_array[i] = temp[i][0]
    else:
        pred_array = (predictions3[:,0] < predictions3[:,1]) * 1

    np.save("predictions_cnn3.npy", pred_array)

if charge_predic:

    # Ensemble model
    predictions_cnn1 = np.load("predictions_cnn1.npy")
    predictions_cnn2 = np.load("predictions_cnn2.npy")
    predictions_cnn3 = np.load("predictions_cnn3.npy")

consolidated = np.c_[predictions_cnn1, predictions_cnn2, predictions_cnn3]
ensemble = ensemble_model(consolidated)

# Output as a .csv file 
submission = create_submission("submission.csv", ensemble)