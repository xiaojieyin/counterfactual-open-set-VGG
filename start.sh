#!/bin/bash
# Break on any error
set -e

image_size=128
RESULT_DIR=/home/sdb2/yinxiaojie/osrci_vgg/$image_size
#RESULT_DIR=./checkpoint/$image_size

# Hyperparameters
GAN_EPOCHS=100
CLASSIFIER_EPOCHS=30
CF_COUNT=50

:<<!
# Download any datasets not currently available
# TODO: do this in python, based on --dataset
if [ ! -f $DATASET_DIR/svhn-split0a.dataset ]; then
    python generativeopenset/datasets/download_svhn.py
fi
if [ ! -f $DATASET_DIR/cifar10-split0a.dataset ]; then
    python generativeopenset/datasets/download_cifar10.py
fi
if [ ! -f $DATASET_DIR/mnist-split0a.dataset ]; then
    python generativeopenset/datasets/download_mnist.py
fi
if [ ! -f $DATASET_DIR/oxford102.dataset ]; then
    python generativeopenset/datasets/download_oxford102.py
fi
if [ ! -f $DATASET_DIR/celeba.dataset ]; then
    python generativeopenset/datasets/download_celeba.py
fi
if [ ! -f $DATASET_DIR/cifar100-animals.dataset ]; then
    python generativeopenset/datasets/download_cifar100.py
fi
!

if [ ! -d  "$RESULT_DIR" ]; then
    mkdir $RESULT_DIR
fi
if [ ! -f $RESULT_DIR/log.txt ]; then
    touch $RESULT_DIR/log.txt
fi


# Train the intial generative model (E+G+D) and the initial classifier (C_K)
python src/train_gan.py --result_dir ${RESULT_DIR} --epochs $GAN_EPOCHS --mode baseline --image_size $image_size | tee $RESULT_DIR/log.txt

# Baseline: Evaluate the standard classifier (C_k+1)
python src/evaluate_classifier.py --result_dir ${RESULT_DIR} --mode baseline --image_size $image_size | tee -a $RESULT_DIR/log.txt
python src/evaluate_classifier.py --result_dir ${RESULT_DIR} --mode weibull --image_size $image_size | tee -a $RESULT_DIR/log.txt

cp ${RESULT_DIR}/checkpoints/classifier_k_best.pth ${RESULT_DIR}/checkpoints/classifier_kplusone_best.pth


GENERATOR_MODE=ge_et_al  # g-opennmax
# Generate a number of counterfactual images (in the K+2 by K+2 square grid format)
python src/generate_${GENERATOR_MODE}.py --result_dir ${RESULT_DIR} --count $CF_COUNT --image_size $image_size

# Automatically label the rightmost column in each grid (ignore the others)
python src/auto_label.py --result_dir ${RESULT_DIR} --mode $GENERATOR_MODE --output_filename ${RESULT_DIR}/generated_images_${GENERATOR_MODE}.dataset --image_size $image_size

# Train a new classifier, now using the aux_dataset containing the counterfactuals
python src/train_classifier.py --result_dir ${RESULT_DIR} --epochs $CLASSIFIER_EPOCHS --aux_dataset ${RESULT_DIR}/generated_images_${GENERATOR_MODE}.dataset --image_size $image_size | tee -a $RESULT_DIR/log.txt

# Evaluate the C_K+1 classifier, trained with the augmented data
python src/evaluate_classifier.py --result_dir ${RESULT_DIR} --mode gen --image_size $image_size --epoch -1 | tee -a $RESULT_DIR/log.txt


GENERATOR_MODE=open_set  # fuxin
# Generate a number of counterfactual images (in the K+2 by K+2 square grid format)
python src/generate_${GENERATOR_MODE}.py --result_dir ${RESULT_DIR} --count $CF_COUNT --image_size $image_size

# Automatically label the rightmost column in each grid (ignore the others)
python src/auto_label.py --result_dir ${RESULT_DIR} --mode $GENERATOR_MODE --output_filename ${RESULT_DIR}/generated_images_${GENERATOR_MODE}.dataset --image_size $image_size

# Train a new classifier, now using the aux_dataset containing the counterfactuals
python src/train_classifier.py --result_dir ${RESULT_DIR} --epochs $CLASSIFIER_EPOCHS --aux_dataset ${RESULT_DIR}/generated_images_${GENERATOR_MODE}.dataset --image_size $image_size | tee -a $RESULT_DIR/log.txt

# Evaluate the C_K+1 classifier, trained with the augmented data
python src/evaluate_classifier.py --result_dir ${RESULT_DIR} --mode fuxin --image_size $image_size --epoch -1 | tee -a $RESULT_DIR/log.txt

#./print_results.sh
