#!/bin/bash

rm -r ./test-images/*
rm -r ./training-images/*


list=( "glass" "metal" "paper" "plastic")
count=0
for name in "${list[@]}"
do

  
  TRAINING_DIR=$(dirname $0)/training-images/${count}
  TEST_DIR=$(dirname $0)/test-images/${count}
  #populate training images
  mkdir ./training-images/${count}
  cp -r ../../data/dataset-resized/${name}/*  ./training-images/${count}/

  #populate validation images
  mkdir ./test-images/${count}


  ls $TRAINING_DIR |sort -R |tail 50 |while read file; do
      cp $TRAINING_DIR/$file $TEST_DIR/$file
  done


  ((count++))
done
