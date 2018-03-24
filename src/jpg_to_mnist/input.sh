#!/bin/bash

rm -r $(dirname $0)/test-images/*
rm -r $(dirname $0)/training-images/*


list=( "glass" "metal" "paper" "plastic")
count=0
for name in "${list[@]}"
do

  echo "copy ${name}" 
  TRAINING_DIR=$(dirname $0)/training-images/${count}
  TEST_DIR=$(dirname $0)/test-images/${count}


  #populate training images
  mkdir $TRAINING_DIR
  cp -r ../../data/dataset-resized/${name}/*  $TRAINING_DIR/

  #populate validation images
  mkdir $TEST_DIR

  ls $TRAINING_DIR | sort -R | tail -n 50 | while read file; do
      cp $TRAINING_DIR/$file $TEST_DIR/$file
  done

  #rename the files in test images correctly
  ls $TEST_DIR | cat -n | while read n f; do mv "$TEST_DIR/$f" "$TEST_DIR/$n.jpg"; done

  ((count++))
done
