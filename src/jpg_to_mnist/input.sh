#!/bin/bash

# rm -r ./test-images/*
# rm -r ./training-images/*


list=("cardboard" "glass" "metal" "paper" "plastic" "trash")
count=0
for name in "${list[@]}"
do

  #populate training images
  mkdir ./training-images/${count}
  cp -r ../../data/dataset-resized/${name}/*  ./training-images/${count}/


  #populate validation images
  # mkdir ./test-images/${count}
  # cp -r ../../data/dataset-resized/${name}/*  ./test-images/${count}/

  #drop some images from test
  # if [ "$(ls test-images/*/*.jpg 2> /dev/null | wc -l )" -gt 0 ]; then
  #   for file in test-images/*/*.jpg; do
  #     convert "$file" -resize 256x256\! "${file%.*}.png"
  #     file "$file" #uncomment for testing
  #     rm "$file"
  #   done
  # fi
  ((count++))
done
