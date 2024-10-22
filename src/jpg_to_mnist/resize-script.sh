#!/bin/bash

#simple script for resizing images in all class directories
#also reformats everything from whatever to png
size="56x56"

if [ `ls test-images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in test-images/*/*.png; do
    convert "$file" -resize ${size}\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls test-images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in test-images/*/*.jpg; do
    convert "$file" -resize ${size}\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi




if [ `ls training-images/*/*.png 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in training-images/*/*.png; do
    convert "$file" -resize ${size}\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi

if [ `ls training-images/*/*.jpg 2> /dev/null | wc -l ` -gt 0 ]; then
  echo hi
  for file in training-images/*/*.jpg; do
    convert "$file" -resize ${size}\! "${file%.*}.png"
    file "$file" #uncomment for testing
    rm "$file"
  done
fi


