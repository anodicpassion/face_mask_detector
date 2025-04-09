#!/bin/bash
curl -L -o ./face-mask-dataset.zip https://www.kaggle.com/api/v1/datasets/download/omkargurav/face-mask-dataset
unzip face-mask-dataset.zip
rm ./face-mask-dataset.zip