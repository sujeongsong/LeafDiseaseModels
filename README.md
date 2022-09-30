# Leaf Disease Classification Models
This is the proposal of project to identify the type of disease present on a Cassava Leaf image as an application of neural networks.

## Introduction
As the second-largest provider of carbohydrates in Africa, cassava is a key food security crop grown by smallholder farmers because it can withstand harsh conditions. At least 80% of household farms in Sub-Saharan Africa grow this starchy root, but viral diseases are major sources of poor yields. With the help of data science, it may be possible to identify common diseases so they can be treated. Existing methods of disease detection require farmers to solicit the help of government-funded agricultural experts to visually inspect and diagnose the plants. This suffers from being labor-intensive, low-supply and costly. As a proposed project, effective solutions for farmers must perform well under significant constraints, since African farmers may only have access to mobile-quality cameras with low-bandwidth. Your task is to classify each cassava image into four disease categories or a fifth category indicating a healthy leaf. With your help, farmers may be able to quickly identify diseased plants, potentially saving their crops before they inflict irreparable damage. 

## Dataset Description
A dataset of 21,367 labeled images are introduced, those images were collected during a regular survey in Uganda. Most images were crowdsourced from farmers taking photos of their gardens, and annotated by experts at the National Crops Resources Research Institute (NaCRRI) in collaboration with the AI lab at Makerere University, Kampala. This is in a format that most realistically represents what farmers would need to diagnose in real life.


### Files

**train_images** the image file. 

**train.csv** the csv file including **image_id** and **label**.

**train_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

### Download
You can easily download the dataset using the kaggle API, or just go to the URL below. 

**API command**: kaggle competitions download -c cassava-leaf-disease-classification

**URL for dataset**: https://www.kaggle.com/competitions/cassava-leaf-disease-classification/data
