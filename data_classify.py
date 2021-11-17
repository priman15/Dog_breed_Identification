import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
import nltk
import string
import os
from nltk.corpus import wordnet as wn

def remove_punctuation(text):
    text=text.lower()
    for punctuation in string.punctuation:
        text=text.replace(punctuation,'')
    return text

def convert_list_to_long_string(targ_list):
    long_string = ""
    for item in targ_list:
        long_string += " " + item
    long_string = long_string[1:]
    return long_string

 
doggo = wn.synset('dog.n.01')
dog_list = list(set([w for s in doggo.closure(lambda s:s.hyponyms()) for w in s.lemma_names()]))
mod_dogList=[]
for dog in dog_list:
    mod_dogList.append(remove_punctuation(dog))
mod_dogList
ImageNet_ID=pd.read_csv("https://raw.githubusercontent.com/mf1024/ImageNet-Datasets-Downloader/master/classes_in_imagenet.csv")
ImageNet_ID['new_class_name']=ImageNet_ID['class_name'].str.replace('[^\W\S]','').str.replace(' ','').str.lower()

dogNet_ID = ImageNet_ID[ImageNet_ID["new_class_name"].isin(mod_dogList)]
NotdogNet_ID = ImageNet_ID[~ImageNet_ID["new_class_name"].isin(mod_dogList)]
#print(dogNet_ID.to_string())

imagenet_dog_class_ids = dogNet_ID["synid"].tolist()
imagenet_dog_class_names = dogNet_ID["class_name"].tolist()
imagenet_dog_class_ids_and_names_dict = dict(zip(imagenet_dog_class_ids, imagenet_dog_class_names))
#len(imagenet_dog_class_ids_and_names_dict)

imagenet_NotDog_class_ids = NotdogNet_ID["synid"].tolist()
imagenet_NotDog_class_names = NotdogNet_ID["class_name"].tolist()
imagenet_NotDog_class_ids_and_names_dict = dict(zip(imagenet_NotDog_class_ids, imagenet_NotDog_class_names))


dog_class_id_list = list(imagenet_dog_class_ids_and_names_dict.keys())
NotDog_class_id_list = list(imagenet_NotDog_class_ids_and_names_dict.keys())
dog_class_id_string = convert_list_to_long_string(dog_class_id_list)
NotDog_class_id_string = convert_list_to_long_string(NotDog_class_id_list)
text_file=open("dogID.txt","w")
text_file.write(dog_class_id_string)
text_file.close()
#print(dog_class_id_string)
#cmd='mkdir dog_images'
#os.system(cmd)

#!python downloader.py -data_root dog_images -use_class_list=True -class_list $dog_class_id_string -images_per_class 10`
#os.system(cmd)

