import matplotlib.pyplot as plt



import os
import numpy as np
import pickle as pkl
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from io import StringIO

import pandas as pd
from collections import defaultdict
import matplotlib.pyplot as plt
import random

FRAMEWORKS = ['angular', 'react', 'node', 'vue', 'meteor', 'polymer']
months_of_experience = [12, 8, 34, 51, 16, 12, 13]
years_of_experience = [round(exp / 12) for exp in months_of_experience]

Y = [1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0]

def convert_pdf_to_txt(path):
    rsrcmgr = PDFResourceManager()
    retstr = StringIO()
    codec = 'utf-8'
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, codec=codec, laparams=laparams)
    fp = open(path, 'rb')
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    password = ""
    maxpages = 0
    caching = True
    pagenos=set()

    for page in PDFPage.get_pages(fp, pagenos, maxpages=maxpages, password=password, caching=caching, check_extractable=True):
        interpreter.process_page(page)

    text = retstr.getvalue()

    fp.close()
    device.close()
    retstr.close()

    # Robert`s code
    lines = text.split('\n')
    filtered_lines = [line for line in lines if line not in ['', "\n"]]
    return filtered_lines, text



def get_role_occurence(role, lines):
    occ = 0
    for line in lines:
        if role in line.lower():
            occ += 1
    return occ

def get_years_of_experience(text):
    words = text.split()
    words = [word.replace('(', '') for word in words]
    words = [word.replace(')', '') for word in words]

    experience = 0
    for index in range(len(words)):
        if 'an' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += 12 * aux
            except:
                pass
        if 'lun' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += aux
            except:
                pass
        if 'year' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += 12 * aux
            except:
                pass
        if 'month' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += aux
            except:
                pass

    feature = []
    if experience < 12:
        return [1, 0, 0, 0]
    elif experience >= 12 and experience < 24:
        return [0, 1, 0, 0]
    elif experience > 24 and experience < 36:
        return [0, 0, 1, 0]

    return [0, 0, 0, 1]

def get_years_of_experience_robert(text):
    words = text.split()
    words = [word.replace('(', '') for word in words]
    words = [word.replace(')', '') for word in words]

    experience = 0
    for index in range(len(words)):
        if 'an' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += 12 * aux
            except:
                pass
        if 'lun' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += aux
            except:
                pass
        if 'year' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += 12 * aux
            except:
                pass
        if 'month' in  words[index] and index != 0:
            try:
                aux = int(words[index - 1])
                experience += aux
            except:
                pass

    if experience > 200:
       return random.randrange(100, 200)

    return experience

def get_features_framework(text):
    words = text.split()
    words = [word.replace('(', '') for word in words]
    words = [word.replace(')', '') for word in words]
    words = [word.lower() for word in words]

    features = [0, 0, 0, 0, 0, 0]
    for word in words:
        if 'angular' in word:
            features[0] = 1
        if 'react' in word:
            features[1] = 1
        if 'node' in word:
            features[2] = 1
        if 'vue' in word:
            features[3] = 1
        if 'meteor' in word:
            features[4] = 1
        if 'polymer' in word:
            features[5] = 1

    return features

def train():
    X = []
    current_directory = os.getcwd()
    data_directory = os.path.join(current_directory, 'train_data')

    for profile in os.listdir(data_directory):
        file_dir = os.path.join(data_directory, profile)
        print(file_dir)
        profile_lines, text = convert_pdf_to_txt(file_dir)
        feature_list = []


        feature_list.append(get_role_occurence('java', profile_lines))
        feature_list += get_years_of_experience(text)
        feature_list += get_features_framework(text)
        X.append(feature_list)

    X = np.asarray(X)
    y = np.asarray(Y)

    model = RandomForestClassifier()
    model.fit(X, y)

    with open('rand_forest_model.pkl', 'wb') as file:
        print ('tiiiiiip')
        pkl.dump(model, file)


def predict_and_plot():
    X = []
    data_directory = 'C:\\Users\\Gicu\\Downloads\\'
    experience = []
    people = []
    for profile in os.listdir(data_directory):
        if '.pdf' in profile:
            file_dir = os.path.join(data_directory, profile)
            print(file_dir)
            profile_lines, text = convert_pdf_to_txt(file_dir)
            feature_list = []
            all_words = [item for item in text.split()]
            people.append([item for item in all_words if 'linkedin.com' in item][0])

            #real_role = role.split()
            feature_list.append(get_role_occurence('javascript', profile_lines))
            feature_list += get_years_of_experience(text)
            feature_list += get_features_framework(text)

            experience.append(get_years_of_experience_robert(text))

            X.append(feature_list)

    experience = [round(item / 12) for item in experience]
    experience.sort()
    write_exp_hist(experience)
    plot_popular_frameworks('angular, angular fdsf node meteor')

    X = np.asarray(X)

    model_path = 'C:\\Users\\Gicu\\Downloads\\rand_forest_model.pkl' 
    with open(model_path, 'rb') as file:
        model = pkl.load(file)

    return str(list(zip(people, model.predict(X))))

def write_exp_hist(experience):
    hist = defaultdict(int)

    for element in experience:
        hist[element] = 0

    for element in experience:
        hist[element] += 1

    df = pd.DataFrame.from_dict(hist, orient='index')
    df.plot(
        kind='bar',
        title='Number of Developers Grouped by Years of Experience',
    )
    plt.show()


def plot_popular_frameworks(lines_string):
    hist = defaultdict(int)

    for element in FRAMEWORKS:
        hist[element] = lines_string.lower().count(element)

    df = pd.DataFrame.from_dict(hist, orient='index')
    df.plot(
        kind='bar',
        title='Usage of Javascript Popular Frameworks',
    )
    plt.show()

def print_line(arg):
    a = predict_and_plot()
    return str(a)
