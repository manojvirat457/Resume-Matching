# from scripts import tabledef
# from scripts import forms
# from scripts import helpers
from flask import Flask, redirect, url_for, render_template, request, session
import json
import sys
import os
# import stripe
import pandas as pd
from werkzeug.utils import secure_filename
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import numpy as np
import pandas as pd

import jieba
import jieba.analyse

import csv
import ast

import sys
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.pdfpage import PDFPage
from pdfminer.converter import XMLConverter, HTMLConverter, TextConverter
from pdfminer.layout import LAParams
import io

app = Flask(__name__)
# app.secret_key = os.urandom(12)  # Generic key for dev purposes only

# stripe_keys = {
#   'secret_key': os.environ['secret_key'],
#   'publishable_key': os.environ['publishable_key']
# }

# stripe.api_key = stripe_keys['secret_key']

# Heroku
#from flask_heroku import Heroku
#heroku = Heroku(app)


# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #
@app.route('/', methods=['GET', 'POST'])
def login():
    # creating a pdf file object

    basepath = os.path.dirname(__file__)

    file_path = os.path.join(basepath, 'uploads', 'sample.pdf')

    fp = open(file_path, 'rb')
    rsrcmgr = PDFResourceManager()
    retstr = io.StringIO()
    laparams = LAParams()
    device = TextConverter(rsrcmgr, retstr, laparams=laparams)
    # Create a PDF interpreter object.
    interpreter = PDFPageInterpreter(rsrcmgr, device)
    # Process each page contained in the document.

    for page in PDFPage.get_pages(fp):
        interpreter.process_page(page)
        data = retstr.getvalue()

    print(data)

    return render_template('home.html', user="manoj")
    # return text


def getFile():
    Tk().withdraw()
    filename = askopenfilename()
    Tk.close()
    return filename


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'GET':
        # f = request.files['file']

        basepath = os.path.dirname(__file__)
        # file_path = os.path.join(
        #     basepath, 'uploads', secure_filename(f.filename))
        # f.save(file_path)

        file_path = os.path.join(basepath, 'uploads', 'test-upload.csv')

        df = pd.read_csv(file_path)

        seg_list01 = df['job-description']
        seg_list02 = df['your-resume']

        item01_list = seg_list01
        item01 = ','.join(item01_list)

        item02_list = seg_list02
        item02 = ','.join(item02_list)

        documents = [item01, item02]

        count_vectorizer = CountVectorizer()
        sparse_matrix = count_vectorizer.fit_transform(documents)

        doc_term_matrix = sparse_matrix.todense()

        df = pd.DataFrame(doc_term_matrix,
                          columns=count_vectorizer.get_feature_names(),
                          index=['item01', 'item02'])

        df.to_csv(os.path.join(basepath, 'uploads', 'result.csv'))

        read_file = pd.read_csv(os.path.join(basepath, 'uploads',
                                             'result.csv'))
        read_file.to_excel(os.path.join(basepath, 'uploads', 'result.xlsx'),
                           index=None,
                           header=True)

        answer = cosine_similarity(df, df)

        print("CSV Created Successfully")
        answer = pd.DataFrame(answer)

        answer = answer.iloc[[1], [0]].values[0]
        answer = round(float(answer), 4) * 100

        return "Your resume matched " + str(
            answer) + " %" + " of the job-description!"
    return None


# ======== Main ============================================================== #
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True)