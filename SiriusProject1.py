from flask import Flask, render_template, request, send_file, redirect, url_for
import os

###     FLASK    ###


app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def home():

    if request.method == 'POST':
        input_image = request.files['input']
        style_image = request.files['style']

        input_image.save(os.path.join('C:/Users/George/PycharmProjects/SiriusProject1/static/images/input.jpg'))
        style_image.save(os.path.join('C:/Users/George/PycharmProjects/SiriusProject1/static/images/style.jpg'))

        return redirect( url_for('result', _anchor='result'))

    return render_template('index.html')

@app.route('/result')
def result():
    #run()
    return render_template('index1.html')


if __name__ == '__main__':
    app.run()
