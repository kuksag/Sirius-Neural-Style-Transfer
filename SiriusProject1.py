from flask import Flask, render_template, request, redirect, url_for, session
import os
from flask_caching import Cache

###     FLASK    ###



app = Flask(__name__)
cache = Cache(app, config={'CACHE_TYPE': 'null'})

i = 0

app.config['UPLOAD_FOLDER'] = 'C:/Users/George/PycharmProjects/SiriusProject1'
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.secret_key = b'_5#y2L"F4q8z\n\xec]/'

@app.context_processor
def override_url_for():
    """
    Generate a new token on every request to prevent the browser from
    caching static  files.
    """
    return dict(url_for=dated_url_for)

def dated_url_for(endpoint, **values):
    if endpoint == 'static':
        filename = values.get('filename', None)
        if filename:
            file_path = os.path.join(app.root_path,
                                     endpoint, filename)
            values['q'] = int(os.stat(file_path).st_mtime)
    return url_for(endpoint, **values)


@app.route('/', methods=['POST', 'GET'])
def home():

    if 'username' not in session:
        global i
        session['username'] = i
        i += 1

    if request.method == 'POST':

        input_image = request.files['input']
        os.makedirs(os.path.join(app.config['UPLOAD_FOLDER'] + '/static/images/' + str(session['username'])), exist_ok=True)
        input_image.save(os.path.join(app.config['UPLOAD_FOLDER'] + '/static/images/' +
                         str(session['username']) + '/input.jpg'))

        return redirect('/l1#work')

    return render_template('./extends.html', showres = 0, show1 = 0, show2 = 0, num = str(session['username']),
                           form1 = 1, form2 = 0)

@app.route('/l1', methods=['POST', 'GET'])
def load1():

    if request.method == 'POST':
        session['choose'] = request.form['slt']
        return redirect('/result')

    return render_template('./extends.html', showres=0, show1=1, show2=0, num = str(session['username']),
                           form1=0, form2=1)

@app.route('/result')
def end():
    print(session['choose'])
    print(session['username'])
    run_name = app.config['UPLOAD_FOLDER'] + '/neural.py'
    model_name = app.config['UPLOAD_FOLDER'] + '/models/' + str(session['choose']) + '.pth'
    in_image_path = app.config['UPLOAD_FOLDER'] + '/static/images/' + str(session['username']) + '/input.jpg'
    output_image_path = app.config['UPLOAD_FOLDER'] + '/static/images/' + str(session['username']) + '/result.jpg'

    os.system('python {} eval --cuda 0 --content-image {} --model {} --output-image {}'.format(run_name, in_image_path, model_name, output_image_path))
    return redirect('/show#result')

@app.route('/show')
def show():
    return render_template('./extends.html', showres=1, show1=1, show2=1, form1=0, form2=0,
                           num=str(session['username']), style_num=session['choose'],
                           path=url_for('static', filename='images/' + str(session['username']) + '/result.jpg'))

if __name__ == '__main__':
    app.run()
