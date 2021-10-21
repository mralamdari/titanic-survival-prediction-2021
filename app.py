import os, flask, pickle, numpy as np, sklearn

#  pip freeze > requirements.txt
app = flask.Flask(__name__)
app.config["SECRET_KEY"] = "mykey"
img_folder = os.path.join('static', 'img')
app.config['UPLOAD_FOLDER'] = img_folder

@app.route("/")
def index():
    titanic_2 = os.path.join(app.config['UPLOAD_FOLDER'], 'titanic_2.jpg')
    return flask.render_template("index.html", titanic_2=titanic_2)


@app.route("/help")
def help():
    titanic_3 = os.path.join(app.config['UPLOAD_FOLDER'], 'titanic_3.jpg')
    return flask.render_template("help.html", titanic_3=titanic_3)


@app.route("/about")
def about():
    titanic_4 = os.path.join(app.config['UPLOAD_FOLDER'], 'titanic_4.jpg')
    return flask.render_template("about.html", titanic_4=titanic_4)

@app.route("/results", methods=["GET", "POST"])
def results():
    titanic_1 = os.path.join(app.config['UPLOAD_FOLDER'], 'titanic_1.jpg')
    titanic_2 = os.path.join(app.config['UPLOAD_FOLDER'], 'titanic_2.jpg')

    try:
        ticket = np.int64(flask.request.args.get("ticket"))
        sex = np.int64(flask.request.args.get("sex"))
        age = np.int64(flask.request.args.get("age"))
        siblings = np.int64(flask.request.args.get("siblings"))
        spouces = np.int64(flask.request.args.get("spouces"))
        parents = np.int64(flask.request.args.get("parents"))
        children = np.int64(flask.request.args.get("children"))
        fare = np.int64(flask.request.args.get("fare"))
        embarked = np.int64(flask.request.args.get("embarked"))
        SibSp = siblings + spouces
        Parch = parents + children

        input_list = np.array([ticket, sex, age, SibSp, Parch, fare, embarked]).reshape(-1, 1)
        loadm = pickle.load(open("h3_best_model.sav", "rb"))
        pred=loadm.predict(input_list.T)
    except TypeError:
        return flask.render_template("index.html", titanic_2=titanic_2)

    if pred[0] == 1:
        res = "Congratulations, If you were in The Titanic, you would probably survive"
    else:
        res = "My Condolences, If you were in The Titanic, you probably wouldn't survive"         
    
    return flask.render_template("results.html", titanic_1=titanic_1, res=res)

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=os.environ.get("PORT", 5000))
