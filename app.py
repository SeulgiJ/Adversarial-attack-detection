from flask import Flask, render_template, request
from attack import ad_attack
from print_input import print_input

app = Flask(__name__)

@app.route("/")
def image_get():
    return render_template('home.html')

@app.route("/plot.png", methods = ['POST', 'GET'])
def plot_png():
    if request.method == 'POST':
        result = request.form
    file_name = result.get("file_name")
    file_path = result.get("file_path")
    epsilon = float(result.get("epsilon"))
    attack_file, detection = ad_attack(file_name, file_path, epsilon)

    # return render_template('myplot.html', image_file=image_file, detection=detection)
    return render_template('myplot.html', input_file = print_input(file_name, file_path),
                           attack_file = attack_file, detection = detection)


if __name__ == "__main__":
    app.run()