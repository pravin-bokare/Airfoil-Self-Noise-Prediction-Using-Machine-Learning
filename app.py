import pickle
from flask import Flask, request, render_template
import numpy as np

app = Flask(__name__)
model = pickle.load(open('reg_model', 'rb'))


@app.route('/')
def home():
    # return 'Hello World'
    return render_template('home.html')


@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    output = model.predict(final_features)[0]
    print(output)
    # output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(round(output, 2)))


if __name__ == "__main__":
    app.run(debug=True)
