from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from efficientnet.tfkeras import EfficientNetB0
import numpy as np
app = Flask(__name__)

dic = {0 : 'fake', 1 : 'real'}
deep = {0 : 'deepfake', 1 : 'real'}

meso = load_model('combined_df_model.h5')
model = load_model('static.h5')
meso.make_predict_function()

# def predict_label(img_path):
# 	i = image.load_img(img_path, target_size=(256,256))
# 	i = image.img_to_array(i)/255.0
# 	i = i.reshape(1, 256,256,3)
# 	preds = meso.predict(i)
# 	x = (preds > 0.5).astype("int32")
# 	y = float(preds)
# 	return dic[int(x)], y

def predict_deepfake(img_path):
	i = load_img(img_path, target_size=(256,256))
	i = img_to_array(i)/255.0
	i = i.reshape(1,256,256,3)
	preds = model.predict(i)
	x = (preds > 0.5).astype("int32")
	y = float(preds)
	return deep[int(x)], y

# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("main.html")

@app.route("/models", methods=['GET', 'POST'])
def models():
	return render_template("models.html")

@app.route("/image", methods=['GET', 'POST'])
def image():
	return render_template("image.html")

@app.route("/drag", methods=['GET', 'POST'])
def drag():
	return render_template("drag.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/images/" + img.filename	
		img.save(img_path)

		# p, confidence = predict_label(img_path)
		# if (confidence < 0.5):
		# 	confidence = 1 - confidence
		# percentage = "{:.02%}".format(confidence)

		d, confidences = predict_deepfake(img_path)
		if (confidences < 0.5):
			confidences = 1 - confidences
		confidences = "{:.02%}".format(confidences)
	return render_template("main.html",  img_path = img_path, deepfake = d, confidences = confidences)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)