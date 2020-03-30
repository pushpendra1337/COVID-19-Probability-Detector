'''
File: app.py
Description: Main app engine

'''
__author__ = "Pushpendra Yadav"
__credits__ = ["Pushpendra Yadav"]
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Pushpendra Yadav"
__email__ = "pushpendray1337@gmail.com"


from flask import Flask, render_template, request
app = Flask(__name__)
import pickle


file = open('model.pkl', 'rb')
clf = pickle.load(file)
file.close()

@app.route('/', methods=["GET", "POST"])


def main_app():
	if request.method == "POST":
		myDict = request.form
		age = int(myDict['age'])
		bodyTemp = int(myDict['bodyTemp'])
		dryCough = int(myDict['dryCough'])
		sneezing = int(myDict['sneezing'])
		soreThroat = int(myDict['soreThroat'])
		weakness = int(myDict['weakness'])
		severeCough = int(myDict['severeCough'])
		diffBreath = int(myDict['diffBreath'])
		pain = int(myDict['pain'])
		travelHist = int(myDict['travelHist'])
		contactWith = int(myDict['contactWith'])
		runnyNose = int(myDict['runnyNose'])
		diabetes = int(myDict['diabetes'])
		highBP = int(myDict['highBP'])
		heartD = int(myDict['heartD'])
		kidneyD = int(myDict['kidneyD'])
		lungD = int(myDict['lungD'])
		lessImmune = int(myDict['lessImmune'])

		#Inference
		inputFeatures = [age, bodyTemp, dryCough, sneezing, soreThroat, weakness, severeCough, diffBreath, pain, travelHist, contactWith, runnyNose, diabetes, highBP, heartD, kidneyD, lungD, lessImmune]
		infProb = clf.predict_proba([inputFeatures])[0][1]
		print(infProb)

		return render_template('show.html', inf=round(infProb*100))
	return render_template('index.html')

		#return 'Hello, World!' + str(infProb)


if __name__=="__main__":
	app.run(debug=True)