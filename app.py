from flask import Flask,render_template, request
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

def ValuePredictor(news): 
	m=pickle.load(open('model.pkl','rb'))
	vectorizer = TfidfVectorizer(stop_words='english',ngram_range=(1,2),vocabulary = pickle.load(open('transform.pkl','rb')))
	news = vectorizer.fit_transform(news)
	p=m.predict(news)
	return p

app=Flask(__name__) 

@app.route('/resultt',methods=['GET'])
def index():
	return render_template('a.html')
@app.route('/resultt', methods = ['GET','POST']) 
def result(): 
	if request.method == 'POST': 
		to_predict_list = request.form.to_dict() 
		news = list(to_predict_list.values()) 
		result = ValuePredictor(news)         
		if result ==0:
			pred='business'
		if result ==1:
			pred='tech'
		if result ==2:
			pred='politics'            
	return render_template('resultt.html', prediction = pred)





if __name__ == '__main__':
	app.run(debug=True)

