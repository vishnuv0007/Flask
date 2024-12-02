from flask import Flask,render_template,request
import pickle
import numpy as np


app = Flask(__name__)

#iris = load_iris()
with open('model.pkl','rb') as model_file:
    rf_clf_model = pickle.load(model_file)

@app.route('/')
def home():
    return render_template('first.html')

@app.route('/predict',methods=['POST'])
def predict():
    Item_Purchased = float(request.form['ItemPurchased'])
    Tot_Spent = float(request.form['TotSpent'])
    Discount = float(request.form['discount'])
    War_Ext = float(request.form['WarExt'])
    Rev = float(request.form['rev'])
    Store_Rat = float(request.form['StoreRat'])
    Loy_score = float(request.form['Loyscore'])
    Mem_status = float(request.form['Memstatus'])
    Ratio_spent = float(request.form['Ratiospent'])
    Disc_Spend = float(request.form['DiscSpend'])

    input_features = np.array([[Item_Purchased,Tot_Spent,Discount,War_Ext,Rev,Store_Rat,Loy_score,Mem_status,Ratio_spent,Disc_Spend]])
    pred = rf_clf_model.predict(input_features)

    target = pred[0]
    #species = iris.target_names[target]
    return render_template('first.html',pred_result = target)

if __name__=='__main__':
    app.run(debug=True)