import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import  r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from flask import Flask, render_template, request,flash
import shutil
import os


app= Flask(__name__)
app.config['UPLOAD_FOLDER'] = r"C:\\Users\\bhuva\\Downloads\\TK43469 (1)\\TK43469\\CODE\\upload"
app.config['SECRET_KEY'] = 'b0b4fbefdc48be27a6123605f02b6b86'

df = pd.read_csv("C:\\Users\\bhuva\\Downloads\\TK43469 (1)\\TK43469\\CODE\\upload\\rainfall.csv")

df= df.fillna(df.median())
df.drop(['Jan-Feb','Mar-May', 'Jun-Sep', 'Oct-Dec'], axis=1, inplace=True)
cols= df.iloc[:,2:14].columns

le= LabelEncoder()
df['DIVISION']= le.fit_transform(df['DIVISION'])

ss = StandardScaler()
df1= pd.DataFrame()
df1['DIVISION']= df['DIVISION']
df1['YEAR']= df['YEAR']
df1['ANNUAL']= df['ANNUAL']

df.drop(['DIVISION','YEAR', 'ANNUAL'], axis=1, inplace= True)

df_month = ss.fit_transform(df)

df= pd.DataFrame(df_month, columns= cols)

final=pd.merge(df, df1,right_index=True,left_index=True, how='inner')

X= final.drop('ANNUAL', axis=1)
y= final['ANNUAL']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20)
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/load', methods=["POST", "GET"])
def load():
    if request.method == "POST":
        file = request.files['data']
        ext = os.path.splitext(file.filename)[1]
        if ext.lower() == ".csv":
            try:
                shutil.rmtree(app.config['UPLOAD_FOLDER'])
            except:
                pass
            os.mkdir(app.config['UPLOAD_FOLDER'])
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'rainfall.csv'))
            flash('The data is loaded successfully', 'success')
            return render_template('load.html')
        else:
            flash('Please upload a csv type documents only', 'warning')
            return render_template('load.html')
    return render_template('l''oad.html')
@app.route('/view', methods=['POST', 'GET'])
def view():

    X = final.drop(['ANNUAL'], axis=1)
    y = final['ANNUAL']
    if request.method == 'POST':
        filedata = request.form['df']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20)
        if filedata == '0':
            flash(r"Please select an option", 'warning')
        elif filedata == '1':
            return render_template('view.html', col=X_train.columns.values, df=list(X_train.values.tolist()))
        else:
            return render_template('view.html', col=X_test.columns.values, df=list(X_test.values.tolist()))


    return render_template('view.html')

x_train = None;
y_train = None;
x_test = None;
y_test = None;




@app.route('/training', methods=['GET', 'POST'])
def training():
    X = final.drop(['ANNUAL'], axis=1)
    y = final['ANNUAL']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20)

    if request.method == 'POST':
        model_no = int(request.form['algo'])

        if model_no == 0:
            flash(r"You have not selected any model", "info")

        elif model_no == 1:
            model = LinearRegression()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2= r2_score(pred, y_test)

            msg = "R2 SCORE FOR LINEAR REGRESSION IS :" + str(r2)

            return render_template('training.html', mag=msg)


        elif model_no == 2:
            model = Ridge()
            parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-2, 1e-1, 2, 5, 10, 20, 40, 75, 100]}
            ridge_regressor = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', cv=5)
            ridge_regressor.fit(X_train, y_train)
            pred = ridge_regressor.predict(X_test)
            r2 = r2_score(pred, y_test)

            msg = "R2 SCORE FOR RIDGE REGRESSION IS :" + str(r2)

            return render_template('training.html', mag=msg)

        elif model_no == 3:
            model = Lasso()
            parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-2, 1e-1, 2, 5, 10, 20, 40, 75, 100]}
            lasso_regressor = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', cv=5)
            lasso_regressor.fit(X_train, y_train)
            pred = lasso_regressor.predict(X_test)
            r2 = r2_score(pred, y_test)

            msg = "R2 SCORE FOR LASSO REGRESSION IS :" + str(r2)

            return render_template('training.html', mag=msg)




        elif model_no == 4:
            model = DecisionTreeRegressor()
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(pred, y_test)

            msg = "R2 SCORE FOR DECISION TREE REGRESSOR IS :" + str(r2)

            return render_template('training.html', mag=msg)




        elif model_no == 5:
            model = RandomForestRegressor(n_estimators=100, n_jobs=-1, verbose=0, random_state=42)
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            r2 = r2_score(pred, y_test)

            msg = "R2 SCORE FOR RANDOM FOREST REGRESSOR IS :" + str(r2)

            return render_template('training.html', mag=msg)



    return render_template('training.html')




@app.route('/prediction', methods=['GET', 'POST'])
def prediction():

    X = final.drop(['ANNUAL'], axis=1)
    y = final['ANNUAL']

    if request.method == "POST":
        JAN = request.form['JAN']
        print(JAN)
        FEB = request.form['FEB']
        print(FEB)
        MAR = request.form['MAR']
        print(MAR)
        APR = request.form['APR']
        print(APR)
        MAY = request.form['MAY']
        print(MAY)
        JUN = request.form['JUN']
        print(JUN)
        JUL = request.form['JUL']
        print(JUL)
        AUG = request.form['AUG']
        print(AUG)
        SEP = request.form['SEP']
        print(SEP)
        OCT = request.form['OCT']
        print(OCT)
        NOV = request.form['NOV']
        print(NOV)
        DEC = request.form['DEC']
        print(DEC)
        DIVISION = request.form['DIVISION']
        print(DIVISION)
        YEAR = request.form['YEAR']
        print(YEAR)

        di = {'JAN': [JAN], 'FEB': [FEB], 'MAR': [MAR],'APR': [APR],
              'MAY': [MAY], 'JUN': [JUN],'JUL': [JUL],'AUG': [AUG],
              'SEP': [SEP], 'OCT': [OCT],'NOV': [NOV],'DEC': [DEC], 'DIVISION': [DIVISION],'YEAR': [YEAR]}

        test = pd.DataFrame.from_dict(di)
        print(test)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.20)

        model = Lasso()
        parameters = {'alpha': [1e-15, 1e-10, 1e-8, 1e-5, 1e-2, 1e-1, 2, 5, 10, 20, 40, 75, 100]}
        lasso_regressor = GridSearchCV(model, parameters, scoring='neg_mean_absolute_error', cv=5)
        model = lasso_regressor.fit(X_train, y_train)
        pred = model.predict(test)
        print(pred)

        msg = "THE PREDICTION VALUE OF RAINFALL IS :" + str(pred)

        return render_template('prediction.html', mag=msg)
    return render_template('prediction.html')




if __name__== '__main__':
    app.run(debug=True)