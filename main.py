import pandas as pd
import numpy as np
import multiprocessing
import random as rd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.linear_model import LinearRegression
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from sympy import *
import os

class RealEstate():
    def __init__(self):
        PATH = "C:/Users/Cem Bartug CEVIK/Documents/kingcountysales.csv"
        self.df = pd.read_csv(PATH)
        self.df = pd.DataFrame(self.df)
        self.df = self.df.drop(["sale_id","pinx"], axis=1)

    def emailpre(self):
        emergingprice1 = int(input("Significant undervalue indicator for emails:  "))
        self.mailtext = open("mail.txt","a")
        self.mailtext.write(str(emergingprice1))

    def smspre(self):
        emergingprice2 = int(input("Significant undervalue indicator for emails:  "))
        self.smstxt = open("sms.txt","a")
        self.smstxt.write(str(emergingprice2))

    def preselection(self):
        # Necessary for the AI process.
        predtypes = self.df.dtypes
        precolumns = self.df.columns
        self.profiltercols = []
        for i in range(len(self.df.columns)):
            if predtypes[i] == object and self.df[precolumns[i]].nunique() == 2:
                self.df[precolumns[i]] = pd.get_dummies(self.df[precolumns[i]])
            if predtypes[i] == object and self.df[precolumns[i]].nunique() > 2:
                self.profiltercols.append(self.df[precolumns[i]])
                self.df = self.df.drop(precolumns[i], axis=1)
            elif self.df[precolumns[i]].isnull().sum() / len(self.df[precolumns[i]]) > 1 / 10:
                self.df = self.df.drop(precolumns[i], axis=1)
            else:
                self.df = self.df[precolumns[i]].dropna()

    def locations(self):
        # Suggested for the AI process.
        self.profiltercols = pd.Series(self.profiltercols)
        print("City options:")
        print(pd.unique(self.df["city"]))
        cityname = input("Your city's name")
        self.df = self.df[self.df["city"] == cityname]
        subdvlist = []
        print("Enter space to add")
        for i in pd.unique(self.df["subdivision"]):
            subdvlist.append(input(i))
        self.df = self.df[self.df["subdivision"] in subdvlist]
        self.profiltercols = self.profiltercols[self.profiltercols["city","subdivision"]]
        dispensablityno = int(input("After what number of categories filter type is outnumbered?  "))
        for i in self.profiltercols:
            if self.df[i].nunique() < dispensablityno:
                thecolsffs = []
                print(pd.unique(self.df[i]))
                ffilterask = input("Type the choosen filters list no's, separate them with comas.")
                for i in ffilterask.split(","): thecolsffs.append(int(i))
                self.df = self.df[self.df[i] not in thecolsffs]

    def filtering(self):
        # Optional
        for i in self.df.columns:
            forr = input("filter or range")
            if forr == "filter":
                felements = []
                for u in list(self.df[i].unique()):
                    finclude = "include " + input(u)
                    if finclude == " ": felements.append(u)
                self.df = self.df[self.df[i] in felements]
            if forr == "range":
                print("Minimum filter unit = " + str(min(self.df[i])) + ",\nMaximum filter unit = " + str(max(self.df[i])))
                minfilter = int(input("Bottom filter:  "))
                maxfilter = int(input("Top filter:  "))
                self.df = self.df[maxfilter >= self.df[i] >= minfilter]

    def y_grouping(self):
        # Necessary for the AI process.
        grunitprice = int(input("select the power of 10 to group the prices for an advantageous output."))
        pricegmin = round(min(self.df["sale_price"]), -1 * grunitprice)
        pricegmax = round(max(self.df["sale_price"]), -1 * grunitprice)
        grunitprice = 10 ** grunitprice
        if pricegmin > min(self.df["sale_price"]): pricegmin -= grunitprice
        if pricegmax < max(self.df["sale_price"]): pricegmax += grunitprice
        pricegrange = pricegmax - pricegmin
        pricegrangel = []
        pricegunit = pricegmin / grunitprice
        pricegrangel.append(pricegunit)
        for i in range(int(pricegrange / grunitprice)):
            pricegunit += 1
            pricegrangel.append(pricegunit)

    def feature_selection(self):
        # Optional
        print("Type d to not consider the factors about this ")
        drops = []
        for i in self.df.columns.drop("sale_price"):
            if input(i) == "d": drops.append(i)
        self.df = self.df.drop(drops, axis=1)

    def ai(self):
        # The AI
        self.df = self.df.sample(frac=0.1)
        self.X = self.df.drop("sale_price", axis=1).values
        y = self.df["sale_price"].values
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, y, test_size=0.25, random_state=101)
        self.scaler = MinMaxScaler()
        self.scaler.fit(self.X_train)
        self.X_train = self.scaler.transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        es = EarlyStopping(monitor="val_loss", mode="min", verbose=1, patience=15)
        self.model = Sequential()
        for i in [8, 4, 2]:
            self.model.add(Dense(i * 3, activation="relu"))
            self.model.add(Dropout(0, 2))
        self.model.compile(optimizer="adam", loss='categorical_crossentropy')
        self.model.fit(x=self.X_train, y=self.y_train, epochs=400, batch_size=50, validation_data=(self.X_test, self.y_test), callbacks=[es])

    def ai_output(self):
        # AI related.
        print("Train error:  ")
        self.model.evaluate(self.X_train, self.y_test, verbose = 0)
        self.model.evaluate(self.X_test, self.y_test, verbose = 0)
        losses = pd.DataFrame(self.model.history.history)
        losses.plot()
        plt.show()
        self.predictm = self.model.predict(self.X_test)
        self.predictm = pd.Series(self.predictm.reshape(300,))
        pred_df = pd.DataFrame(self.y_test, columns=["true","prediction"])
        pred_df = pd.concat([pred_df, self.predictm], axis=1)

    def graphical(self):
        # AI related.
        y1 = 0
        y2 = 0
        options = {0: self.predictm, 1: self.y_test}
        self.options2 = {0: y1, 1: y2}
        self.coloropts = ["red", "blue"]
        rsquareds = []
        self.xs = Symbol("x")
        degreelimit = int(input("Number of degrees that would be considered until:  "))
        for t in range(2):
            for i in range(degreelimit, 1, 1):
                poly = PolynomialFeatures()
                polyX = poly.fit_transform(self.X_test.reshape(-1, 1))
                lmodel = LinearRegression
                lmodel.fit(polyX, options[t].reshape(-1, 1))
                rsquared = lmodel.score(polyX, options[t].reshape(-1, 1))
                rsquareds.append(rsquared)
            bestdeg = int(np.argmax(rsquareds))
            polyfeaturesm = np.polyfit(self.X_test, options[t], bestdeg)
            for p in range(len(polyfeaturesm)):
                self.options2[t] += polyfeaturesm[p] * self.xs ** (len(polyfeaturesm) - p)
            plt.plot(self.xs, self.options2[t], color = self.coloropts[t])
        plt.show()

    def sorting(self):
        # Post-AI, optional.
        self.df = pd.DataFrame(self.df)
        def fff(ff):
            print(pd.unique(self.df[priorcolumn]))
            choosenfilters = input("Which options do you choose, separate them with coma")
            choosenfilters = list(choosenfilters.split(","))
            def app1f(tvar):
                if tvar == choosenfilters in self.df[priorcolumn]: return 0
                else: return 1
            self.df["filter"+ff] = pd.Series(self.df[priorcolumn]).apply(app1f)
        print("Maximum number of different priorities is 3")
        prioritydict = {}
        filterslist = []
        ascendinglist = []
        for i in range(int(input("What is the number of different priorities?"))):
            priorcolumn = input("Which column will you consider during prioritization")
            filterslist.append(priorcolumn)
            if pd.Series(self.df).dtype != object:
                prioritydict[i] = []
                print("Type the priority type you prefer")
                priorityype = input("binary, distance, ascending, descending")
                print(self.df.columns)
                if priorityype == "ascending": self.df["filter"+str(i)] = self.df[sorted(self.df[priorcolumn])]
                if priorityype == "descending":
                    self.df["filter" + str(i)] = self.df[sorted(self.df[priorcolumn])]
                    ascendinglist.append(False)
                if priorityype == "distance":
                    culmination = int(input("What is the point identifies priority culmination"))
                    self.df["filter"+str(i)] = abs(self.df[priorcolumn] - culmination)
                    self.df = self.df[reversed(sorted(self.df["filter"+str(i)]))]
                if priorityype == "binary":
                    forr = input("Filter(f) or Range(r)")
                    if forr == "f": fff(i)
                    if forr == "r":
                        print("Min = "+str(min(self.df[priorcolumn]))+",\nMax = "+str(max(self.df[priorcolumn])))
                        minfilter = int(input("Bottom filter:  "))
                        maxfilter = int(input("Top filter:  "))
                        def app2f(tvar):
                            if maxfilter > tvar > minfilter: return 0
                            else: return 1
                        self.df["filter"+str(i)] = pd.Series(self.df[priorcolumn]).apply(app2f)
            else:
                print("Only binary type's filter feature is present")
                fff(i)
            if ascendinglist[i] != False: ascendinglist.append(True)
        self.df = self.df.sort_values(filterslist, axis=0, ascending=ascendinglist, inplace=True)

    def newinput(self):
        # New input, AI related
        self.cdatasX = []
        mx = pd.DataFrame(self.X)
        autorman = input("Automatic (a), Manuel(m)")
        for i in mx.columns:
            if autorman == "m": self.cdatasX.append(int(input(str(i) + " value:  ")))
            if autorman == "a": self.cdatasX.append(rd.randint(mx[i].min(),mx[i].max()))
        self.yvalue = int(input("price:  "))
        self.newX = np.array(self.cdatasX)
        self.newX = self.scaler.transform(self.newX)
        self.thepred = self.model.predict(self.newX)

    def nioutput(self):
        output = {
            "difference (predicted-true)": [self.yvalue, self.thepred, self.thepred-self.yvalue],
            "percentage (predicted-true)": [self.yvalue, self.thepred, self.thepred/self.yvalue * 100],
        }
        output = pd.DataFrame(output, columns=["unit1","unit2","result"])

    def aireport(self):
        from sklearn.metrics import confusion_matrix, classification_report
        print("Error types: 1.abs_error, 2.sq_error, 3.sq_log_error, 4.med_abs_error ")
        errortype = int(input("What is the error you will consider"))

    def Infotransmition(self):
        if self.yvalue / self.thepred <= int(self.mailtext.readline(0)) / 100:
            self.attachment = open("report.txt", "w+")
            for i in range(len(self.cdatasX)):
                self.attachment.write(str(pd.DataFrame(self.X).columns[i]) + ":  " + str(self.cdatasX[i]) + "\n")
            self.attachment.write("Real Price:  " + str(self.yvalue))
            self.attachment.write("Optimum Price:  " + str(self.thepred))
            self.attachment.write("Price Difference:  " + str(self.thepred - self.yvalue))
            self.attachment.write("Price Percentage Difference:  " + str(1 - (self.yvalue / self.thepred) * 100))

    def emailsender(self):
        senderm = "ccembartu@gmail.com"
        messagesend = "undervalued real estate detected (" + str((1 - self.yvalue / self.thepred) * 100) + "% lower)."
        msg = MIMEMultipart()
        msg['Subject'] = "Real Estate Detection"
        msg.attach(MIMEText(messagesend, 'plain'))
        p = MIMEBase('application', 'octet-stream')
        p.set_payload((self.attachment).read())
        encoders.encode_base64(p)
        p.add_header('Content-Disposition', "attachment; filename= %s" % "report")
        msg.attach(p)
        msession = smtplib.SMTP("smtp.gmail.com", 587)
        msession.starttls()
        msession.login(senderm, "7901aaaa")
        msession.sendmail(senderm, "cem.cevik@bilgeadam.com", msg.as_string())
        msession.quit()

    def smssender(self):
        from twilio.rest import Client
        account_sid = "AC434e667e471ef5e334a6779ed0f736e6"
        auth_token = "8eb22d135c453a9b6d5c54c019c8c42d"
        twilio_pn = "+13188181684"
        recipient_pn = "5322869988"
        def send_sms(message):
            try:
                client = Client(account_sid, auth_token)
                message = client.messages.create(body=message,from_=twilio_pn,to=recipient_pn)
                print("SMS sent successfully. SID:", message.sid)
            except Exception as e: print("Error sending SMS:", str(e))
        send_sms(str(self.smstxt.read()))

    def statsetdefault(self):
        self.sttf = open("statfile.txt","a+")
        statsettlist = ["+",5]
        self.sttf.writelines(statsettlist)

    def statsetchange(self):
        self.statsetl = {0: "no of units",1: "first or lasts"}
        print(self.statsetl)
        with open('statfile.txt', 'r', encoding='utf-8') as file: sdata = file.readlines()
        schange = int(input("Which number to change?"))
        sdata[schange] = input(self.statsetl[schange])
        with open('statfile.txt', 'w', encoding='utf-8') as file:
            file.writelines(sdata)
    def basics(self):
        print("Type the no of the ones you want to add, separate them with comas without spaces.")
        print("0: headdata, 1: datatypes, 2: uniqueno, 3: uniques, 4: average, 5:derivation, 6: min, 7: max, 8:difference between min")
        options = input(": ")
        options = options.split(",")
        self.sttf = open("statfile.txt","r")
        for i in self.df.columns:
            if "0" in options:
                if self.sttf.readline(1) == "firsts": print(self.df[i][:self.sttf.readline(0)])
                if self.sttf.readline(1) == "lasts": print(self.df[i][self.sttf.readline(0):])
            if "1" in options: print(self.df[i].dtype)
            if "2" in options: print(self.df[i].nunique())
            if "3" in options: print(self.df[i].unique())
            if "4" in options: print(self.df[i].mean())
            if "5" in options: print(self.df[i].std())
            if "6" in options: print(self.df[i].min())
            if "7" in options: print(self.df[i].max())
            if "8" in options: print(self.df[i][self.df[i] - self.df[i].min()])

    def savelist(self):
        nofnewslist = int(input("Number of new savelists:  "))
        options = ["ascending", "descending","specific","ideal point","worstpoint"]
        filenames = {}
        for i in range(nofnewslist):
            newdf = self.df
            filenames[i] = "file"+str(i)+".txt"
            thelist = open("file"+str(i)+".txt","w+")
            change = input("Filename: file"+str(i)+", type c to change the name")
            if change == "c":
                newnamef = input("New file name:  ")
                filenames[i] = newnamef+".txt"
                os.rename("file"+str(i)+".txt",newnamef+".txt")
            newcolumn = input("Add a new column feature (yes/no)")
            while newcolumn == "yes":
                print(newdf.columns)
                columnname = input("Which column do you want to include")
                if newdf[columnname].dtype != object:
                    for a in range(3): print(str(a) + "- " + options[a])
                else: print("0- "+options[2])
                optionsc = int(input("Which number do you prefer"))
                if optionsc == 0: newdf = newdf[newdf[columnname] > int(input("Minimum interval?  "))]
                if optionsc == 1:  newdf = newdf[newdf[columnname] < int(input("Maximum interval?  "))]
                if optionsc == 2:
                    print(newdf[columnname].unique())
                    choosens = input("Type unique values to add, separate with comas without spaces").split(",")
                    newdf = newdf[newdf[columnname] in choosens]
                if optionsc == 3 or optionsc == 4:
                    sigpoint = int(input("What is the basis point"))
                    distance = int(input("What is the distance"))
                    if optionsc == 3: newdf = newdf[abs(newdf[columnname] - sigpoint) < distance]
                    if optionsc == 4: newdf = newdf[abs(newdf[columnname] - sigpoint) > distance]
            else: thelist.writelines(newdf)







obj2 = RealEstate()
obj2.basics()








































