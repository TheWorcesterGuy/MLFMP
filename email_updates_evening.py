#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 16 04:44:47 2021

@author: christian
"""

import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import pandas as pd
import numpy as np

def main():
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "Wilkinson.Chabannes@gmail.com"
    receiver_emails = ["Wilkinson.Chabannes@gmail.com","chabannes.francois@gmail.com"]
    password = 'zemhyp-hasmof-saGwe0'
    
    message = MIMEMultipart("alternative")
    message["Subject"] = "Wilkinson & Chabannes trading : Evening performance update"
    message["From"] = sender_email
    
    
    def table_trades() :
        df = pd.read_csv('./data/record_traded.csv').dropna()
        df = df[df['Date']==df['Date'].iloc[-1]]
        df = df.drop(['Date', 'Prob_distance'], axis = 1)
        delta = df.groupby(['Traded']).mean()['Delta']
        proba = df.groupby(['Traded']).mean()['Probability']
        df = df.groupby(['Traded']).sum()
        df['Delta'] = delta * 100
        df['Probability'] = proba * 100
        df = df.round(2)
        
        html_table = df.to_html()
        return html_table
    
    def table_account() :
        df = pd.read_csv('./data/account.csv')
        df = df[df['Date']==df['Date'].iloc[-1]]
        table = df.set_index('Date')
        html_table = table.to_html()
        return html_table

#np.sum(np.array(df['Prediction'].tolist()) * np.array(df['Delta'].tolist()))/np.sum(np.abs(np.array(df['Prediction'].tolist())))
    
    # Create the plain-text and HTML version of your message
    text = """\
    Email update,
    The trades executed today can be found in the subsequent table.
    Make sure you can receive HTML type emails if table not visible.
    Kind regards,
    Wilkinson & Chabannes
    
    This message is sent from Python."""
    html = """\
    <html>
      <body>
        <p><b>Trading performance update,</b><br>
        <br>
           The trades executed today can be found in the first subsequent table.<br>
           <b>- Traded :</b> Corresponds to the products traded today <br>
           <b>- Prediction :</b> Corresponds to the sum of all traded predictions on a given stock <br>
           <b>- Outcome :</b> Corresponds to the true outcome and gives the total number of predictions <br>
           <b>- Delta (in %) :</b> The percentage move of the traded product between market open and close <br>
           <b>- Probability (in %) :</b> Corresponds to The average probability of all predictions <br>
           <br>
           The trading account information can be found in the second subsequent table. <br>
           <b>- AM :</b> Total account value before market open <br>
           <b>- PM :</b> Total account value after market close <br>
           <b>- Trade_value :</b> Total value of products traded today <br>
           <b>- Change_account_% :</b> Percentage change of total account value <br>
           <br>
           Kind regards, <br>
           <b> Wilkinson & Chabannes Trading © </b> <br>
           <br>
           <i>Sent from python in HTML </i> <br>
        </p>
      </body>
    </html>
    """ + table_trades() + table_account()
    
    
    # Turn these into plain/html MIMEText objects
    part1 = MIMEText(text, "plain")
    part2 = MIMEText(html, "html")
    
    # Add HTML/plain-text parts to MIMEMultipart message
    # The email client will try to render the last part first
    message.attach(part1)
    message.attach(part2)

    # Create secure connection with server and send email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(sender_email, password)
        for receiver_email in receiver_emails :
            server.sendmail(
                sender_email, receiver_email, message.as_string()
            )
            
if __name__ == "__main__":
    main()