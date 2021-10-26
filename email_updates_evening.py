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
        
        table = pd.DataFrame()
        table['Traded'] = df['Traded'].drop_duplicates().tolist()
        table['Direction_and_total_prediction'] = (df.groupby(['Traded']).sum())['predictions'].tolist()
        table['Total_number_of_predictions_and_outcome'] = (df.groupby(['Traded']).sum())['outcome'].tolist()
        table['Average_probability'] = np.round((df.groupby(['Traded']).mean())['Probability'].tolist(),2)
        table['Market_variation_%'] = np.round(np.array((df.groupby(['Traded']).mean())['Delta'].tolist()) * 100,2)
    
        table = table.set_index('Traded')
        html_table = table.to_html()
        return html_table
    
    def table_account() :
        df = pd.read_csv('./data/account.csv').dropna()
        df = df[df['Date']==df['Date'].iloc[-1]]
        table = df.set_index('Date')
        html_table = table.to_html()
        return html_table
    
    
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
           <b>- Direction_and_total_prediction :</b> Corresponds to the direction of trade, positive for long, negative for short <br>
               and the effective number of prediction for that direction (sum of all predictions long and short) <br>
           <b>- Total_number_of_predictions_and_outcome :</b> Corresponds to the true outcome and gives the total number of predictions <br>
           <b>- Average_probability :</b> The average probability of all predictions <br>
           <b>- Market_variation_% :</b> The percentage move of the traded product between market open and close <br>
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