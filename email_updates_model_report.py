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
    send_report()

def send_report():
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "Wilkinson.Chabannes@gmail.com"
    receiver_emails = ["Wilkinson.Chabannes@gmail.com","chabannes.francois@gmail.com"]
    password = 'zemhyp-hasmof-saGwe0'
    
    message = MIMEMultipart("alternative")
    message["Subject"] = "Wilkinson & Chabannes trading : Evening performance update"
    message["From"] = sender_email
    
    
    def table() :
        table = pd.read_csv('./data/model_report.csv').dropna()
        table = table.set_index('Date')
        html_table = table.to_html()
        return html_table
    
    
    # Create the plain-text and HTML version of your message
    text = """\
    Email update,
    The last day's overall model quality of new models can be found in the subsequent table.
    Make sure you can receive HTML type emails if table not visible.
    Kind regards,
    Wilkinson & Chabannes
    
    This message is sent from Python."""
    html = """\
    <html>
      <body>
        <p><b>Model quality update,</b><br>
        <br>
           The last day's overall model quality of new models can be found in the subsequent table.<br>
           <br>
           <b>- Total_models :</b> Corresponds to the total number of new models evaluated <br>
           <b>- Strong_models :</b> Corresponds to the number of models with a traded accuracy above 0.58  <br>
           <b>- Accuracy_over_100_days :</b> Corresponds to the average accuracy  over 100 days <br>
           <b>- Estimated performance_100_days :</b> Corresponds to the estimated average market performance over 100 days<br>
           <b>- Weekly_accuracy :</b> Correpsonds to the average weekly accuracy estimated for 150 days (trained/tested week by week) <br>
           <b>- Weekly_ROC :</b> Corresponds to the average weekly ROC_AUC estimated for 150 days (trained/tested week by week) <br>
           <b>- Weekly_traded_accuracy :</b> Corresponds to the weekly traded accuracy where trades are only considered if they have a probability above 0.55 <br>
           <br>
           Kind regards, <br>
           <b> Wilkinson & Chabannes Trading © </b> <br>
           <br>
           <i>Sent from python in HTML </i> <br>
        </p>
      </body>
    </html>
    """ + table()
    
    
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