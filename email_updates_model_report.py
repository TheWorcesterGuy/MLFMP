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
from features_report import *

def main():
    features_report()
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
    
    
    def table_model_report() :
        table = pd.read_csv('./data/model_report.csv').dropna()
        table = table.set_index('date')
        html_table = table.to_html()
        return html_table
    
        
    def table_top() :
        table = pd.read_csv('./data/Top.csv')
        table = table.set_index('Features')
        html_table = table.to_html()
        return html_table
    
    def table_bottom() :
        table = pd.read_csv('./data/Bottom.csv')
        table = table.set_index('Features')
        html_table = table.to_html()
        return html_table
    
    def table_features_not_used() :
        table = pd.read_csv('./data/features_not_used.csv')
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
           The last day's overall quality of new models can be found in the subsequent table as well as the feature usage. <br>
           <br>
           <b>- Test :</b> Corresponds to the the average metrics aquired during the model testing <br>
           <b>- Live :</b> Corresponds to the average metrics aquired during the simulated live trading  <br>
           <b>- Second table :</b> Gives the frequency (in %) of appearance of features in the top 10 used features  <br>
           <b>- Third table :</b> Gives the frequency (in %) of appearance of features in the bottom 10 used features  <br>
           <b>- Fouth table :</b> Gives the features unused  <br>
           <br>
           Kind regards, <br>
           <b> Wilkinson & Chabannes Trading © </b> <br>
           <br>
           <i>Sent from python in HTML </i> <br>
        </p>
      </body>
    </html>
    """  + table_model_report() + table_top() + table_bottom() + table_features_not_used()
    
    
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