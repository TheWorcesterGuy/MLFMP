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
    message["Subject"] = "Wilkinson & Chabannes trading : Todays trades"
    message["From"] = sender_email
    
    def table() :
        df = pd.read_csv('./data/days_trades.csv').dropna()
        df = df.rename(columns={"product": "To_trade", "side": "Direction"})
        df = df.set_index('To_trade')
        html_table = df.to_html()
        return html_table
    
    # Create the plain-text and HTML version of your message
    text = """\
    Email update,
    The trades to be executed today can be found in the subsequent table.
    Make sure you can receive HTML type emails if table not visible.
    Kind regards,
    Wilkinson & Chabannes
    
    This message is sent from Python."""
    html = """\
    <html>
      <body>
        <p>Trading performance update,<br>
        <br>
           The trades to be executed today can be found in the subsequent table.<br>
           <br>
           <b>- To_trade :</b> Corresponds to the products that will be traded today <br>
           <b>- Direction :</b> Corresponds to the direction of trade, positive for long, negative for short <br>
               and the effective number of prediction for that direction (sum of all predictions long and short) <br>
           <b>- Value_to_trade_$ :</b> Approximate dollar value of trade <br>
           <b>- Fractionnable :</b> Whether the trade is a fraction of product <br>
           <b>- Last_Stock_Value_$ :</b> The value of the product at previous close <br>
           <b>- Quantity :</b> Quantity of shares to be traded <br>
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