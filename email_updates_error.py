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

def error(error):
    port = 587  # For starttls
    smtp_server = "smtp.gmail.com"
    sender_email = "Wilkinson.Chabannes@gmail.com"
    receiver_emails = ["Wilkinson.Chabannes@gmail.com"]
    password = 'zemhyp-hasmof-saGwe0'
    
    message = MIMEMultipart("alternative")
    message["Subject"] = "Wilkinson & Chabannes trading : Error in code"
    message["From"] = sender_email
    
    # Create the plain-text and HTML version of your message
    text = """\
    Email update,
    A fatal error occured in the code please see subsequent report
    Kind regards,
    Wilkinson & Chabannes
    
    This message is sent from Python."""
    html = """\
    <html>
      <body>
        <p><b>Email update : Error,</b><br>
            <br>
            <br>
           <b>A fatal error occured in the code please see subsequent report.</b><br>
        </p>
      </body>
    </html>
    """ + error.to_html()
    
    
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