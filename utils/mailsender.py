#!/usr/bin/env python3
"""
A simple tool to send email by python

References
----------
[1] http://jingyan.baidu.com/article/b24f6c822784b886bfe5dabe.html
[2] email https://docs.python.org/3.4/library/email.mime.html
"""

import smtplib
from email.mime.text import MIMEText
import argparse

def send_mail(from_user, from_user_pw, to_user, mail_sub, mail_msg):
    """
    Send email by python

    Parameters
    ----------
    from_user: str
        Email address of the sender
    from_user_pw: str
        Password of the user
    to_users: str list
        Email address of the recievers
    mail_sub: str
        Mail subject
    mail_msg: str
        The message to be sent

    Return
    ------
    result: booling
        If email is sent successfully, return True,
        else, return False.
    """
    # Basic  parmaters
    smtp_server_postfix = from_user.split("@")[-1]
    smtp_server = "smtp." + smtp_server_postfix
    smtp_port = 25
    # Build message
    msg = MIMEText(_text=mail_msg,_subtype='html',_charset='utf-8')
    msg['Subject'] = mail_sub
    msg['From'] = from_user
    msg['To'] = ";".join(to_user)  # Multiple recievers
    # Try server
    try:
        server = smtplib.SMTP()
        server.connect(smtp_server,port=smtp_port)
        server.login(from_user.split("@")[0],from_user_pw)
        server.sendmail(from_user,to_user,msg.as_string())
        server.quit()
        result = True
    except Exception:
        result = False

    return result

def main():
    """
    The main method
    """
    # Init
    parser = argparse.ArgumentParser(description='Send email by python.')
    # parmaters
    parser.add_argument("from_user", help="Email address of the sender.")
    parser.add_argument("from_user_pw",help="Password of the sender.")
    parser.add_argument("to_user", help="Email address list of the recievers.")
    parser.add_argument("mail_sub", help="Subject of the mail.")
    parser.add_argument("mail_msg", help="Content of the mail.")
    args = parser.parse_args()

    from_user = args.from_user
    from_user_pw = args.from_user_pw
    to_user = []
    to_user.append(args.to_user)
    mail_sub = args.mail_sub
    mail_msg = args.mail_msg

    # send email
    print("Sending email from %s to %s..." % (from_user, to_user))
    result = send_mail(from_user, from_user_pw, to_user, mail_sub, mail_msg)
    if result == True:
        print("Successfully sent the mail.")
    else:
        print("Error happend.")

if __name__=="__main__":
    main()
