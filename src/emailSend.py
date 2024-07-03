import smtplib
from email.message import EmailMessage

class Email():
    def __init__(self):
        self.emailServer = 'pstupc-win1002'
        self.emailPort = 25
        self.toEmail = 'jeoffrey.rifaz@pstechnology.com'

    def sendEmail(self, body):
        fromaddr = 'from@email.com'
        subject = 'Helpdesk'
        mailtext = body

        msg = EmailMessage()
        msg['Subject'] = subject
        msg['From'] = fromaddr
        msg['To'] = self.toEmail
        msg.set_content(mailtext)

        server = smtplib.SMTP(self.emailServer, self.emailPort)
        server.send_message(msg)
        server.quit()

        return "SUCCESS"
