from ml.model import TicketClassifier
import joblib
import pandas as pd

model = TicketClassifier()

data = pd.read_csv("data/support_tickets.csv")

df = data[['subject','body','language','queue','priority']].dropna().copy()

x = df[['subject','body']]

y_queue = df['queue']
y_language = df['language']
y_priority = df['priority']

model.train(x, y_queue, y_language, y_priority)

joblib.dump(model, 'support-tickets-classifier-model.pkl')