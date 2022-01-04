import pandas as pd
from datetime import datetime
from datetime import timedelta
import matplotlib.pyplot as plt

df = pd.read_csv('entries.csv')
df['date_dt'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
date_generated = [(df['date_dt'].min() + timedelta(days=x)).strftime('%Y-%m-%d') for x in range(0, (df['date_dt'].max()-df['date_dt'].min()).days)]
df_filled = pd.DataFrame(date_generated,columns=['date'])
df = df_filled.merge(df,on='date',how='left')[['date', 'entry']]
df['date'] = df['date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d"))
df['has_entry'] = df['entry'].isna()==False
df.loc[df['has_entry']==True]
fig, ax = plt.subplots(figsize=(10, 6))
# Specify how our lines should look
ax.bar(df.date, df.has_entry)
plt.show()