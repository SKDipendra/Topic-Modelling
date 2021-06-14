import pandas as pd

df = pd.read_csv(r'para_frame_topic.csv', names=['text','frame','topic'])
#df = df[1:]
print(df)
frame = df.frame.values
topic = df.topic.values

df1 = pd.DataFrame()

print(df.corr(method='spearman'))