import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame()

data = pd.read_csv('./loss_without_dropout.csv')
df['Training error w/o droput'] = data['Value']

data = pd.read_csv('./loss_with_dropout.csv')
df['Training error w/ droput'] = data['Value']

data = pd.read_csv('./valid_loss_without_dropout.csv')
df['Validation error w/o dropout'] = data['Value']

data = pd.read_csv('./valid_loss_with_dropout.csv')
df['Validation error w/ dropout'] = data['Value']

print(df)

ax = df.plot()
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE')
plt.savefig('mse.png')
plt.show()
