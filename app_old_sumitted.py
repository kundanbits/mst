import dash

from dash import dcc
from dash import html
from dash import Dash, html, dcc
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output

import pandas as pd
import numpy as np
from sklearn.svm import LinearSVC
import matplotlib.pyplot as plt

dfn=pd.read_excel('data_ML.xlsx')

X = dfn[['displacement', 'roughness']]
y = dfn['class']

clf = LinearSVC()
clf.fit(X, y)


w = clf.coef_[0]
b = clf.intercept_

plt.scatter(X['displacement'], X['roughness'], c=y)
plt.xlabel('Displacement (\u03BCm)')
plt.ylabel('Roughness (\u03BCm)')

xmin, xmax = plt.xlim()
ymin, ymax = plt.ylim()
xx = np.linspace(xmin, xmax)
yy = np.linspace(ymin, ymax)
XX, YY = np.meshgrid(xx, yy)
Z = clf.decision_function(np.c_[XX.ravel(), YY.ravel()])
Z = Z.reshape(XX.shape)

predin=clf.predict([[2, 2],[30,2]])
print(predin)

dis=([30],[2])

data=pd.read_excel('ax_1.xlsx',header=None,names=["time","acc"])

dfc=pd.read_excel('displacement_dash.xlsx')

df = pd.DataFrame({
    "Time (Sec)": dfc["time"],
    "Displacement (\u03BCm)": dfc["disp"],
    })
df1 = pd.DataFrame({
    "Displacement (\u03BCm)": ([2,30]),
    "Class": ([1,0]),
    })

fig = px.line(df, x="Time (Sec)", y="Displacement (\u03BCm)",width=500,)
fig1=px.scatter(df1, x="Displacement (\u03BCm)", y="Class",size_max=15,width=500,)
fig1.update_traces(marker=dict(size=12,
                              line=dict(width=2,
                                        color='DarkSlateGrey')),
                  selector=dict(mode='markers'))
#fig1=px.bar(df1, x="Displacement", y="Class").update_traces(width = 0.7)
#fig1.update_layout(yaxis_range=[-1,1])
fig.update_yaxes(title_font=dict(size=20))
fig1.update_yaxes(title_font=dict(size=20))
fig.update_xaxes(title_font=dict(size=20))
fig1.update_xaxes(title_font=dict(size=20))

app = dash.Dash(__name__)

fig_names = ['Acceleration Spectrum', 'Displacement Spectrum']

app.layout = html.Div(children=[
    html.H1(children='Stability Analysis During Thin-wall Machining', style={'text-align': 'center'}),
    html.Div(children='Analyze the stability of machining process during micromachining of Thin-walled Ti6Al4V'
                      , style={'text-align': 'center'}),

    html.Div([
        html.Label(['Choose a Figure:'], style={'font-weight': 'bold'}),
        dcc.Dropdown(
            id='dropdown',
            options=[
                {'label': 'Displacement Spectrum', 'value': 'graph1'},
                {'label':'Class Prediction','value' : 'graph2'},
            ],
            value='graph1',
            style={"width": "60%"}),

        html.Div(dcc.Graph(id='graph')),
    ]),

])

@app.callback(
    Output('graph', 'figure'),
    [Input(component_id='dropdown', component_property='value')]
)
def select_graph(value):
    if value == 'graph1':
        return fig
    else:
        return fig1

if __name__ == '__main__':
    app.run_server(debug=True,port=8098)
