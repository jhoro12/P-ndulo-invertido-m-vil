import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import os
import base64

# Cargar imagen y codificar en base64
image_path = 'pendulo.png'
encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()

# Inicializar app con Bootstrap
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.SLATE])
app.title = "Péndulo invertido móvil"

g = 9.81

app.layout = dbc.Container([
    dbc.Row([
        dbc.Col(html.H1("Péndulo invertido móvil", className="text-center text-primary mb-4"))
    ]),

    dbc.Row([
        dbc.Col(html.Img(
            src='data:image/png;base64,{}'.format(encoded_image),
            style={
                'width': '40%',
                'margin': '0 auto',
                'display': 'block',
                'borderRadius': '12px',
                'boxShadow': '0 4px 8px rgba(0,0,0,0.2)',
                'border': '1px solid #ccc',
                'backgroundColor': '#fff',
                'padding': '10px'
            }), width=12)
    ]),

    dbc.Row([
        dbc.Col(html.P("Esquema del sistema físico", className="text-center fst-italic text-muted mb-4"))
    ]),

    dbc.Row([
        dbc.Col([
            html.Label("Longitud (l) [m]:"),
            dcc.Slider(id='l-slider', min=0.1, max=8.0, step=0.1, value=1.0,
                       marks={i: str(i) for i in range(1, 9)})
        ], md=6),

        dbc.Col([
            html.Label("Parámetro (a = M/m):"),
            dcc.Slider(id='a-slider', min=0.1, max=10.0, step=0.1, value=1.0,
                       marks={i: str(i) for i in range(1, 11)})
        ], md=6)
    ], className="mb-4"),

    dbc.Row([
        dbc.Col(html.Div(id='error-output', className="text-center mb-4"))
    ]),

    dbc.Row([
        dbc.Col(dcc.Graph(id='multi-plot'), width=12)
    ])
], fluid=True)


@app.callback(
    [Output('multi-plot', 'figure'),
     Output('error-output', 'children')],
    [Input('l-slider', 'value'),
     Input('a-slider', 'value')]
)
def update_graph(l, a):
    h = 0.01
    theta_full = np.linspace(0, 2*np.pi, 1000)
    theta_time = np.arange(0.1, 2*np.pi - 0.00001, h)
    if len(theta_time) % 2 == 0:
        theta_time = theta_time[:-1]

    omega_time = np.sqrt((2*g*(1-np.cos(theta_time))) / (l*(1 - (2*np.cos(theta_time)**2)/(2+a))))
    integrand = 1 / omega_time
    time_intervals = []

    for n in range(0, len(theta_time)-2, 2):
        fx0 = integrand[n]
        fx1 = integrand[n+1]
        fx2 = integrand[n+2]
        integral = (h/3) * (fx0 + 4*fx1 + fx2)
        time_intervals.append(integral)

    cumulative_time = np.cumsum(np.insert(time_intervals, 0, 0))
    theta_sampled = theta_time[::2]

    def f_inversa(theta):
        return 1 / np.sqrt((2 * g * (1 - np.cos(theta))) / (l * (1 - (2 * np.cos(theta)**2) / (2 + a))))

    def cuarta_derivada_f_inversa(theta, h):
        return (f_inversa(theta - 2*h) - 4*f_inversa(theta - h) + 6*f_inversa(theta) 
                - 4*f_inversa(theta + h) + f_inversa(theta + 2*h)) / h**4

    f4_inversa = cuarta_derivada_f_inversa((theta_time[0] + theta_time[-1]) / 2, h)
    interval_length = theta_time[-1] - theta_time[0]
    E_t = -((interval_length * h**4) / 180) * f4_inversa

    x_cart_time = (2*l/(2+a)) * np.sin(theta_sampled)
    v_cart_time = (np.cos(theta_sampled)/(2+a)) * np.sqrt((8*g*l*(1-np.cos(theta_sampled)))/(1 - (2*np.cos(theta_sampled)**2)/(2+a)))

    x_cart = (2*l/(2+a)) * np.sin(theta_full)
    omega = np.sqrt((2*g*(1-np.cos(theta_full)))/(l*(1 - (2*np.cos(theta_full)**2)/(2+a))))
    v_cart = (np.cos(theta_full)/(2+a)) * np.sqrt((8*g*l*(1-np.cos(theta_full)))/(1 - (2*np.cos(theta_full)**2)/(2+a)))

    fig = make_subplots(rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.08,
                        subplot_titles=[
                            "Posición del Carro vs Tiempo",
                            "Velocidad Angular vs Ángulo",
                            "Velocidad Angular vs Tiempo",
                            "Velocidad del Carro vs Ángulo",
                            "Velocidad del Carro vs Tiempo"])

    fig.add_trace(go.Scatter(x=cumulative_time, y=x_cart_time, name="x vs t"), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.degrees(theta_full), y=omega, name="ω vs θ"), row=2, col=1)
    fig.add_trace(go.Scatter(x=cumulative_time, y=omega_time[::2], name="ω vs t"), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.degrees(theta_full), y=v_cart, name="v vs θ"), row=4, col=1)
    fig.add_trace(go.Scatter(x=cumulative_time, y=v_cart_time, name="v vs t"), row=5, col=1)

    fig.update_layout(height=1600, showlegend=False, template="plotly_white")
    return fig, f"Error estimado por Simpson: {E_t:.2e}"


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)

