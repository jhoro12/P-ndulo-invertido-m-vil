import os
import base64

import dash
from dash import dcc, html
from dash.dependencies import Input, Output

import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# --- Carga de imagen ---
IMAGE_PATH = 'pendulo.png'
with open(IMAGE_PATH, 'rb') as f:
    encoded_image = base64.b64encode(f.read()).decode()

app = dash.Dash(__name__)
app.title = "Péndulo invertido móvil"

g = 9.81

# --- LAYOUT ---
app.layout = html.Div([
    html.H1("Péndulo invertido móvil", style={'textAlign': 'center'}),

    # Imagen
    html.Img(
        src=f"data:image/png;base64,{encoded_image}",
        style={'width': '30%', 'display': 'block', 'margin': '20px auto'}
    ),

    # Sliders
    html.Div([
        html.Label("Longitud (l) [m]:"),
        dcc.Slider(id='l-slider', min=0.1, max=8.0, step=0.1, value=1.0,
                   marks={i: str(i) for i in range(1, 9)}),

        html.Label("Parámetro (a = M/m):", style={'marginTop': '20px'}),
        dcc.Slider(id='a-slider', min=0.1, max=10.0, step=0.1, value=1.0,
                   marks={i: str(i) for i in range(1, 11)}),
    ], style={'width': '80%', 'margin': 'auto'}),

    # Mensaje de error
    html.Div(id='error-output', style={'textAlign': 'center', 'marginTop': '20px'}),

    # Gráficas de datos
    dcc.Graph(id='multi-plot'),

    # Animación del péndulo
    dcc.Graph(id='pendulum-animation', style={'height': '400px'})
])


# --- CALLBACK ---
@app.callback(
    [Output('multi-plot', 'figure'),
     Output('pendulum-animation', 'figure'),
     Output('error-output', 'children')],
    [Input('l-slider', 'value'),
     Input('a-slider', 'value')]
)
def update_all(l, a):
    # Cálculos comunes
    h = 0.01
    theta_full = np.linspace(0, 2*np.pi, 1000)
    theta_time = np.arange(0.1, 2*np.pi - 0.00001, h)
    if len(theta_time) % 2 == 0:
        theta_time = theta_time[:-1]
    omega_time = np.sqrt((2*g*(1-np.cos(theta_time))) /
                         (l*(1 - (2*np.cos(theta_time)**2)/(2+a))))
    integrand = 1/omega_time
    n_chunks = (len(theta_time) - 1) // 2
    intervals = []
    for i in range(n_chunks):
        n0 = 2*i
        intervals.append((h/3)*(integrand[n0] + 4*integrand[n0+1] + integrand[n0+2]))
    cumulative_time = np.cumsum(np.insert(intervals, 0, 0))
    theta_sampled = theta_time[::2]

    # Error de Simpson
    theta_c = (theta_time[0] + theta_time[-1]) / 2
    def f_inv(t): return 1/np.sqrt((2*g*(1-np.cos(t))) /
                                     (l*(1 - (2*np.cos(t)**2)/(2+a))))
    def fourth(f, t, h):
        return (f(t-2*h) - 4*f(t-h) + 6*f(t) - 4*f(t+h) + f(t+2*h)) / h**4
    E_t = -((theta_time[-1]-theta_time[0]) * h**4 / 180) * fourth(f_inv, theta_c, h)

    # Datos para gráficas
    x_t = (2*l/(2+a))*np.sin(theta_sampled)
    v_t = (np.cos(theta_sampled)/(2+a))* \
        np.sqrt((8*g*l*(1-np.cos(theta_sampled))) /
                (1 - (2*np.cos(theta_sampled)**2)/(2+a)))
    omega_full = np.sqrt((2*g*(1-np.cos(theta_full))) /
                         (l*(1 - (2*np.cos(theta_full)**2)/(2+a))))
    v_full = (np.cos(theta_full)/(2+a))* \
        np.sqrt((8*g*l*(1-np.cos(theta_full))) /
                (1 - (2*np.cos(theta_full)**2)/(2+a)))

    # --- Figura de datos ---
    fig_data = make_subplots(rows=5, cols=1, vertical_spacing=0.08,
                             subplot_titles=[
                                 "x(t)", "ω(θ)", "ω(t)", "v(θ)", "v(t)"
                             ])
    fig_data.add_trace(go.Scatter(x=cumulative_time, y=x_t), row=1, col=1)
    fig_data.add_trace(go.Scatter(x=np.degrees(theta_full), y=omega_full), row=2, col=1)
    fig_data.add_trace(go.Scatter(x=cumulative_time, y=omega_time[::2]), row=3, col=1)
    fig_data.add_trace(go.Scatter(x=np.degrees(theta_full), y=v_full), row=4, col=1)
    fig_data.add_trace(go.Scatter(x=cumulative_time, y=v_t), row=5, col=1)
    fig_data.update_layout(height=1600, showlegend=False, template='plotly_white')

    # --- Animación del péndulo ---
    frames = []
    for i, th in enumerate(theta_sampled):
        x_cart = (2*l/(2+a))*np.sin(th)
        y_p = -l*np.cos(th)
        frames.append(go.Frame(
            data=[
                # Carrito como línea horizontal (o podrías usar un rectángulo)
                go.Scatter(x=[x_cart-0.2, x_cart+0.2], y=[0, 0], mode='lines', line=dict(width=10, color='black')),
                # Masa como círculo rojo
                go.Scatter(x=[x_cart], y=[y_p], mode='markers', marker=dict(size=12, color='red'))
            ],
            name=str(i)
        ))

    fig_anim = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            xaxis=dict(range=[-l-1, l+1], autorange=False),
            yaxis=dict(range=[-l-1, 1], autorange=False),
            showlegend=False,
            updatemenus=[dict(
                type='buttons',
                buttons=[dict(label='▶︎ Play',
                              method='animate',
                              args=[None, {"frame": {"duration": 50, "redraw": True},
                                           "fromcurrent": True}])]
            )]
        )
    )

    return fig_data, fig_anim, f"Error estimado: {E_t:.2e}"

# --- RUN SERVER ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    app.run(host='0.0.0.0', port=port, debug=False)

