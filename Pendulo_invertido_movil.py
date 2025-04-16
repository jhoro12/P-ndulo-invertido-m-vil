import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
import os
import plotly.express as px

app = dash.Dash(__name__)
app.title = "Péndulo invertido móvil"

g = 9.81

# Layout con sliders, contenedor para las gráficas y la animación
app.layout = html.Div([
    html.H1("Péndulo invertido móvil", style={'textAlign': 'center'}),

    html.Div([
        html.Label("Longitud (l) [m]:"),
        dcc.Slider(id='l-slider', min=0.1, max=8.0, step=0.1, value=1.0,
                   marks={i: str(i) for i in range(1, 9)}),
        
        html.Label("Parámetro (a = M/m):", style={'marginTop': '20px'}),
        dcc.Slider(id='a-slider', min=0.1, max=10.0, step=0.1, value=1.0,
                   marks={i: str(i) for i in range(1, 11)}),
    ], style={'width': '80%', 'margin': 'auto'}),

    html.Div(id='error-output', style={'textAlign': 'center', 'marginTop': '20px'}),
    
    # Gráficas
    dcc.Graph(id='multi-plot'),

    # Animación del carrito y pesas
    html.Div([
        dcc.Graph(id='pendulum-animation', config={'staticPlot': False}),
    ], style={'width': '80%', 'margin': 'auto', 'height': '400px'})
])

@app.callback(
    [Output('multi-plot', 'figure'),
     Output('pendulum-animation', 'figure'),
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
    n_chunks = (len(theta_time) - 1) // 2
    time_intervals = []

    n = 0
    for _ in range(n_chunks):
        fx0 = integrand[n]
        fx1 = integrand[n+1]
        fx2 = integrand[n+2]
        integral = (h/3) * (fx0 + 4*fx1 + fx2)
        time_intervals.append(integral)
        n += 2

    cumulative_time = np.cumsum(np.insert(time_intervals, 0, 0))
    theta_sampled = theta_time[::2]

    # Error truncamiento
    theta_start = 0.1
    theta_end = theta_time[-1]
    interval_length = theta_end - theta_start
    theta_c = (theta_start + theta_end) / 2

    def f_inversa(theta):
        return 1 / np.sqrt((2 * g * (1 - np.cos(theta))) / (l * (1 - (2 * np.cos(theta)**2) / (2 + a))))

    def cuarta_derivada_f_inversa(theta, h):
        return (f_inversa(theta - 2*h) - 4*f_inversa(theta - h) + 6*f_inversa(theta) 
                - 4*f_inversa(theta + h) + f_inversa(theta + 2*h)) / h**4

    f4_inversa = cuarta_derivada_f_inversa(theta_c, h)
    E_t = -((interval_length * h**4) / 180) * f4_inversa

    # Dinámica del carro
    x_cart_time = (2*l/(2+a)) * np.sin(theta_sampled)
    v_cart_time = (np.cos(theta_sampled)/(2+a)) * np.sqrt((8*g*l*(1-np.cos(theta_sampled)))/(1 - (2*np.cos(theta_sampled)**2)/(2+a)))

    # Crear la animación del péndulo
    frames = []
    for t in range(len(cumulative_time)):
        theta = theta_sampled[t]
        x_cart = (2*l/(2+a)) * np.sin(theta)
        y_pendulum = l * np.cos(theta)

        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=[x_cart, x_cart], y=[0, y_pendulum], mode='lines', line=dict(color='blue', width=2),
                ),
                go.Scatter(
                    x=[x_cart], y=[y_pendulum], mode='markers', marker=dict(size=12, color='red'),
                )
            ],
            name=str(t)
        ))

    fig_animation = go.Figure(
        data=[go.Scatter(x=[0], y=[0], mode='markers+lines')],
        layout=go.Layout(
            xaxis=dict(range=[-l, l], autorange=False),
            yaxis=dict(range=[-l, l], autorange=False),
            showlegend=False,
            updatemenus=[dict(type='buttons', showactive=False, buttons=[dict(label='Play', method='animate', args=[None, dict(frame=dict(duration=50, redraw=True), fromcurrent=True)])])]
        ),
        frames=frames
    )

    # Actualizar las gráficas de datos
    fig_data = make_subplots(rows=5, cols=1, shared_xaxes=False, vertical_spacing=0.08,
                             subplot_titles=[
                                 "Posición del Carro vs Tiempo",
                                 "Velocidad Angular vs Ángulo",
                                 "Velocidad Angular vs Tiempo",
                                 "Velocidad del Carro vs Ángulo",
                                 "Velocidad del Carro vs Tiempo"
                             ])

    fig_data.add_trace(go.Scatter(x=cumulative_time, y=x_cart_time, name="x vs t"), row=1, col=1)
    fig_data.add_trace(go.Scatter(x=np.degrees(theta_full), y=omega_time[::2], name="ω vs θ"), row=2, col=1)
    fig_data.add_trace(go.Scatter(x=cumulative_time, y=omega_time[::2], name="ω vs t"), row=3, col=1)
    fig_data.add_trace(go.Scatter(x=np.degrees(theta_full), y=v_cart_time, name="v vs θ"), row=4, col=1)
    fig_data.add_trace(go.Scatter(x=cumulative_time, y=v_cart_time, name="v vs t"), row=5, col=1)

    fig_data.update_layout(height=1600, showlegend=False, template="plotly_white")

    return fig_data, fig_animation, f"Error estimado por Simpson: {E_t:.2e}"

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))  # usa el puerto que Render indique
    app.run(host='0.0.0.0', port=port, debug=False)

