import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import plotly.graph_objs as go
from plotly.subplots import make_subplots

# Inicializar app Dash
dash_app = dash.Dash(__name__)
dash_app.title = "Péndulo invertido móvil"

# Parámetros físicos
g = 9.81  # gravedad

# Layout
dash_app.layout = html.Div([
    html.H1("Péndulo invertido móvil", style={'textAlign': 'center'}),

    # Deslizadores
    html.Div([
        html.Label("Longitud (l) [m]:"),
        dcc.Slider(
            id='l-slider', min=0.1, max=8.0, step=0.1, value=1.0,
            marks={i: str(i) for i in range(1, 9)}
        ),
        html.Label("Parámetro (a = M/m):", style={'marginTop': '20px'}),
        dcc.Slider(
            id='a-slider', min=0.1, max=10.0, step=0.1, value=1.0,
            marks={i: str(i) for i in range(1, 11)}
        ),
    ], style={'width': '80%', 'margin': 'auto'}),

    html.Div(id='error-output', style={'textAlign': 'center', 'marginTop': '20px'}),

    # Gráficas de datos
    dcc.Graph(id='multi-plot'),

    # Canvas para animación
    html.Div([
        html.Canvas(
            id='my-canvas', width=600, height=400,
            style={'border': '1px solid black', 'display': 'block', 'margin': '20px auto'}
        ),
        dcc.Interval(id='interval', interval=50, n_intervals=0),
        dcc.Store(id='traj-data')
    ], style={'width': '80%', 'margin': 'auto'})
])

# Callback Python: actualiza gráficas y trayectorias
@dash_app.callback(
    [Output('multi-plot', 'figure'),
     Output('traj-data', 'data'),
     Output('error-output', 'children')],
    [Input('l-slider', 'value'),
     Input('a-slider', 'value')]
)
def update_data(l, a):
    # Cálculos
    h = 0.01
    theta_full = np.linspace(0, 2*np.pi, 1000)
    theta_time = np.arange(0.1, 2*np.pi - 1e-5, h)
    if len(theta_time) % 2 == 0:
        theta_time = theta_time[:-1]
    omega_time = np.sqrt((2*g*(1-np.cos(theta_time))) / (l*(1 - (2*np.cos(theta_time)**2)/(2+a))))
    integrand = 1 / omega_time
    n_chunks = (len(theta_time) - 1) // 2
    time_intervals = []
    for i in range(n_chunks):
        n0 = 2*i
        fx0, fx1, fx2 = integrand[n0], integrand[n0+1], integrand[n0+2]
        time_intervals.append((h/3)*(fx0 + 4*fx1 + fx2))
    cumulative_time = np.cumsum(np.insert(time_intervals, 0, 0))
    theta_sampled = theta_time[::2]

    # Error de Simpson
    theta_c = 0.5*(theta_time[0] + theta_time[-1])
    def f_inv(th):
        return 1/np.sqrt((2*g*(1-np.cos(th))) / (l*(1 - (2*np.cos(th)**2)/(2+a))))
    def fourth(f, th, H):
        return (f(th-2*H) - 4*f(th-H) + 6*f(th) - 4*f(th+H) + f(th+2*H)) / H**4
    E_t = -((theta_time[-1]-theta_time[0]) * h**4 / 180) * fourth(f_inv, theta_c, h)

    # Datos para gráficas
    x_t = (2*l/(2+a))*np.sin(theta_sampled)
    v_t = (np.cos(theta_sampled)/(2+a))*np.sqrt((8*g*l*(1-np.cos(theta_sampled))) / (1 - (2*np.cos(theta_sampled)**2)/(2+a)))
    omega_full = np.sqrt((2*g*(1-np.cos(theta_full))) / (l*(1 - (2*np.cos(theta_full)**2)/(2+a))))
    v_full = (np.cos(theta_full)/(2+a))*np.sqrt((8*g*l*(1-np.cos(theta_full))) / (1 - (2*np.cos(theta_full)**2)/(2+a)))

    # Figura de datos
    fig = make_subplots(rows=5, cols=1, vertical_spacing=0.08,
                        subplot_titles=["x(t)", "ω(θ)", "ω(t)", "v(θ)", "v(t)"])
    fig.add_trace(go.Scatter(x=cumulative_time, y=x_t), row=1, col=1)
    fig.add_trace(go.Scatter(x=np.degrees(theta_full), y=omega_full), row=2, col=1)
    fig.add_trace(go.Scatter(x=cumulative_time, y=omega_time[::2]), row=3, col=1)
    fig.add_trace(go.Scatter(x=np.degrees(theta_full), y=v_full), row=4, col=1)
    fig.add_trace(go.Scatter(x=cumulative_time, y=v_t), row=5, col=1)
    fig.update_layout(height=1600, showlegend=False, template='plotly_white')

    # Trayectorias para canvas
    x_cart_time = (2*l/(2+a))*np.sin(theta_time)
    y_cart = np.zeros_like(theta_time)
    # Coordenadas de las pesas
    x_pend = x_cart_time + l*np.sin(theta_time)
    y_pend = y_cart + (-l*np.cos(theta_time))

    traj_data = {
        'x_cart': x_cart_time.tolist(),
        'y_cart': y_cart.tolist(),
        'x_pend': x_pend.tolist(),
        'y_pend': y_pend.tolist()
    }

    return fig, traj_data, f"Error de Simpson: {E_t:.2e}"

# Callback cliente: dibuja en canvas
dash_app.clientside_callback(
    """
    function(n, traj) {
        const canvas = document.getElementById('my-canvas');
        const ctx = canvas.getContext('2d');
        const N = traj.x_cart.length;
        const i = n % N;
        // Escalas
        const cx = 300 + 100 * traj.x_cart[i];
        const cy = 200;
        const px = 300 + 100 * traj.x_pend[i];
        const py = 200 + 100 * traj.y_pend[i];

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        // Carrito
        ctx.fillStyle = 'blue'; ctx.fillRect(cx - 20, cy - 10, 40, 20);
        // Línea del péndulo
        ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(px, py);
        ctx.strokeStyle = 'black'; ctx.lineWidth = 2; ctx.stroke();
        // Pesa
        ctx.beginPath(); ctx.arc(px, py, 10, 0, 2*Math.PI);
        ctx.fillStyle = 'red'; ctx.fill();
        return '';
    }
    """,
    Output('traj-data', 'modified_timestamp'),
    Input('interval', 'n_intervals'),
    Input('traj-data', 'data')
)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8050))
    dash_app.run(host='0.0.0.0', port=port, debug=False)


