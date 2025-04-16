import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import numpy as np
import json

# Inicialización de la app
app = dash.Dash(__name__)
app.title = "Péndulo invertido móvil"

g = 9.81  # Aceleración debido a la gravedad
l = 1.0  # Longitud del péndulo
a = 1.0  # Parámetro (M/m)
theta_time = np.linspace(0, 2 * np.pi, 200)

# Trayectorias para la animación
x_cart_time = (2 * l / (2 + a)) * np.sin(theta_time)
y_pend_time = -l * np.cos(theta_time)

# Guardamos las posiciones en un diccionario para pasarlas al cliente
traj = {
    'x': x_cart_time.tolist(),
    'y': y_pend_time.tolist(),
    'theta': theta_time.tolist()
}

# Layout de la app
app.layout = html.Div([
    html.H1("Péndulo invertido móvil", style={'textAlign': 'center'}),

    # Canvas para la animación
    html.Canvas(id='my-canvas', width=600, height=400, style={'border': '1px solid black'}),

    # Intervalo para animar
    dcc.Interval(id='interval', interval=50, n_intervals=0),

    # Almacenamos las trayectorias
    dcc.Store(id='traj-data', data=traj)
])

# Callback en JavaScript para dibujar en el canvas
app.clientside_callback(
    """
    function(n, traj) {
        const canvas = document.getElementById('my-canvas');
        const ctx = canvas.getContext('2d');
        const N = traj.x.length;
        const i = n % N;  // índice cíclico para animación

        // Mapeamos las posiciones de las trayectorias
        const cx = 300 + 100 * traj.x[i];  // Ajuste de escala para la posición x del carrito
        const cy = 200;  // El carrito se mueve en una línea horizontal en y = 200
        const theta = traj.theta[i];

        // Calculamos la posición de la pesa en base al ángulo
        const x_pend = cx + 100 * Math.sin(theta);  // Posición circular de la pesa
        const y_pend = cy - 100 * Math.cos(theta);  // Posición circular de la pesa

        // Limpia el canvas en cada frame
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Dibuja el carrito como un rectángulo
        ctx.fillStyle = 'blue';
        ctx.fillRect(cx - 20, cy - 10, 40, 20);  // Carrito moviéndose a lo largo de la línea

        // Dibuja la línea del péndulo
        ctx.beginPath();
        ctx.moveTo(cx, cy);
        ctx.lineTo(x_pend, y_pend);
        ctx.strokeStyle = 'black';
        ctx.lineWidth = 2;
        ctx.stroke();

        // Dibuja la pesa como un círculo rojo
        ctx.beginPath();
        ctx.arc(x_pend, y_pend, 10, 0, 2*Math.PI);
        ctx.fillStyle = 'red';
        ctx.fill();

        return '';
    }
    """,
    Output('traj-data', 'modified_timestamp'),  # Un dummy output, ya que no necesitamos un cambio visible
    Input('interval', 'n_intervals'),
    Input('traj-data', 'data')
)

if __name__ == '__main__':
    app.run(debug=True)

