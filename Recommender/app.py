import dash
from dash import dcc, Output, Input
from dash import html
import dash_bootstrap_components as dbc

# Local File
import recommend
import analysis

theme = [dbc.themes.BOOTSTRAP]
app = dash.Dash(__name__, external_stylesheets=theme, suppress_callback_exceptions=True,
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])
server = app.server
app.title = 'Image Search'
app._favicon = "shop.png"

app.layout = dbc.Container([
    dbc.NavbarSimple(
        children=[

            dbc.NavItem(dbc.NavLink(
                html.A("Recommend", href="/", className="me-1 text-white text-decoration-none fs-5", id='predict_link',
                       n_clicks=0)))
        
        ], fixed='top',
        brand="Image Search",
        brand_href="/",
        color="warning",
        dark=True,
        className='py-0'),
    html.Br(),
    html.Br(),

    dbc.Row([
        dcc.Location(id='url', refresh=True),
    ], id='display')

], fluid=True)


@app.callback(
    Output('display', 'children'),
    Input('url', 'pathname'))
def update(x):
    if x == '/':
        return recommend.recommend_div
        pass


if __name__ == '__main__':
    app.run_server(host='127.0.0.1', port=8080, debug=False)
