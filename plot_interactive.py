import dash
from dash import dcc, html, Input, Output, State
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import gymnasium as gym
import dill
import numpy as np
import seaborn as sns
from copy import deepcopy

from four_room.env import FourRoomsEnv
from four_room.constants import train_config, size
from four_room.wrappers import gym_wrapper
from io import BytesIO
import base64
gym.register('MiniGrid-FourRooms-v1', FourRoomsEnv)

# Your existing helper functions
def find_states(history: list, timestep: int, capacity: int):
    """Find states within the buffer capacity window."""
    states = []
    begin = timestep - capacity if timestep > capacity else 0
    
    for i in range(len(history)):
        step, context, x, y = history[i]    
        if begin <= step-1 <= timestep:
            states.append((context, x, y))
    
    return states


def plot_env_heatmap(results, context_info: tuple, label: str, title: str, ax=None, annot: bool = True, intize: bool = True):
    """Plot environment heatmap with seaborn styling."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 5))
    matrix = results
    colors = np.full((19, 19), np.nan, dtype=float)
    colors[0, :] = 0
    colors[-1, :] = 0
    colors[:, 0] = 0
    colors[:, -1] = 0
    colors[9, :] = 0
    colors[:, 9] = 0
    start_pos, doors_pos, goal_pos, valid_pos = context_info
    room_w = 19 // 2
    room_h = 19 // 2
    highlight_cells = []
    for j in range(2):
        for i in range(2):
            xL = i * room_w
            yT = j * room_h
            xR = xL + room_w
            yB = yT + room_h
            if i + 1 < 2:
                pos = (xR, yT + 1 + doors_pos[j])
                highlight_cells.append(pos)
            if j + 1 < 2:
                pos = (xL + 1 + doors_pos[2 + i], yB)
                highlight_cells.append(pos)
    colors[start_pos[0], start_pos[1]] = 1
    colors[goal_pos[0], goal_pos[1]] = 2
    
    for r, c in highlight_cells:
        colors[r, c] = np.nan
    matrix_data = deepcopy(matrix).astype(str)
    
    for r in range(matrix_data.shape[0]):
        for c in range(matrix_data.shape[1]):
            # if (r, c) in valid_pos:
            #     matrix_data[r, c] = str(int(matrix[r, c])) if intize else str(round(matrix[r, c], 2))
            # else:
            #     matrix_data[r, c] = 'Wall'
            matrix_data[r, c] = str(int(matrix[r, c])) if intize else str(round(matrix[r, c], 2))
    matrix_data[goal_pos[0], goal_pos[1]] = 'Goal'
    
    sns.heatmap(
        matrix,
        cmap='magma',
        annot=matrix_data if annot else False,
        cbar_kws={'label': label},
        ax=ax,
        xticklabels=False, 
        yticklabels=False,
        fmt=''
    )
    overlay_cmap = sns.color_palette(["grey", "red", "lime"])
    sns.heatmap(
        colors,
        cmap=overlay_cmap,
        mask=np.isnan(colors),
        cbar=False,
        alpha=0.6,
        xticklabels=False, 
        yticklabels=False,
        ax=ax
    )
    ax.set_title(f"{title}")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return ax


def plot_timestep_matplotlib(file_name: str, timestep: int, timestep_idx: int, annot_expl: bool = True, 
                             results=None, env=None):
    """
    Create matplotlib figure for a single timestep (original function logic)
    """
    if results is None:
        with open(f'results/dqn_exps/{file_name}.pl', 'rb') as file:
            results = dill.load(file)
    
    if env is None:
        env = gym_wrapper(gym.make(
                'MiniGrid-FourRooms-v1', 
                agent_pos=train_config['agent positions'],
                goal_pos=train_config['goal positions'],
                doors_pos=train_config['topologies'],
                agent_dir=train_config['agent directions'],
                size=size, 
                render_mode='rgb_array',
                disable_env_checker=True
            ),
            original_obs=True
        )  
    
    # get switches
    switch_history = results['switch_states']
    buffer_switch_counts = np.zeros_like(results['heatmap'])
    buffer_moving = results['counter_moving']
    buffer_full = results['counter_full']
    full_switches = results['heatmap']
    explored_states = results['explore_heatmap']
    aux_states = results['aux_heatmap']
    dqn_scores = results['dqn_scores'][timestep_idx]
    
    relevant_switches = find_states(switch_history, timestep, buffer_moving.capacity)
    for switch in relevant_switches:
        buffer_switch_counts[*switch] += 1
    
    relevant_buffer_states = find_states(buffer_moving.all_states, timestep, buffer_moving.capacity)
    relevant_buffer = np.zeros_like(aux_states)
    for state in relevant_buffer_states:
        relevant_buffer[*state] += 1
        
    context = results['context_history'][timestep]
    context_info = env.get_wrapper_attr('context_info')(context)
    env.get_wrapper_attr('set_context')(context)
    valid_pos = env.get_wrapper_attr('valid_pos')
    context_info = (*context_info, valid_pos)
    
    fig, axes = plt.subplots(4, 2, figsize=(20, 20))
    
    # moving switch count
    plot_env_heatmap(buffer_switch_counts[context], context_info,
                     'Switch Counts', f'Switches Heatmap for context {context}', axes[1, 0]) 
    # moving buffer count
    plot_env_heatmap(relevant_buffer[context], context_info, 
                     'Buffer Counts', f'Buffer Heatmap for context {context}', axes[1, 1]) 
    # full switches
    plot_env_heatmap(full_switches[context], context_info, 
                     'Switch Counts (full)', f'Switches Heatmap (full) for context {context}', axes[2, 0]) 
    # full buffer
    plot_env_heatmap(buffer_full.counts_matrix[context], context_info, 
                     'Buffer Counts (full)', f'Buffer Heatmap (full) for context {context}', axes[2, 1])
    # explored states 
    plot_env_heatmap(explored_states[context], context_info, 
                     'Explore Counts', f'Explored States (full) for context {context}', axes[3, 0], annot=annot_expl) 
    # aux selected states
    plot_env_heatmap(aux_states[context], context_info, 
                     'Aux Counts', f'Aux Selected Heatmap (full) for context {context}', axes[3, 1], annot=annot_expl)
    
    plt.tight_layout()
    
    return fig, axes, context


def generate_timestep_data(file_name: str, timesteps: list, annot_expl: bool = True):
    # Load results once
    with open(f'results/dqn_exps/{file_name}.pl', 'rb') as file:
        results = dill.load(file)
    
    # Create env once
    env = gym_wrapper(gym.make(
            'MiniGrid-FourRooms-v1', 
            agent_pos=train_config['agent positions'],
            goal_pos=train_config['goal positions'],
            doors_pos=train_config['topologies'],
            agent_dir=train_config['agent directions'],
            size=size, 
            render_mode='rgb_array',
            disable_env_checker=True
        ),
        original_obs=True
    )
    
    timestep_data = {}
    
    print(f"Generating {len(timesteps)} matplotlib figures...")
    for i, timestep in enumerate(timesteps):
        print(f"  Processing timestep {timestep} ({i+1}/{len(timesteps)})")
        
        fig, axes, context = plot_timestep_matplotlib(
            file_name, timestep, i, annot_expl, results, env
        )
        
        # Convert matplotlib figure to base64 image
        buf = BytesIO()
        fig.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        
        timestep_data[timestep] = {
            'image': f'data:image/png;base64,{img_base64}',
            'context': context
        }
        
        plt.close(fig)
        buf.close()
    
    env.close()
    print("Done generating figures!")
    
    return timestep_data


def create_dash_app(file_name: str, timesteps: list, annot_expl: bool = True, port: int = 8050):
    # Pre-generate all data
    print("Loading data and generating visualizations...")
    timestep_data = generate_timestep_data(file_name, timesteps, annot_expl)
    
    # Initialize Dash app
    app = dash.Dash(__name__)
    
    # Define layout
    app.layout = html.Div([
        html.Div([
            html.H1("Count Based Timestep Visualization", 
                   style={'textAlign': 'center', 'color': '#fff', 'marginBottom': '10px'}),
            html.P("Interactive exploration of agent behavior across timesteps",
                  style={'textAlign': 'center', 'color': '#fff', 'fontSize': '1.1em'})
        ], style={
            'background': 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
            'padding': '30px',
            'marginBottom': '20px'
        }),
        
        html.Div([
            html.Div([
                html.Div([
                    html.Div([
                        html.H3(f"Timestep: ", style={'display': 'inline', 'color': '#495057'}),
                        html.H2(id='current-timestep', children=str(timesteps[0]), 
                               style={'display': 'inline', 'color': '#667eea', 'marginLeft': '10px'}),
                        html.Span(f" / {timesteps[-1]}", 
                                 style={'color': '#adb5bd', 'fontSize': '0.8em', 'marginLeft': '5px'})
                    ], style={'marginBottom': '20px'}),
                    
                    html.Div([
                        html.Button('◄ Previous', id='prev-button', n_clicks=0,
                                   style={
                                       'padding': '10px 20px',
                                       'marginRight': '10px',
                                       'background': '#667eea',
                                       'color': 'white',
                                       'border': 'none',
                                       'borderRadius': '5px',
                                       'cursor': 'pointer',
                                       'fontSize': '1em'
                                   }),
                        dcc.Slider(
                            id='timestep-slider',
                            min=0,
                            max=len(timesteps) - 1,
                            value=0,
                            marks={i: str(timesteps[i]) for i in range(0, len(timesteps), max(1, len(timesteps)//10))},
                            step=1,
                            tooltip={"placement": "bottom", "always_visible": True}
                        ),
                        html.Button('Next ►', id='next-button', n_clicks=0,
                                   style={
                                       'padding': '10px 20px',
                                       'marginLeft': '10px',
                                       'background': '#667eea',
                                       'color': 'white',
                                       'border': 'none',
                                       'borderRadius': '5px',
                                       'cursor': 'pointer',
                                       'fontSize': '1em'
                                   })
                    ], style={
                        'display': 'flex',
                        'alignItems': 'center',
                        'gap': '20px',
                        'marginBottom': '30px'
                    }),
                    
                    html.Div([
                        html.Div([
                            html.Div("Current Index", style={'fontSize': '0.9em', 'color': '#6c757d', 'marginBottom': '5px'}),
                            html.Div(id='current-index', children='0', style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#495057'})
                        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center'}),
                        
                        html.Div([
                            html.Div("Total Timesteps", style={'fontSize': '0.9em', 'color': '#6c757d', 'marginBottom': '5px'}),
                            html.Div(str(len(timesteps)), style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#495057'})
                        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center'}),
                        
                        html.Div([
                            html.Div("Context", style={'fontSize': '0.9em', 'color': '#6c757d', 'marginBottom': '5px'}),
                            html.Div(id='current-context', children='-', style={'fontSize': '2em', 'fontWeight': 'bold', 'color': '#495057'})
                        ], style={'background': '#f8f9fa', 'padding': '20px', 'borderRadius': '10px', 'textAlign': 'center'})
                    ], style={
                        'display': 'grid',
                        'gridTemplateColumns': 'repeat(3, 1fr)',
                        'gap': '20px'
                    })
                ], style={
                    'background': 'white',
                    'padding': '30px',
                    'borderRadius': '10px',
                    'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
                })
            ], style={'marginBottom': '30px'}),
            
            html.Div([
                html.Img(id='timestep-image', 
                        style={'width': '100%', 'borderRadius': '10px', 'boxShadow': '0 4px 12px rgba(0,0,0,0.1)'})
            ], style={
                'background': 'white',
                'padding': '20px',
                'borderRadius': '10px',
                'boxShadow': '0 4px 6px rgba(0,0,0,0.1)'
            })
        ], style={
            'maxWidth': '1600px',
            'margin': '0 auto',
            'padding': '20px'
        })
    ], style={
        'background': 'linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%)',
        'minHeight': '100vh',
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Oxygen, Ubuntu, Cantarell, sans-serif'
    })
    
    # Callbacks
    @app.callback(
        [Output('timestep-slider', 'value'),
         Output('timestep-image', 'src'),
         Output('current-timestep', 'children'),
         Output('current-index', 'children'),
         Output('current-context', 'children')],
        [Input('timestep-slider', 'value'),
         Input('prev-button', 'n_clicks'),
         Input('next-button', 'n_clicks')],
        [State('timestep-slider', 'value')]
    )
    def update_display(slider_value, prev_clicks, next_clicks, current_value):
        ctx = dash.callback_context
        
        if not ctx.triggered:
            idx = 0
        else:
            trigger_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if trigger_id == 'prev-button':
                idx = max(0, current_value - 1)
            elif trigger_id == 'next-button':
                idx = min(len(timesteps) - 1, current_value + 1)
            else:
                idx = slider_value
        
        timestep = timesteps[idx]
        data = timestep_data[timestep]
        
        return idx, data['image'], str(timestep), str(idx), str(data['context'])
    
    # Run the app
    print(f"\n Starting Dash app on http://localhost:{port}")
    app.run(debug=True, port=port)


if __name__ == '__main__':
    timesteps_to_plot = list(range(100000-10, 100000+1))
    
    create_dash_app('dqn_count_test_seed_0_100000', timesteps_to_plot, annot_expl=True, port=8050)
