import numpy as np
import os
import plotly.graph_objects as go
import dash
from dash import dcc, html
from matplotlib.colors import ListedColormap
from utils.helpers import ensure_dir

# BraTS-specific colormap
brats_colors = [(0, 0, 0, 0), (0.1, 0.2, 0.9, 0.6), (0.1, 0.8, 0.1, 0.6), (0.9, 0.1, 0.1, 0.6)]
brats_cmap = ListedColormap(brats_colors)

def map_labels(mask):
    mapped = np.zeros_like(mask)
    mapped[mask == 1] = 1
    mapped[mask == 2] = 2
    mapped[mask == 4] = 3
    return mapped

class BraTSVisualizer:
    def __init__(self, config):
        self.config = config
        self.output_path = os.path.join(config['prediction_output_path'], 'visualizations')
        ensure_dir(self.output_path)
        self.app = dash.Dash(__name__)

    def visualize(self, patient_id, original_mri, ground_truth_seg, predicted_seg):
        # Map segmentation labels for BraTS color coding
        print('visualize .....................')
        gt_volume = map_labels(ground_truth_seg)
        pred_volume = map_labels(predicted_seg)

        # Default slice index (center slice)
        slice_idx = original_mri.shape[2] // 2

        # Layout
        self.app.layout = html.Div([
            html.H1(f"BraTS Visualization: {patient_id}", style={'textAlign': 'center'}),

            html.Div([
                html.Label("Select Plane:"),
                dcc.Dropdown(
                    id='plane-dropdown',
                    options=[
                        {'label': 'Axial', 'value': 'axial'},
                        {'label': 'Coronal', 'value': 'coronal'},
                        {'label': 'Sagittal', 'value': 'sagittal'}
                    ],
                    value='axial'
                ),
            ], style={'width': '30%', 'display': 'inline-block'}),

            html.Div([
                html.Label("Slice Index:"),
                #dcc.Slider(id='slice-slider', min=0, max=original_mri.shape[2]-1, step=1, value=slice_idx)
                dcc.Slider(id='slice-slider', min=0, max=original_mri.shape[2]-1, step=1, value=slice_idx)

            ], style={'width': '60%', 'padding': '20px'}),

            html.Div([
                dcc.Graph(id='visualization-graph')
            ])
        ])

        # Callback for interactivity
        @self.app.callback(
            [dash.dependencies.Output('visualization-graph', 'figure'),
            dash.dependencies.Output('slice-slider', 'max')],
            [dash.dependencies.Input('plane-dropdown', 'value'),
            dash.dependencies.Input('slice-slider', 'value')]
        )
        def update_image(plane, idx):
            if plane == 'axial':
                img_slice = original_mri[:, :, min(idx, original_mri.shape[2]-1)]
                gt_slice = gt_volume[:, :, min(idx, gt_volume.shape[2]-1)]
                pred_slice = pred_volume[:, :, min(idx, pred_volume.shape[2]-1)]
                max_idx = original_mri.shape[2]-1
            elif plane == 'coronal':
                img_slice = original_mri[:, min(idx, original_mri.shape[1]-1), :]
                gt_slice = gt_volume[:, min(idx, gt_volume.shape[1]-1), :]
                pred_slice = pred_volume[:, min(idx, pred_volume.shape[1]-1), :]
                max_idx = original_mri.shape[1]-1
            else:  # sagittal
                img_slice = original_mri[min(idx, original_mri.shape[0]-1), :, :]
                gt_slice = gt_volume[min(idx, gt_volume.shape[0]-1), :, :]
                pred_slice = pred_volume[min(idx, pred_volume.shape[0]-1), :]
                max_idx = original_mri.shape[0]-1

            fig = go.Figure()

            # MRI background (grayscale)
            fig.add_trace(go.Heatmap(
                z=img_slice,
                colorscale='gray',
                showscale=False,
                name='MRI'
            ))

            # Ground Truth overlay
            fig.add_trace(go.Heatmap(
                z=gt_slice,
                colorscale=['rgba(0,0,0,0)', 'blue', 'green', 'red'],
                showscale=False,
                opacity=0.4,
                name='Ground Truth'
            ))

            # Prediction overlay
            fig.add_trace(go.Heatmap(
                z=pred_slice,
                colorscale=['rgba(0,0,0,0)', 'blue', 'green', 'red'],
                showscale=False,
                opacity=0.4,
                name='Prediction'
            ))

            fig.update_layout(
                title=f"{plane.capitalize()} Plane - Slice {idx}",
                xaxis_visible=False,
                yaxis_visible=False
            )

            return fig, max_idx

        self.app.run(debug=True)

