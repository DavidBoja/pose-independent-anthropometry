# import _thread as thread
import visdom
# import os
import plotly.graph_objects as go

# from utils import create_body_model, SMPL_INDEX_LANDAMRKS_REVISED


class Visualizer(object):
    """
    Visdom viusualization of training loss curves
    """
    def __init__(self, port, env):
        super(Visualizer, self).__init__()
        # thread.start_new_thread(os.system, (f"visdom -p {port} > /dev/null 2>&1",))
        vis = visdom.Visdom(port=port, env=env)
        self.env = env
        self.vis = vis


def visualize_lm_connections(fig,landmarks,list_of_landmark_tuple_names, landmarks_order):
    """
    Visualize landmarks and segments between them given by list_of_landmark_tuple_names
    """


    connection_indices = [(landmarks_order.index(name1),landmarks_order.index(name2)) 
                          for (name1,name2) in  list_of_landmark_tuple_names]

    
    x = []
    y = []
    z = []
    
    for ci in connection_indices:
        for i in [0,1]:
            x.append(landmarks[ci[i],0])
            y.append(landmarks[ci[i],1])
            z.append(landmarks[ci[i],2])
        x.append(None)
        y.append(None)
        z.append(None)
        
    plot_segm = go.Scatter3d(x=x, 
                            y=y, 
                            z=z,
                            marker=dict(
                                    size=8,
                                    color="rgba(1,1,1,1)",
                                ),
                                line=dict(
                                    color="red",
                                    width=10),
                                name="LM segments"
                                )
    fig.add_trace(plot_segm)

    return fig

def viz_mesh(fig,verts,faces,color="green",name="mesh", opacity=1):
    """
    Visualize mesh with plotly
    """
        
    plot_mesh = go.Mesh3d(
        x=verts[:,0],
        y=verts[:,1],
        z=verts[:,2],
        i=faces[:,0],
        j=faces[:,1],
        k=faces[:,2],
        name=name,
        color=color,
        opacity=opacity
    )
    
    fig.add_trace(plot_mesh)
    return fig

def viz_scatter(fig,pts,color,pt_size=8,name="pts",symbol="circle"):
    """
    Visualize point cloud with plotly
    """
    
    plot_subj = go.Scatter3d(x = pts[:,0], 
                             y = pts[:,1], 
                             z = pts[:,2], 
                           mode='markers',
                           marker=dict(
                               color=color,
                               size=pt_size,
                               symbol=symbol,
                               line=dict(
                                   color='black',
                                   width=1)
                           ),
                           name=name)
    fig.add_trace(plot_subj)

    return fig