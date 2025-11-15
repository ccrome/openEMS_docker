# -*- coding: utf-8 -*-
"""
 Simple Patch Antenna Tutorial

 Tested with
  - python 3.10
  - openEMS v0.0.34+

 (c) 2015-2023 Thorsten Liebig <thorsten.liebig@gmx.de>

"""

### Import Libraries
import os
import tempfile
import numpy as np
import streamlit as st
import plotly.graph_objects as go

from CSXCAD  import ContinuousStructure
from openEMS import openEMS
from openEMS.physical_constants import *


def get_simulation_parameters():
    """Get all simulation parameters from Streamlit sidebar."""
    st.sidebar.header("Simulation Parameters")
    
    params = {
        'post_proc_only': st.sidebar.checkbox("Post-processing only", value=False),
        'patch_width': st.sidebar.number_input("Patch Width - X direction (mm)", min_value=10.0, max_value=100.0, value=32.0, step=1.0),
        'patch_length': st.sidebar.number_input("Patch Length - Y direction (mm)", min_value=10.0, max_value=100.0, value=40.0, step=1.0),
        'substrate_epsR': st.sidebar.number_input("Substrate εr", min_value=1.0, max_value=10.0, value=3.38, step=0.1),
        'substrate_width': st.sidebar.number_input("Substrate Width - X direction (mm)", min_value=20.0, max_value=200.0, value=60.0, step=5.0),
        'substrate_length': st.sidebar.number_input("Substrate Length - Y direction (mm)", min_value=20.0, max_value=200.0, value=60.0, step=5.0),
        'substrate_thickness': st.sidebar.number_input("Substrate Thickness (mm)", min_value=0.1, max_value=10.0, value=1.524, step=0.1),
        'substrate_cells': st.sidebar.number_input("Substrate Cells", min_value=1, max_value=20, value=4, step=1),
        'feed_pos': st.sidebar.number_input("Feed Position - X direction (mm)", min_value=-50.0, max_value=50.0, value=-6.0, step=1.0),
        'feed_R': st.sidebar.number_input("Feed Resistance (Ω)", min_value=1.0, max_value=200.0, value=50.0, step=1.0),
        'f0': st.sidebar.number_input("Center Frequency (GHz)", min_value=0.1, max_value=10.0, value=2.0, step=0.1) * 1e9,
        'fc': st.sidebar.number_input("Corner Frequency (GHz)", min_value=0.1, max_value=5.0, value=1.0, step=0.1) * 1e9,
    }
    
    # Calculate derived parameters
    params['substrate_kappa'] = 1e-3 * 2*np.pi*2.45e9 * EPS0*params['substrate_epsR']
    params['SimBox'] = np.array([200, 200, 150])
    
    return params


def setup_fdtd(f0, fc):
    """Setup FDTD simulation parameters."""
    FDTD = openEMS(NrTS=30000, EndCriteria=1e-4)
    FDTD.SetGaussExcite(f0, fc)
    FDTD.SetBoundaryCond(['MUR', 'MUR', 'MUR', 'MUR', 'MUR', 'MUR'])
    return FDTD


def create_antenna_structure(FDTD, params):
    """Create the antenna geometry: patch, substrate, ground plane, and feed."""
    CSX = ContinuousStructure()
    FDTD.SetCSX(CSX)
    mesh = CSX.GetGrid()
    mesh.SetDeltaUnit(1e-3)
    
    # Calculate mesh resolution
    mesh_res = C0/(params['f0']+params['fc'])/1e-3/20
    
    # Initialize mesh with air-box dimensions
    mesh.AddLine('x', [-params['SimBox'][0]/2, params['SimBox'][0]/2])
    mesh.AddLine('y', [-params['SimBox'][1]/2, params['SimBox'][1]/2])
    mesh.AddLine('z', [-params['SimBox'][2]/3, params['SimBox'][2]*2/3])
    
    # Create patch (metal)
    patch = CSX.AddMetal('patch')
    patch_start = [-params['patch_width']/2, -params['patch_length']/2, params['substrate_thickness']]
    patch_stop = [params['patch_width']/2, params['patch_length']/2, params['substrate_thickness']]
    patch.AddBox(priority=10, start=patch_start, stop=patch_stop)
    FDTD.AddEdges2Grid(dirs='xy', properties=patch, metal_edge_res=mesh_res/2)
    
    # Create substrate
    substrate = CSX.AddMaterial('substrate', epsilon=params['substrate_epsR'], kappa=params['substrate_kappa'])
    sub_start = [-params['substrate_width']/2, -params['substrate_length']/2, 0]
    sub_stop = [params['substrate_width']/2, params['substrate_length']/2, params['substrate_thickness']]
    substrate.AddBox(priority=0, start=sub_start, stop=sub_stop)
    
    # Add extra cells to discretize the substrate thickness
    mesh.AddLine('z', np.linspace(0, params['substrate_thickness'], params['substrate_cells']+1))
    
    # Create ground plane (same size as substrate)
    gnd = CSX.AddMetal('gnd')
    gnd_start = [-params['substrate_width']/2, -params['substrate_length']/2, 0]
    gnd_stop = [params['substrate_width']/2, params['substrate_length']/2, 0]
    gnd.AddBox(start=gnd_start, stop=gnd_stop, priority=10)
    FDTD.AddEdges2Grid(dirs='xy', properties=gnd)
    
    # Apply excitation & resistance as a current source
    feed_start = [params['feed_pos'], 0, 0]
    feed_stop = [params['feed_pos'], 0, params['substrate_thickness']]
    port = FDTD.AddLumpedPort(1, params['feed_R'], feed_start, feed_stop, 'z', 1.0, priority=5, edges2grid='xy')
    
    # Smooth mesh
    mesh.SmoothMeshLines('all', mesh_res, 1.4)
    
    # Add near-field to far-field recording box
    nf2ff = FDTD.CreateNF2FFBox()
    
    return CSX, port, nf2ff


def create_box_mesh(x_min, x_max, y_min, y_max, z_min, z_max):
    """Create a proper triangular mesh for a 3D box.
    Returns vertices and triangular face indices (i, j, k) for Plotly Mesh3d.
    """
    # Define 8 vertices of the box
    vertices = np.array([
        [x_min, y_min, z_min],  # 0: bottom-front-left
        [x_max, y_min, z_min],  # 1: bottom-front-right
        [x_max, y_max, z_min],  # 2: bottom-back-right
        [x_min, y_max, z_min],  # 3: bottom-back-left
        [x_min, y_min, z_max],  # 4: top-front-left
        [x_max, y_min, z_max],  # 5: top-front-right
        [x_max, y_max, z_max],  # 6: top-back-right
        [x_min, y_max, z_max],  # 7: top-back-left
    ])
    
    # Create triangular faces (each quad face split into 2 triangles)
    # Face indices: i, j, k for each triangle
    i, j, k = [], [], []
    
    # Bottom face (z=z_min): 0-1-2, 0-2-3
    i.extend([0, 0])
    j.extend([1, 2])
    k.extend([2, 3])
    
    # Top face (z=z_max): 4-5-6, 4-6-7
    i.extend([4, 4])
    j.extend([5, 6])
    k.extend([6, 7])
    
    # Front face (y=y_min): 0-1-5, 0-5-4
    i.extend([0, 0])
    j.extend([1, 5])
    k.extend([5, 4])
    
    # Back face (y=y_max): 2-3-7, 2-7-6
    i.extend([2, 2])
    j.extend([3, 7])
    k.extend([7, 6])
    
    # Left face (x=x_min): 0-3-7, 0-7-4
    i.extend([0, 0])
    j.extend([3, 7])
    k.extend([7, 4])
    
    # Right face (x=x_max): 1-2-6, 1-6-5
    i.extend([1, 1])
    j.extend([2, 6])
    k.extend([6, 5])
    
    return vertices, i, j, k


def create_cylinder_mesh(center_x, center_y, z_bottom, z_top, radius, n_points=16):
    """Create a proper triangular mesh for a cylinder.
    Returns vertices and triangular face indices (i, j, k) for Plotly Mesh3d.
    """
    # Generate circle points
    theta = np.linspace(0, 2*np.pi, n_points, endpoint=False)
    x_circ = radius * np.cos(theta)
    y_circ = radius * np.sin(theta)
    
    # Create vertices: bottom circle + top circle
    bottom_vertices = np.column_stack([
        center_x + x_circ,
        center_y + y_circ,
        np.full(n_points, z_bottom)
    ])
    
    top_vertices = np.column_stack([
        center_x + x_circ,
        center_y + y_circ,
        np.full(n_points, z_top)
    ])
    
    vertices = np.vstack([bottom_vertices, top_vertices])
    
    # Create triangular faces
    i, j, k = [], [], []
    
    # Side walls (quads split into triangles)
    for idx in range(n_points):
        next_idx = (idx + 1) % n_points
        bottom_curr = idx
        bottom_next = next_idx
        top_curr = idx + n_points
        top_next = next_idx + n_points
        
        # First triangle: bottom_curr, bottom_next, top_curr
        i.append(bottom_curr)
        j.append(bottom_next)
        k.append(top_curr)
        
        # Second triangle: bottom_next, top_next, top_curr
        i.append(bottom_next)
        j.append(top_next)
        k.append(top_curr)
    
    # Bottom cap: create triangles fanning from first point
    for idx in range(1, n_points - 1):
        i.append(0)
        j.append(idx)
        k.append(idx + 1)
    
    # Top cap: create triangles fanning from first point
    top_start = n_points
    for idx in range(1, n_points - 1):
        i.append(top_start)
        j.append(top_start + idx)
        k.append(top_start + idx + 1)
    
    return vertices, i, j, k


def create_3d_visualization(params):
    """Create and display 3D visualization of the antenna structure."""
    st.header("3D Antenna Structure Visualization")
    
    fig_3d = go.Figure()
    
    # Ground plane (make it more visible by making it slightly thicker)
    gnd_vertices, gnd_i, gnd_j, gnd_k = create_box_mesh(
        -params['substrate_width']/2, params['substrate_width']/2,
        -params['substrate_length']/2, params['substrate_length']/2,
        -0.05, 0.0  # Thicker ground plane for better visibility
    )
    fig_3d.add_trace(go.Mesh3d(
        x=gnd_vertices[:, 0],
        y=gnd_vertices[:, 1],
        z=gnd_vertices[:, 2],
        i=gnd_i,
        j=gnd_j,
        k=gnd_k,
        color='darkgray',
        opacity=0.8,
        name='Ground Plane',
        showlegend=True,
        flatshading=True
    ))
    
    # Substrate
    sub_vertices, sub_i, sub_j, sub_k = create_box_mesh(
        -params['substrate_width']/2, params['substrate_width']/2,
        -params['substrate_length']/2, params['substrate_length']/2,
        0, params['substrate_thickness']
    )
    fig_3d.add_trace(go.Mesh3d(
        x=sub_vertices[:, 0],
        y=sub_vertices[:, 1],
        z=sub_vertices[:, 2],
        i=sub_i,
        j=sub_j,
        k=sub_k,
        color='lightgreen',
        opacity=0.5,
        name='Substrate',
        showlegend=True,
        flatshading=True
    ))
    
    # Patch
    patch_vertices, patch_i, patch_j, patch_k = create_box_mesh(
        -params['patch_width']/2, params['patch_width']/2,
        -params['patch_length']/2, params['patch_length']/2,
        params['substrate_thickness'] - 0.001, params['substrate_thickness'] + 0.001
    )
    fig_3d.add_trace(go.Mesh3d(
        x=patch_vertices[:, 0],
        y=patch_vertices[:, 1],
        z=patch_vertices[:, 2],
        i=patch_i,
        j=patch_j,
        k=patch_k,
        color='gold',
        opacity=0.9,
        name='Patch (Metal)',
        showlegend=True,
        flatshading=True
    ))
    
    # Feed point (cylinder)
    feed_vertices, feed_i, feed_j, feed_k = create_cylinder_mesh(
        params['feed_pos'], 0, 0, params['substrate_thickness'], 0.5, n_points=16
    )
    fig_3d.add_trace(go.Mesh3d(
        x=feed_vertices[:, 0],
        y=feed_vertices[:, 1],
        z=feed_vertices[:, 2],
        i=feed_i,
        j=feed_j,
        k=feed_k,
        color='red',
        opacity=0.8,
        name='Feed Point',
        showlegend=True,
        flatshading=True
    ))
    
    # Update layout
    fig_3d.update_layout(
        title='3D Antenna Structure',
        scene=dict(
            xaxis_title='X (mm)',
            yaxis_title='Y (mm)',
            zaxis_title='Z (mm)',
            aspectmode='data',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2),
                center=dict(x=0, y=0, z=params['substrate_thickness']/2)
            )
        ),
        width=800,
        height=600
    )
    
    st.plotly_chart(fig_3d, use_container_width=True)
    
    # Display structure information
    st.subheader("Structure Dimensions")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Patch Size", f"{params['patch_width']} × {params['patch_length']} mm")
    col2.metric("Substrate Size", f"{params['substrate_width']} × {params['substrate_length']} mm")
    col3.metric("Substrate Thickness", f"{params['substrate_thickness']} mm")
    col4.metric("Feed Position", f"{params['feed_pos']} mm")


def run_simulation(FDTD, sim_path, post_proc_only):
    """Run the FDTD simulation."""
    if not post_proc_only:
        FDTD.Run(sim_path, verbose=3, cleanup=True)


def plot_s11_parameter(f, s11_dB):
    """Create and display S11 parameter plot."""
    fig_s11 = go.Figure()
    
    # Add shaded region for 2.4 GHz ISM band (2.400 - 2.500 GHz)
    fig_s11.add_vrect(
        x0=2.400,
        x1=2.500,
        fillcolor="lightblue",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="2.4 GHz ISM Band",
        annotation_position="top left"
    )
    
    fig_s11.add_trace(go.Scatter(
        x=f/1e9,
        y=s11_dB,
        mode='lines',
        name='S₁₁',
        line=dict(color='black', width=2)
    ))
    fig_s11.update_layout(
        title='S-Parameter S₁₁',
        xaxis_title='Frequency (GHz)',
        yaxis_title='S-Parameter (dB)',
        hovermode='x unified',
        template='plotly_white'
    )
    fig_s11.add_hline(y=-10, line_dash="dash", line_color="red", annotation_text="-10 dB")
    st.plotly_chart(fig_s11, use_container_width=True)


def plot_far_field_pattern(nf2ff_res, f_res, theta):
    """Create and display far-field directivity pattern."""
    E_norm = 20.0*np.log10(nf2ff_res.E_norm[0]/np.max(nf2ff_res.E_norm[0])) + 10.0*np.log10(nf2ff_res.Dmax[0])
    
    fig_ff = go.Figure()
    fig_ff.add_trace(go.Scatter(
        x=theta,
        y=np.squeeze(E_norm[:,0]),
        mode='lines',
        name='xz-plane',
        line=dict(color='black', width=2)
    ))
    fig_ff.add_trace(go.Scatter(
        x=theta,
        y=np.squeeze(E_norm[:,1]),
        mode='lines',
        name='yz-plane',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig_ff.update_layout(
        title=f'Far-Field Directivity Pattern (Frequency: {f_res/1e9:.3f} GHz)',
        xaxis_title='Theta (deg)',
        yaxis_title='Directivity (dBi)',
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig_ff, use_container_width=True)


def plot_input_impedance(f, Zin):
    """Create and display input impedance plot."""
    fig_zin = go.Figure()
    
    # Add shaded region for 2.4 GHz ISM band (2.400 - 2.500 GHz)
    fig_zin.add_vrect(
        x0=2.400,
        x1=2.500,
        fillcolor="lightblue",
        opacity=0.3,
        layer="below",
        line_width=0,
        annotation_text="2.4 GHz ISM Band",
        annotation_position="top left"
    )
    
    fig_zin.add_trace(go.Scatter(
        x=f/1e9,
        y=np.real(Zin),
        mode='lines',
        name='Re{Z_in}',
        line=dict(color='black', width=2)
    ))
    fig_zin.add_trace(go.Scatter(
        x=f/1e9,
        y=np.imag(Zin),
        mode='lines',
        name='Im{Z_in}',
        line=dict(color='red', width=2, dash='dash')
    ))
    fig_zin.update_layout(
        title='Input Impedance',
        xaxis_title='Frequency (GHz)',
        yaxis_title='Z_in (Ohm)',
        hovermode='x unified',
        template='plotly_white'
    )
    st.plotly_chart(fig_zin, use_container_width=True)


def display_key_results(f_res, s11_dB, Zin, idx):
    """Display key simulation results."""
    st.header("Key Results")
    col1, col2, col3 = st.columns(3)
    
    if f_res is not None:
        col1.metric("Resonance Frequency", f"{f_res/1e9:.3f} GHz")
        col2.metric("S₁₁ at Resonance", f"{s11_dB[idx[0]]:.2f} dB")
        col3.metric("Input Impedance at Resonance", 
                   f"{np.real(Zin[idx[0]]):.2f} + j{np.imag(Zin[idx[0]]):.2f} Ω")
    else:
        col1.metric("Resonance Frequency", "Not found")
        col2.metric("S₁₁ at Resonance", "N/A")
        col3.metric("Input Impedance at Resonance", "N/A")


def post_process_results(port, nf2ff, sim_path, f0, fc):
    """Calculate and display post-processing results."""
    st.header("Simulation Results")
    
    # Progress bar for calculation
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Calculate S-parameters
    status_text.text("Calculating S-parameters...")
    f = np.linspace(max(1e9, f0-fc), f0+fc, 401)
    port.CalcPort(sim_path, f)
    s11 = port.uf_ref/port.uf_inc
    s11_dB = 20.0*np.log10(np.abs(s11))
    progress_bar.progress(33)
    
    # Plot S11
    plot_s11_parameter(f, s11_dB)
    
    # Find resonance frequency and calculate far-field
    idx = np.where((s11_dB < -10) & (s11_dB == np.min(s11_dB)))[0]
    if not len(idx) == 1:
        st.warning('No resonance frequency found for far-field calculation')
        f_res = None
        nf2ff_res = None
    else:
        f_res = f[idx[0]]
        status_text.text(f"Calculating far-field at resonance frequency: {f_res/1e9:.3f} GHz...")
        progress_bar.progress(66)
        theta = np.arange(-180.0, 180.0, 2.0)
        phi = [0., 90.]
        nf2ff_res = nf2ff.CalcNF2FF(sim_path, f_res, theta, phi, center=[0, 0, 1e-3])
        
        # Plot far-field pattern
        plot_far_field_pattern(nf2ff_res, f_res, theta)
    
    # Calculate and plot input impedance
    status_text.text("Calculating input impedance...")
    progress_bar.progress(100)
    Zin = port.uf_tot/port.if_tot
    
    plot_input_impedance(f, Zin)
    
    progress_bar.empty()
    status_text.empty()
    
    # Display key results
    display_key_results(f_res, s11_dB, Zin, idx)


def main():
    """Main function to orchestrate the simulation."""
    # Streamlit page configuration
    st.set_page_config(page_title="Simple Patch Antenna Simulation", layout="wide")
    
    st.title("Simple Patch Antenna Tutorial")
    st.markdown("""
    This simulation demonstrates a simple patch antenna using openEMS.
    """)
    
    # Get simulation parameters
    params = get_simulation_parameters()
    
    # Setup simulation path - use absolute path and ensure it exists
    # Use a fixed location in /tmp to avoid path resolution issues
    temp_dir = tempfile.gettempdir()
    sim_path = os.path.join(temp_dir, 'Simp_Patch')
    sim_path = os.path.abspath(sim_path)  # Ensure absolute path
    os.makedirs(sim_path, exist_ok=True)
    
    # Setup FDTD
    FDTD = setup_fdtd(params['f0'], params['fc'])
    
    # Create antenna structure
    CSX, port, nf2ff = create_antenna_structure(FDTD, params)
    
    # Create and display 3D visualization
    create_3d_visualization(params)
    
    # Run simulation
    run_simulation(FDTD, sim_path, params['post_proc_only'])
    
    # Post-process and display results
    post_process_results(port, nf2ff, sim_path, params['f0'], params['fc'])


if __name__ == "__main__":
    main()
