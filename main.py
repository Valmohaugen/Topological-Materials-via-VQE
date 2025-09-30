import os, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
from qiskit.primitives import StatevectorEstimator, BackendEstimatorV2
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist, squareform
from scipy.optimize import minimize

# 
num_materials = 500
num_selected_materials = 3
diversity_weight = 0.3

# 
num_qubits = 2
reps = 5
shots = 4096
seed = 42
maxiter = 250
entanglement = 'full'
optimization_level = 3

# Kitaev-chain model variables
mu_values = np.linspace(-3, -3, 25)
t_values = np.linspace(0.5, 2.0, 1)
delta_values = np.linspace(0.5, 2.0, 1)

# 
script_dir = os.path.dirname(os.path.abspath(__file__))


# 
class QuantumMaterialDatabase:
    def __init__(self, num_materials):
        self.num_materials = num_materials
        self.materials = self.generate_materials_database()
    
    def generate_materials_database(self):
        # Material composition
        transition_metal_content = np.random.uniform(0, 1, self.num_materials)
        rare_earth_content = np.random.uniform(0, 1, self.num_materials)
        carbon_content = np.random.uniform(0, 1, self.num_materials)
        main_group_content = np.random.uniform(0, 1, self.num_materials)
        alkali_metal_content = np.random.uniform(0, 1, self.num_materials)
        halogen_content = np.random.uniform(0, 1, self.num_materials)
        post_transition_metal_content = np.random.uniform(0, 1, self.num_materials)
        
        # Normalize the material composition
        total_content = (transition_metal_content + rare_earth_content + carbon_content + main_group_content + alkali_metal_content + halogen_content + post_transition_metal_content)
        transition_metal_content /= total_content
        rare_earth_content /= total_content
        carbon_content /= total_content
        main_group_content /= total_content
        alkali_metal_content /= total_content
        halogen_content /= total_content
        post_transition_metal_content /= total_content
        
        # Electronic structure
        s_electron_count = np.random.uniform(0, 2, self.num_materials)
        p_electron_count = np.random.uniform(0, 6, self.num_materials)
        d_electron_count = np.random.uniform(0, 10, self.num_materials)
        f_electron_count = np.random.uniform(0, 14, self.num_materials)
        total_valence_electrons = (s_electron_count + p_electron_count + d_electron_count + f_electron_count)

        # Lattice and QC properties
        crystalline_symmetry = np.random.choice(['cubic', 'hexagonal', 'tetragonal', 'trigonal', 'orthorhombic', 'monoclinic', 'triclinic'], self.num_materials)
        spin_orbit_coupling = np.random.uniform(0, 2, self.num_materials)        # 
        magnetic_anisotropy = np.random.uniform(-1, 1, self.num_materials)       # 
        electron_phonon_coupling = np.random.uniform(0, 1, self.num_materials)   # 

        # Environmental stability
        thermal_stability = np.random.uniform(50, 800, self.num_materials)        # 50-800 Kelvin
        chemical_stability = np.random.uniform(0, 1, self.num_materials)          # 
        
        # Manufacturing complexity
        synthesis_difficulty = np.random.uniform(0, 1, self.num_materials)       # 
        cost_factor = np.random.uniform(0.1, 10, self.num_materials)             # 

        # Material database
        materials_db = pd.DataFrame({
            'material_id': range(1, self.num_materials+1),
            'transition_metal_content': transition_metal_content,
            'rare_earth_content': rare_earth_content,
            'carbon_content': carbon_content,
            'main_group_content': main_group_content,
            'alkali_metal_content': alkali_metal_content,
            'halogen_content': halogen_content,
            'post_transition_metal_content': post_transition_metal_content,
            's_electron_count': s_electron_count,
            'p_electron_count': p_electron_count,
            'd_electron_count': d_electron_count,
            'f_electron_count': f_electron_count,
            'total_valence_electrons': total_valence_electrons,
            'crystalline_symmetry': crystalline_symmetry,
            'spin_orbit_coupling': spin_orbit_coupling,
            'magnetic_anisotropy': magnetic_anisotropy,
            'electron_phonon_coupling': electron_phonon_coupling,
            'thermal_stability': thermal_stability,
            'chemical_stability': chemical_stability,
            'synthesis_difficulty': synthesis_difficulty,
            'cost_factor': cost_factor
        })

        return materials_db
    
    def get_material_properties(self, material_id):
        return self.materials.iloc[material_id-1]
    
class QuantumPropertyCalculator:
    def __init__(self):
        self.estimator = StatevectorEstimator()
        self.cache = {}

    def create_material_hamiltonian(self, material_properties):
        spin_orbit_coupling = material_properties['spin_orbit_coupling']
        magnetic_anisotropy = material_properties['magnetic_anisotropy']
        s_electrons = material_properties['s_electron_count']
        p_electrons = material_properties['p_electron_count']
        d_electrons = material_properties['d_electron_count']
        f_electrons = material_properties['f_electron_count']
        transition_metal_content = material_properties['transition_metal_content']
        rare_earth_content = material_properties['rare_earth_content']
        main_group_content = material_properties['main_group_content']
        alkali_content = material_properties['alkali_metal_content']
        halogen_content = material_properties['halogen_content']
        post_transition_metal_content = material_properties['post_transition_metal_content']

        pauli_terms = [
            ('ZZ', -0.5 * transition_metal_content),
            ('XX', -0.3 * d_electrons / 10),
            ('YY', -0.3 * d_electrons / 10),
            ('ZI', -spin_orbit_coupling * magnetic_anisotropy),
            ('IZ', -spin_orbit_coupling * magnetic_anisotropy),
            ('XY', 0.2 * spin_orbit_coupling),
            ('YX', -0.2 * spin_orbit_coupling),
            ('ZZ', 0.1 * main_group_content),
            ('XZ', -0.1 * rare_earth_content),
            ('ZX', 0.1 * alkali_content),
            ('YZ', 0.1 * halogen_content),
            ('ZY', 0.1 * -post_transition_metal_content),
            ('XX', 0.1 * s_electrons),
            ('YY', -0.1 * p_electrons),
            ('ZZ', 0.05 * f_electrons)
        ]

        return SparsePauliOp.from_list(pauli_terms)
    
    def calculate_band_gap(self, material_properties):
        hamiltonian = self.create_material_hamiltonian(material_properties)
        eigenvalues = np.linalg.eigvals(hamiltonian.to_matrix())
        eigenvalues = np.sort(np.real(eigenvalues))
        band_gap = eigenvalues[2] - eigenvalues[1] if len(eigenvalues) > 2 else eigenvalues[1] - eigenvalues[0]

        return max(0, band_gap)
    
    def calculate_berry_curvature(self, material_properties):
        spin_orbit_coupling = material_properties['spin_orbit_coupling']
        magnetic_anisotropy = material_properties['magnetic_anisotropy']
        s_electrons = material_properties['s_electron_count']
        p_electrons = material_properties['p_electron_count']
        d_electrons = material_properties['d_electron_count']
        f_electrons = material_properties['f_electron_count']
        transition_metal_content = material_properties['transition_metal_content']
        rare_earth_content = material_properties['rare_earth_content']
        main_group_content = material_properties['main_group_content']
        alkali_content = material_properties['alkali_metal_content']
        halogen_content = material_properties['halogen_content']
        post_transition_metal_content = material_properties['post_transition_metal_content']

        berry_curvature = (
            spin_orbit_coupling * np.sin(magnetic_anisotropy + np.pi) * (d_electrons/10) +
            s_electrons * 0.1 + 
            p_electrons * 0.2 + 
            f_electrons * 0.05 + 
            transition_metal_content * 0.3 +
            rare_earth_content * 0.2 + 
            main_group_content * 0.1 + 
            alkali_content * 0.1 + 
            halogen_content * 0.15 + 
            post_transition_metal_content * 0.1
        )

        return berry_curvature
    
    def calculate_coherence_time(self, material_properties):
        electron_phonon_coupling =  material_properties['electron_phonon_coupling']
        thermal_stability =  material_properties['thermal_stability']
        chemical_stability =  material_properties['chemical_stability']
        s_electrons = material_properties['s_electron_count']
        p_electrons = material_properties['p_electron_count']
        d_electrons = material_properties['d_electron_count']
        f_electrons = material_properties['f_electron_count']
        transition_metal_content = material_properties['transition_metal_content']
        rare_earth_content = material_properties['rare_earth_content']
        main_group_content = material_properties['main_group_content']
        alkali_content = material_properties['alkali_metal_content']
        halogen_content = material_properties['halogen_content']
        post_transition_metal_content = material_properties['post_transition_metal_content']

        base_coherence = 100.0 # microseconds
        coherence_time = (
            base_coherence * chemical_stability * (thermal_stability/500) / (1+electron_phonon_coupling) + 
            s_electrons * 0.1 + 
            p_electrons * 0.2 + 
            f_electrons * 0.05 + 
            transition_metal_content * 0.3 +
            rare_earth_content * 0.2 + 
            main_group_content * 0.1 + 
            alkali_content * 0.1 + 
            halogen_content * 0.15 + 
            post_transition_metal_content * 0.1
        )

        return max(0.1, coherence_time)
    
    def calculate_gate_fidelity(self, material_properties):
        band_gap = self.calculate_band_gap(material_properties)
        berry_curvature = self.calculate_berry_curvature(material_properties)
        coherence_time = self.calculate_coherence_time(material_properties)

        spin_orbit_coupling = material_properties['spin_orbit_coupling']
        magnetic_anisotropy = material_properties['magnetic_anisotropy']
        s_electrons = material_properties['s_electron_count']
        p_electrons = material_properties['p_electron_count']
        d_electrons = material_properties['d_electron_count']
        f_electrons = material_properties['f_electron_count']
        transition_metal_content = material_properties['transition_metal_content']
        rare_earth_content = material_properties['rare_earth_content']
        main_group_content = material_properties['main_group_content']
        alkali_content = material_properties['alkali_metal_content']
        halogen_content = material_properties['halogen_content']
        post_transition_metal_content = material_properties['post_transition_metal_content']

        spin_orbit_coupling_factor = 1 - abs(spin_orbit_coupling - 1.0)
        gate_fidelity = (
            0.95 * (1-np.exp(-band_gap)) * np.tanh(coherence_time/50) * spin_orbit_coupling_factor +
            s_electrons * 0.1 + 
            p_electrons * 0.2 + 
            f_electrons * 0.05 + 
            transition_metal_content * 0.3 +
            rare_earth_content * 0.2 + 
            main_group_content * 0.1 + 
            alkali_content * 0.1 + 
            halogen_content * 0.15 + 
            post_transition_metal_content * 0.1
        )

        return max(0.5, min(0.999, gate_fidelity))
    
class MaterialSelector:
    def __init__(self, properties_dataframe):
        self.properties = properties_dataframe.copy()
        self.scaler = StandardScaler()
        self.calculate_composite_score()

    def calculate_composite_score(self):
        named_columns = ['band_gap', 'berry_curvature', 'coherence_time', 'gate_fidelity', 'thermal_stability', 'chemical_stability']
        normalized_properties = self.scaler.fit_transform(self.properties[named_columns])
        normalized_dataframe = pd.DataFrame(normalized_properties, columns=named_columns)

        performance_weights = {
            'band_gap': 0.2,
            'coherence_time': 0.3,
            'gate_fidelity': 0.4,
            'thermal_stability': 0.1
        }

        self.properties['performance_score'] = sum(performance_weights[column] * normalized_dataframe[column] for column in performance_weights.keys())
        self.properties['topological_score'] = np.abs(normalized_dataframe['berry_curvature'])
        self.properties['manufacturability_score'] = (-(self.properties['synthesis_difficulty'] + np.log(self.properties['cost_factor'])) / 2)
        self.properties['composite_score'] = (
            0.5 * self.properties['performance_score'] + 
            0.3 * self.properties['topological_score'] + 
            0.2 * self.properties['manufacturability_score']
        ) 

        self.calculate_property_space_coverage()

    def calculate_property_space_coverage(self):
        feature_columns = ['band_gap', 'berry_curvature', 'coherence_time', 'gate_fidelity', 'thermal_stability', 'chemical_stability']
        self.pca = PCA(n_components=2)
        pca_coordinates = self.pca.fit_transform(self.scaler.transform(self.properties[feature_columns]))

        self.properties['pca_1'] = pca_coordinates[:, 0]
        self.properties['pca_2'] = pca_coordinates[:, 1]

        distances = squareform(pdist(pca_coordinates))
        np.fill_diagonal(distances, np.inf)
        min_distances = np.min(distances, axis=1)
        self.properties['property_space_coverage'] = min_distances

    def select_diverse_materials(self, num_selected_materials, diversity_weight):
        diversity_scores = self.properties['property_space_coverage']
        composite_scores = self.properties['composite_score']

        normalized_diversity_scores = (diversity_scores - diversity_scores.min()) / (diversity_scores.max() - diversity_scores.min())
        normalized_composite_scores = (composite_scores - composite_scores.min()) / (composite_scores.max() - composite_scores.min())
        
        final_scores = (1 - diversity_weight) * normalized_composite_scores + diversity_weight * normalized_diversity_scores
        selected_indices = np.argsort(final_scores)[-num_selected_materials:]

        self.properties['selected'] = False
        self.properties.loc[selected_indices, 'selected'] = True
        self.properties['final_score'] = final_scores

        return self.properties[self.properties['selected']]
    
class VQE:
    def __init__(self, estimator, ansatz, maxiter=maxiter, optimizer_method='COBYLA'):
        self.estimator = estimator
        self.ansatz = ansatz
        self.maxiter = maxiter
        self.optimizer_method = optimizer_method
        self.evaluation_count = 0

    def objective_function(self, parameters, hamiltonian):
        self.evaluation_count += 1
        job = self.estimator.run(pubs=[(self.ansatz, hamiltonian, parameters)])
        result = job.result()
        expectation_value = float(result[0].data.evs)

        return expectation_value
    
    def compute_minimum_eigenvalue(self, hamiltonian, initial_point=None):
        start_time = time.time()
        self.evaluation_count = 0
        if initial_point is None:
            initial_point = np.random.random(self.ansatz.num_parameters) * 2 * np.pi

        result = minimize(
            fun=self.objective_function,
            x0=initial_point,
            args=(hamiltonian,),
            options={'maxiter': self.maxiter},
            method=self.optimizer_method
        )
        end_time = time.time()

        return {
            'eigenvalue': result.fun,
            'optimal_point': result.x,
            'result': result, 
            'function_evaluations': self.evaluation_count,
            'compute_time': end_time - start_time
        }

def evaluate_all_materials(materials_dataframe, property_calculator):
    properties = {
        'material_id': [],
        'band_gap': [],
        'berry_curvature': [],
        'coherence_time': [],
        'gate_fidelity': [],
        'synthesis_difficulty': [],
        'cost_factor': [],
        'thermal_stability': [],
        'chemical_stability': []
    }

    for idx, (index, material) in enumerate(materials_dataframe.materials.iterrows(), 1):
        band_gap = property_calculator.calculate_band_gap(material)
        berry_curvature = property_calculator.calculate_berry_curvature(material)
        coherence_time = property_calculator.calculate_coherence_time(material)
        gate_fidelity = property_calculator.calculate_gate_fidelity(material)
        
        properties['material_id'].append(material['material_id'])
        properties['band_gap'].append(band_gap)
        properties['berry_curvature'].append(berry_curvature)
        properties['coherence_time'].append(coherence_time)
        properties['gate_fidelity'].append(gate_fidelity)
        properties['synthesis_difficulty'].append(material['synthesis_difficulty'])
        properties['cost_factor'].append(material['cost_factor'])
        properties['thermal_stability'].append(material['thermal_stability'])
        properties['chemical_stability'].append(material['chemical_stability'])

    return pd.DataFrame(properties)

def create_materials_discovery_dashboard(selector):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    properties = selector.properties
    selected_mask = properties['selected']

    scatter_color = '#4a90e2'
    selected_color = '#e74c3c'
    
    # Plot 1: Final score vs. band gap
    x1 = properties['final_score']
    y1 = properties['band_gap']

    # Highlight selected materials
    ax1.scatter(x1[~selected_mask], y1[~selected_mask], 
                alpha=0.6, s=30, c=scatter_color, 
                label='Materials')
    scatter_selected = ax1.scatter(
        x1[selected_mask], y1[selected_mask],
        s=80, c=selected_color, marker='o',
        edgecolors='darkred', linewidth=1, 
        label='Selected'
    )

    # Annotate selected materials with their material_id
    for i in selected_mask[selected_mask].index:
        material_id = properties.loc[i, 'material_id']
        ax1.annotate(
            str(material_id),
            (x1[i], y1[i]),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center'
        )

    # Set up plot
    ax1.set_xlabel('Score Distribution', fontsize=16)
    ax1.set_ylabel('Band Gap (eV)', fontsize=16)
    ax1.set_title('Band Structure Properties', fontsize=18, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Property space coverage vs. Berry curvature
    x2 = properties['property_space_coverage']
    y2 = properties['berry_curvature']

    # Highlight selected materials
    ax2.scatter(x2[~selected_mask], y2[~selected_mask], 
                alpha=0.6, s=30, c=scatter_color, 
                label='Materials')
    scatter_selected = ax2.scatter(
        x2[selected_mask], y2[selected_mask],
        s=80, c=selected_color, marker='o',
        edgecolors='darkred', linewidth=1, 
        label='Selected'
    )

    # Annotate selected materials with their material_id
    for i in selected_mask[selected_mask].index:
        material_id = properties.loc[i, 'material_id']
        ax2.annotate(
            str(material_id),
            (x2[i], y2[i]),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center'
        )

    # Set up plot
    ax2.set_xlabel('Property Space Coverage', fontsize=16)
    ax2.set_ylabel('Berry Curvature', fontsize=16)
    ax2.set_title('Topological Properties', fontsize=18, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: 
    scores = properties['final_score']

    num_bins = 20
    counts, bins, patches = ax3.hist(scores, bins=num_bins,
                                     alpha=0.7, color=scatter_color,
                                     edgecolor='black', linewidth=0.5
                                    )
    # Set up plot
    ax3.set_xlabel('Score', fontsize=16)
    ax3.set_ylabel('Count', fontsize=16)
    ax3.set_title('Score Distribution', fontsize=18, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # Plot 4: 
    x4 = properties['pca_1']
    y4 = properties['pca_2']

    # Highlight selected materials
    ax4.scatter(
        x4[~selected_mask], y4[~selected_mask], 
        alpha=0.6, s=30, c=scatter_color,
        label='something'
    )
    scatter_selected = ax4.scatter(
        x4[selected_mask], y4[selected_mask],
        s=80, c=selected_color, marker='o',
        edgecolors='darkred', linewidth=1, 
        label='Selected'
    )

    # Annotate selected materials with their material_id
    for i in selected_mask[selected_mask].index:
        material_id = properties.loc[i, 'material_id']
        ax4.annotate(
            str(material_id),
            (x4[i], y4[i]),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center'
        )

    # Set up plot
    ax4.set_xlabel('First Principal Component', fontsize=16)
    ax4.set_ylabel('Second Principal Component', fontsize=16)
    ax4.set_title('Principal Component Analysis', fontsize=18, fontweight='bold')
    ax4.grid(True, alpha=0.3)

    fig.suptitle('Exploring for New Materials', fontsize=18, fontweight='bold', y=0.95)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)

    return fig

def analyze_discovery_results():
    properties = selector.properties
    selected = properties[properties['selected']]
    
    analysis_results = []
    analysis_results.append("Database statistics\n")
    analysis_results.append(f"  Total materials: {len(properties)}\n")
    analysis_results.append(f"  Materials selected: {len(selected)}\n")

    analysis_results.append("Performance Factor Ranges\n")
    analysis_results.append(f"  Band gap: {selected['band_gap'].min():.2f} - {selected['band_gap'].max():.2f} eV\n")
    analysis_results.append(f"  Berry curvature: {selected['berry_curvature'].min():.2f} - {selected['berry_curvature'].max():.2f}\n")
    analysis_results.append(f"  Coherence time: {selected['coherence_time'].min():.2f} - {selected['coherence_time'].max():.2f}\n")
    analysis_results.append(f"  Gate fidelity: {selected['gate_fidelity'].min():.2f} - {selected['gate_fidelity'].max():.2f}\n")

    analysis_results.append(f"Environmental Stability Ranges\n")
    analysis_results.append(f"  Thermal stability: {selected['thermal_stability'].min():.0f} - {selected['thermal_stability'].max():.0f}\n")
    analysis_results.append(f"  Chemical stability: {selected['chemical_stability'].min():.0f} - {selected['chemical_stability'].max():.0f}\n")
    
    analysis_results.append(f"Manufacturing Complexity Averages\n")
    analysis_results.append(f"  Average synthesis difficulty: {selected['synthesis_difficulty'].mean():.3f}\n")
    analysis_results.append(f"  Average cost factor: {selected['cost_factor'].mean():.2f}\n\n")

    analysis_results.append(f"Top {num_selected_materials} Materials\n")
    top_materials = selected.nlargest(num_selected_materials, 'final_score')
    for i, (idx, material) in enumerate(top_materials.iterrows(), 1):
        analysis_results.append(f"  #{i} Material ID {material['material_id']}\n")
        analysis_results.append(f"  Final score: {material['final_score']:.3f}\n")
        analysis_results.append(f"  Band gap: {material['band_gap']:.2f}\n")
        analysis_results.append(f"  Berry curvature: {material['berry_curvature']:.3f}\n")
        analysis_results.append(f"  Coherence time: {material['coherence_time']:.1f} μs\n")
        analysis_results.append(f"  Gate fidelity: {material['gate_fidelity']:.3f}\n")

    return ''.join(analysis_results), top_materials

def create_advanced_analysis_plots():
    fig, axes = plt.subplots(2, 3, figsize=(10, 12))
    properties = selector.properties
    selected_mask = properties['selected']

    # Plot 1: Performance vs. Manufacturability Trade-Off
    ax = axes[0, 0]
    scatter = ax.scatter(properties['manufacturability_score'], properties['performance_score'], 
                         alpha=0.6, s=30, c=properties['final_score'], cmap='viridis')
    ax.scatter(properties[selected_mask]['manufacturability_score'],
               properties[selected_mask]['performance_score'],
               s=80, c='red', marker='o', edgecolors='darkred', linewidth=1)
    ax.set_xlabel('Manufacturability Score', fontsize=16)
    ax.set_ylabel('Performance Score', fontsize=16)
    ax.set_title('Performance vs. Manufacturability Trade-Off', fontsize=18)
    plt.colorbar(scatter, ax=ax, label='Final Score')
    ax.grid(True, alpha=0.3)

    # Annotate selected materials with their material_id
    for i in selected_mask[selected_mask].index:
        material_id = properties.loc[i, 'material_id']
        ax.annotate(
            str(material_id),
            (properties.loc[i, 'manufacturability_score'], properties.loc[i, 'performance_score']),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center'
        )

    # Plot 2: Coherence Time vs. Gate Fidelity
    ax = axes[0, 1]
    scatter = ax.scatter(properties['coherence_time'], properties['gate_fidelity'], 
                         alpha=0.6, s=30, c='lightblue')
    ax.scatter(properties[selected_mask]['coherence_time'],
               properties[selected_mask]['gate_fidelity'],
               s=80, c='red', marker='o', edgecolors='darkred', linewidth=1)
    ax.set_xlabel('Coherence Time (μs)', fontsize=16)
    ax.set_ylabel('Gate Fidelity', fontsize=16)
    ax.set_title('Coherence Time vs. Gate Fidelity', fontsize=18)
    ax.grid(True, alpha=0.3)

    # Annotate selected materials with their material_id
    for i in selected_mask[selected_mask].index:
        material_id = properties.loc[i, 'material_id']
        ax.annotate(
            str(material_id),
            (properties.loc[i, 'coherence_time'], properties.loc[i, 'gate_fidelity']),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center'
        )

    # Plot 3: Thermal vs. Chemical Stability
    ax = axes[0, 2]
    scatter = ax.scatter(properties['thermal_stability'], properties['chemical_stability'], 
                         alpha=0.6, s=30, c='lightgreen')
    ax.scatter(properties[selected_mask]['thermal_stability'],
               properties[selected_mask]['chemical_stability'],
               s=80, c='red', marker='o', edgecolors='darkred', linewidth=1)
    ax.set_xlabel('Thermal Stability (K)', fontsize=16)
    ax.set_ylabel('Chemical Stability', fontsize=16)
    ax.set_title('Thermal vs. Chemical Stability', fontsize=18)
    ax.grid(True, alpha=0.3)

    # Annotate selected materials with their material_id
    for i in selected_mask[selected_mask].index:
        material_id = properties.loc[i, 'material_id']
        ax.annotate(
            str(material_id),
            (properties.loc[i, 'thermal_stability'], properties.loc[i, 'chemical_stability']),
            textcoords='offset points',
            xytext=(0, 10),
            ha='center'
        )

    # Plot 4: Score Components Comparison
    ax = axes[1, 0]
    selected_data = properties[selected_mask]
    scores = ['performance_score', 'topological_score', 'manufacturability_score']
    avg_scores = [selected_data[score].mean() for score in scores]
    bars = ax.bar(range(len(scores)), avg_scores, color=['blue', 'green', 'orange'], alpha=0.7)
    ax.set_xticks(range(len(scores)))
    ax.set_xticklabels(['Performance', 'Topological', 'Manufacturability'], rotation=45)
    ax.set_ylabel('Average Score')
    ax.set_title('Score Components of Selected Materials')
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar, score in zip(bars, avg_scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom')
        
    # Plot 5: Property Distribution Comparison
    ax = axes[1, 1]
    properties_to_compare = ['band_gap', 'berry_curvature', 'coherence_time', 'gate_fidelity']

    all_data = [properties[prop] for prop in properties_to_compare]
    selected_data_list = [properties[selected_mask][prop] for prop in properties_to_compare]

    positions = np.arange(len(properties_to_compare))
    bp1 = ax.boxplot(all_data, positions=positions-0.2, widths=0.3,
                     patch_artist=True, boxprops=dict(facecolor='lightblue', alpha=0.7))
    bp2 = ax.boxplot(selected_data_list, positions=positions+0.2, widths=0.3,
                     patch_artist=True, boxprops=dict(facecolor='red', alpha=0.7))
    ax.set_xticks(positions)
    ax.set_xticklabels(['Band Gap', 'Berry Curvature', 'Coherence Time', 'Gate Fidelity'], rotation=45)
    ax.set_ylabel('Normalized Values')
    ax.set_title('Property Distribution: All vs. Selected')
    ax.legend([bp1['boxes'][0], bp2['boxes'][0]], ['All Materials', 'Selected'], loc='upper right')
    ax.grid(True, alpha=0.3)

    # Plot 6: Selection Efficiency Analysis
    ax = axes[1, 2]
    num_bins = 10
    bin_edges = np.linspace(properties['final_score'].min(), properties['final_score'].max(), num_bins+1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    selected_rates = []
    for i in range(num_bins):
        mask = (properties['final_score'] >= bin_edges[i]) & (properties['final_score'] < bin_edges[i+1])
        if i == num_bins-1:
            mask = (properties['final_score'] >= bin_edges[i]) & (properties['final_score'] <= bin_edges[i+1])

        total_in_bin = mask.sum()
        selected_in_bin = (mask & selected_mask).sum()
        rate = selected_in_bin / total_in_bin if total_in_bin > 0 else 0
        selected_rates.append(rate)

    bars = ax.bar(bin_centers, selected_rates, width=(bin_edges[1]-bin_edges[0])*0.8,
                  alpha=0.7, color='purple')
    ax.set_xlabel('Final Score Bins')
    ax.set_ylabel('Selection Rate')
    ax.set_title('Selection Efficiency by Score Range')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig
    
def export_materials_recommendations():
    selected = selector.properties[selector.properties['selected']].copy()

    report_dataframe = selected[['material_id', 'final_score', 'band_gap', 'berry_curvature', 'coherence_time', 'gate_fidelity', 'thermal_stability', 'cost_factor']].copy()
    report_dataframe.columns = ["Material ID", "Final Score", "Band Gap (eV)", "Berry Curvature", "Coherence Time (μs)", "Gate Fidelity", "Thermal Stability (K)", "Cost Factor"]
    
    pd.set_option('display.colheader_justify', 'center')

    report_output = "Quantum Materials Discovery\n"
    report_output += report_dataframe.to_string(index=False, justify='center')

    return report_output

def get_kitaev_hamiltonian(mu, t, delta, num_qubits):
    pauli_terms = []

    # Chemical potential terms
    for qubit in range(num_qubits):
                      pauli_str = ['I'] * num_qubits
                      pauli_str[qubit] = 'Z'
                      pauli_terms.append((''.join(pauli_str), -mu/2))

    # Hopping terms
    for qubit in range(num_qubits-1):
        # X_qubit X_{qubit+1} term
        pauli_str = ['I'] * num_qubits
        pauli_str[qubit] = 'X'
        pauli_str[qubit+1] = 'X'
        pauli_terms.append((''.join(pauli_str), -t/2))

        # Y_qubit Y_{qubit+1} term
        pauli_str = ['I'] * num_qubits
        pauli_str[qubit] = 'Y'
        pauli_str[qubit+1] = 'Y'
        pauli_terms.append((''.join(pauli_str), -t/2))

    # Pairing terms
    for qubit in range(num_qubits-1):
        # X_qubit Y_{qubit+1} term
        pauli_str = ['I'] * num_qubits
        pauli_str[qubit] = 'X'
        pauli_str[qubit+1] = 'Y'
        pauli_terms.append((''.join(pauli_str), delta/2))

        # -Y_qubit X_{qubit+1} term
        pauli_str = ['I'] * num_qubits
        pauli_str[qubit] = 'Y'  
        pauli_str[qubit+1] = 'X'
        pauli_terms.append((''.join(pauli_str), -delta/2))

    return SparsePauliOp.from_list(pauli_terms)

def calculate_exact_ground_state(hamiltonian):
    # Converts to dense matrix and finds eigenvalues
    matrix = hamiltonian.to_matrix()
    eigenvalues = np.linalg.eigvals(matrix)

    return np.real(eigenvalues)

def calculate_time(total_seconds):
    total_time = sum(total_seconds)
    minutes, seconds = divmod(total_time, 60)

    return f'{int(minutes)} mins {seconds:.0f} s'

def setup_backend():
    backend = AerSimulator(shots=shots, seed_simulator=seed)
    estimator = BackendEstimatorV2(backend=backend)

    return backend, estimator

def initialize_vqe(backend, estimator):
    ansatz = RealAmplitudes(
        num_qubits=num_qubits,
        reps=reps,
        entanglement=entanglement,
        insert_barriers=True,
        flatten=True
    )
    transpiled_ansatz = transpile(ansatz, backend=backend, optimization_level=optimization_level)
    vqe = VQE(estimator=estimator, ansatz=transpiled_ansatz, optimizer_method='COBYLA', maxiter=maxiter)

    return vqe

def run_vqe_for_selected_materials(vqe, mu_values, t_values, delta_values, selected_materials):
    all_results = []
    for i, material in selected_materials.iterrows():
        material_output = []
        vqe_energies = []
        exact_energies = []
        optimal_parameters = []
        total_seconds = []
        parameter_combinations = []
        total_combinations = len(mu_values) * len(t_values) * len(delta_values)
        combination_counter = 1

        material_output.append(f"Running VQE for material #{material['material_id']} with properties:")
        for delta in delta_values:
            for t in t_values:
                for mu in mu_values:
                    material_output.append(f"μ = {mu:.2f}, t = {t:.2f}, delta = {delta:.2f}, run {combination_counter}/{total_combinations}")

                    hamiltonian = get_kitaev_hamiltonian(mu, t, delta, num_qubits)
                    exact_energy = np.min(calculate_exact_ground_state(hamiltonian))
                    exact_energies.append(exact_energy)
                    initial_point = optimal_parameters[-1] if optimal_parameters else None
                    vqe_result = vqe.compute_minimum_eigenvalue(hamiltonian, initial_point)
                    vqe_energies.append(vqe_result['eigenvalue'])
                    optimal_parameters.append(vqe_result['optimal_point'])
                    total_seconds.append(vqe_result['compute_time'])
                    combination_counter += 1

                    material_output.append(f"  VQE Energy: {vqe_result['eigenvalue']:.4f}")
                    material_output.append(f"  Exact Energy: {exact_energy:.4f}")
                    material_output.append(f"  Error: {abs(vqe_result['eigenvalue'] - exact_energy):.6f}")
                    material_output.append(f"  Function Evaluations: {vqe_result['function_evaluations']}")
                    material_output.append(f"  Vqe Time: {vqe_result['compute_time']:.2f} s\n")

                    parameter_combinations.append((mu, t, delta))

        all_results.append({'material_id': material['material_id'], 
                            'vqe_energies': np.array(vqe_energies),
                            'exact_energies': np.array(exact_energies),
                            'optimal_parameters': np.array(optimal_parameters),
                            'total_seconds': np.array(total_seconds),
                            'parameter_combinations': parameter_combinations,
                            'output': material_output
                            })
        
    return all_results

def plot_result_analysis_by_parameter(parameter_name, parameter_values, vqe_energies, exact_energies, optimal_parameters, errors, energy_gaps):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)

    # Plot 1: Energy comparison
    unique_parameters = set((t, delta) for mu, t, delta in parameter_combinations)
    for parameter_set in unique_parameters:
        indices = [i for i, (pmu, pt, pdelta) in enumerate(parameter_combinations) if (pt, pdelta) == parameter_set]
        label = f"t={parameter_set[0]:.1f}, delta = {parameter_set[1]:.1f}"
        axes[0, 0].plot([parameter_values[i] for i in indices],
                        [vqe_energies[i] for i in indices],
                         'o-', linewidth=2, markersize=6, label=label)
        axes[0, 0].plot([parameter_values[i] for i in indices],
                        [exact_energies[i] for i in indices],
                         's-', linewidth=2, markersize=4)
        
    axes[0, 0].set_xlabel(parameter_name)
    axes[0, 0].set_ylabel("Ground State Energy")
    axes[0, 0].set_title("VQE vs. Exact Ground State Energy")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: VQE errors
    for parameter_set in unique_parameters:
        indices = [i for i, (pmu, pt, pdelta) in enumerate(parameter_combinations) if (pt, pdelta) == parameter_set]
        axes[0, 1].semilogy([parameter_values[i] for i in indices], 
                            [errors[i] for i in indices], 
                            'ro-', linewidth=2, markersize=6)
        
    axes[0, 1].set_xlabel(parameter_name)
    axes[0, 1].set_ylabel("|VQE Error| (log scale)")
    axes[0, 1].set_title("VQE Accuracy vs. " + parameter_name)
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Optimal parameter evaluation
    for i in range(optimal_parameters.shape[1]):
        for parameter_set in unique_parameters:
            indices = [j for j, (pmu, pt, pdelta) in enumerate(parameter_combinations) if (pt, pdelta) == parameter_set]
            axes[1, 0].plot([parameter_values[j] for j in indices],
                        [optimal_parameters[j, i] for j in indices],
                         'o-', linewidth=2, markersize=4, label=f"{parameter_name} Parameter {i+1}")
            
    axes[1, 0].set_xlabel(parameter_name)
    axes[1, 0].set_ylabel("Optimal Parameter Value")
    axes[1, 0].set_title("Optimal VQE Parameters vs. " + parameter_name)
    axes[1, 0].legend(loc='upper right')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Energy gap analysis
    for parameter_set in unique_parameters:
        indices = [i for i, (pmu, pt, pdelta) in enumerate(parameter_combinations) if (pt, pdelta) == parameter_set]
        axes[1, 1].plot([parameter_values[i] for i in indices], 
                            [energy_gaps[i] for i in indices], 
                            'ro-', linewidth=2, markersize=6)
        
    axes[1, 1].set_xlabel(parameter_name)
    axes[1, 1].set_ylabel("Energy Gap")
    axes[1, 1].set_title("Energy Gap vs. " + parameter_name)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    return fig

def analyze_results(vqe_energies, exact_energies, optimal_parameters, parameter_combinations, total_seconds):
    errors = np.abs(vqe_energies - exact_energies)
    # Calculate energy gaps - pad with zeros to match array length
    if len(exact_energies) > 1:
        energy_gaps = np.diff(exact_energies)
        energy_gaps = np.append(energy_gaps, 0)  # Add one element to match length
    else:
        energy_gaps = np.array([0])
    output = []
    
    parameter_values_mu = [parameter_combo[0] for parameter_combo in parameter_combinations]
    fig_mu = plot_result_analysis_by_parameter("Chemical Potential (μ)", parameter_values_mu, vqe_energies, exact_energies, optimal_parameters, errors, energy_gaps)

    parameter_values_t = [parameter_combo[1] for parameter_combo in parameter_combinations]
    fig_t = plot_result_analysis_by_parameter("Hopping Parameter (t)", parameter_values_t, vqe_energies, exact_energies, optimal_parameters, errors, energy_gaps)

    parameter_values_delta = [parameter_combo[2] for parameter_combo in parameter_combinations]
    fig_delta = plot_result_analysis_by_parameter("Pairing Potential (Δ)", parameter_values_delta, vqe_energies, exact_energies, optimal_parameters, errors, energy_gaps)

    output.append("\nVQE Performance Statistics:")
    output.append(f"  Mean absolute error: {np.mean(errors):.6f}")
    output.append(f"  Max absolute error: {np.max(errors):.6f}")
    output.append(f"  Min absolute error: {np.min(errors):.6f}")
    output.append(f"  Std deviation of errors: {np.std(errors):.6f}")

    return vqe_energies, exact_energies, optimal_parameters, errors, energy_gaps, output, fig_mu, fig_t, fig_delta

def identify_top_candidates(vqe_energies, exact_energies, optimal_parameters, parameter_combinations, errors, energy_gaps, total_seconds):
    output = []
    top_candidates = []

    def is_topological(mu):
        return -2 <= mu <= 2
    
    def score_candidate(i):
        energy_score = -vqe_energies[i]
        accuracy_score = -errors[i]
        gap_score = energy_gaps[i]

        return energy_score + 10 * accuracy_score + gap_score
    
    for i, scores in enumerate(zip(parameter_combinations, vqe_energies, exact_energies, errors, energy_gaps, optimal_parameters)):
        mu, t, delta = parameter_combinations[i]
        if is_topological(mu):
            candidate_score = score_candidate(i)
            if len(top_candidates) < 5 or candidate_score > min([candidate[1] for candidate in top_candidates]):
                if len(top_candidates) ==5:
                    top_candidates.sort(key=lambda x: x[1])
                    top_candidates.pop(0)
                top_candidates.append((i, candidate_score, mu, t, delta, scores[1:]))
    
    top_candidates.sort(key=lambda x: x[1], reverse=True)
    
    output.append("\n\nTop 5 Candidate Materials:")
    for rank, candidate in enumerate(top_candidates, 1):
        i, score, mu, t, delta, other_scores = candidate
        vqe_energy, exact_energy, error, energy_gap, optimal_params = other_scores
        output.append(f"  Candidate {rank} (Rank Score: {score:.3f}):")
        output.append(f"    μ = {mu:.2f}, t = {t:.2f}, Δ = {delta:.2f}):")
        output.append(f"    VQE Ground State Energy = {vqe_energy:.4f}")
        output.append(f"    Exact Ground State Energy = {exact_energy:.4f}")
        output.append(f"    VQE Error = {error:.6f}")
        output.append(f"    Energy Gap = {energy_gap:.4f}")
        output.append(f"    Optimal Parameters = {[f'{param:.3f}' for param in optimal_params]}\n")

    plt.figure(figsize=(12, 8))

    # Main energy plots for each parameter combination
    unique_parameters = set((t, delta) for mu, t, delta in parameter_combinations)
    for parameter_set in unique_parameters:
        indices = [i for i, (pmu, pt, pdelta) in enumerate(parameter_combinations) if (pt, pdelta) == parameter_set]
        plt.plot([parameter_combinations[i][2] for i in indices], [vqe_energies[i] for i in indices], 'o-', linewidth=2, markersize=6, color='blue', label="VQE Ground State Energy")
        plt.plot([parameter_combinations[i][2] for i in indices], [exact_energies[i] for i in indices], 's--', linewidth=2, markersize=4, color='green', label="Exact Ground State Energy")

    for rank, candidate in enumerate(top_candidates, 1):
        i = candidate[0]
        plt.plot(parameter_combinations[i][2], vqe_energies[i], 'o', markersize=15, markerfacecolor='red', markeredgecolor='darkred', markeredgewidth=2, alpha=0.8)
        plt.annotate(f'#{rank}', (parameter_combinations[i][2], vqe_energies[i]), xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold', color='darkred')

        plt.axvline(-2, color='gray', linestyle=':', alpha=0.7, label="Topological boundary")
        plt.axvline(2, color='gray', linestyle=':', alpha=0.7)
        plt.axvspan(-2, 2, alpha=0.1, color='yellow', label="Topological regime")

        plt.xlabel("Chemical Potential (μ)", fontsize=16)
        plt.ylabel("Ground State Energy (eV)", fontsize=16)
        plt.title("Kitaev Chain Ground State Energy with Top Material Candidates", fontsize=18, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        output.append("Analysis Summary:")
        output.append(f"  Total VQE time: {calculate_time(total_seconds)}")
        output.append(f"  Simulated: {len(mu_values)} parameter points")
        output.append(f"  Average VQE accuracy: {np.mean(errors):.6f}")

    return output, plt.gcf()

if __name__ == "__main__":
    # Generate material database
    materials_database = QuantumMaterialDatabase(num_materials)
    pd.set_option('display.colheader_justify', 'center')
    materials_database_path = os.path.join(script_dir, "All Materials.txt")
    with open(materials_database_path, 'w', encoding='utf-8') as file:
        file.write("Sample material properties:\n")
        file.write(materials_database.materials.head(num_materials).to_string(index=False))

    # Calculate properties for all materials
    property_calculator = QuantumPropertyCalculator()
    material_properties = evaluate_all_materials(materials_database, property_calculator)
    with open(materials_database_path, 'a', encoding='utf-8') as file:
        file.write("\n\nSample calculated properties:\n")
        file.write(material_properties.head(num_materials).to_string(index=False))

    # Initialize selector and perform selection
    selector = MaterialSelector(material_properties)
    selected_materials = selector.select_diverse_materials(num_selected_materials, diversity_weight)
    with open(materials_database_path, 'a', encoding='utf-8') as file:  
        file.write(f"\n\nSelected {len(selected_materials)} materials:\n")
        file.write(selected_materials[['material_id', 'performance_score', 'topological_score', 'composite_score', 'final_score']].round(3).to_string(index=False))

    # Create and save dashboard
    dashboard_fig = create_materials_discovery_dashboard(selector)
    dashboard_fig_path = os.path.join(script_dir, "Material Property Plots.png")
    dashboard_fig.savefig(dashboard_fig_path)
    discovery_results, top_materials = analyze_discovery_results()
    with open(materials_database_path, 'a', encoding='utf-8') as file:  
        file.write("\n\n" + discovery_results)

    # Create and show analysis plots
    advanced_fig = create_advanced_analysis_plots()
    advanced_fig_path = os.path.join(script_dir, "Advanced Analysis Plots.png")
    advanced_fig.savefig(advanced_fig_path)
    with open(materials_database_path, 'a', encoding='utf-8') as file:
        file.write("\n\n" + export_materials_recommendations())

    # Initialize and setup backend for VQE
    backend, estimator = setup_backend()
    vqe = initialize_vqe(backend, estimator)

    # Select the top materials for VQE
    top_materials = top_materials.copy()

    # Run VQE for top materials
    vqe_results_for_selected_materials = run_vqe_for_selected_materials(vqe, mu_values, t_values, delta_values, top_materials)

    # Process results for each material
    for result in vqe_results_for_selected_materials:
        vqe_energies = result['vqe_energies']
        exact_energies = result['exact_energies']
        optimal_parameters = result['optimal_parameters']
        parameter_combinations = result['parameter_combinations']
        total_seconds = result['total_seconds']
        material_id = result['material_id']
        material_output = result['output']

        # Print or save the output for each material
        material_output_path = os.path.join(script_dir, f"Mat. {material_id} VQE Material Output.txt")
        with open(material_output_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join(material_output))
        
        # Unpack the VQE outcomes and analyze the results
        vqe_energies, exact_energies, optimal_parameters, errors, energy_gaps, output_list, fig_mu, fig_t, fig_delta = analyze_results(vqe_energies, exact_energies, optimal_parameters, parameter_combinations, total_seconds)
        
        # Save the figures for each parameter
        energy_gap_fig_mu_path = os.path.join(script_dir, f"Mat. {material_id} Energy Gap vs. mu.png")
        fig_mu.savefig(energy_gap_fig_mu_path)

        energy_gap_fig_t_path = os.path.join(script_dir, f"Mat. {material_id} Energy Gap vs. t.png")
        fig_t.savefig(energy_gap_fig_t_path)

        energy_gap_fig_delta_path = os.path.join(script_dir, f"Mat. {material_id} Energy Gap vs. delta.png")
        fig_delta.savefig(energy_gap_fig_delta_path)

        # Identify top candidates for each material
        top_candidates_output, top_candidates_fig = identify_top_candidates(vqe_energies, exact_energies, optimal_parameters, parameter_combinations, errors, energy_gaps, total_seconds)

        # Save the top candidates figure for each material
        top_candidates_fig_path = os.path.join(script_dir, f"Mat. {material_id} Top Material Candidates.png")
        top_candidates_fig.savefig(top_candidates_fig_path)

        # Add top candidates output to the VQE outcomes file for each material
        with open(material_output_path, 'a', encoding='utf-8') as file:
            file.write('\n'.join(output_list))
            file.write('\n'.join(top_candidates_output))