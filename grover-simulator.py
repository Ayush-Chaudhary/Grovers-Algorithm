import matplotlib.pyplot as plt
import numpy as np
import math
import argparse
from qiskit_aer import AerSimulator
from qiskit_aer import AerSimulator, noise
from qiskit import QuantumCircuit
from qiskit.circuit.library import GroverOperator, MCMT, ZGate, XGate
from qiskit import transpile
from qiskit.quantum_info import Kraus

# Define the function to build a Grover oracle
def grover_oracle(marked_states):
    """Build a Grover oracle for multiple marked states."""
    if not isinstance(marked_states, list):
        marked_states = [marked_states]
    num_qubits = len(marked_states[0])
    qc = QuantumCircuit(num_qubits)
    for target in marked_states:
        rev_target = target[::-1]
        zero_inds = [ind for ind in range(num_qubits) if rev_target.startswith("0", ind)]
        if zero_inds:
            qc.x(zero_inds)
            qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
            qc.x(zero_inds)
        else:
            qc.compose(MCMT(ZGate(), num_qubits - 1, 1), inplace=True)
    return qc

def noise_func(noise_type):
    # Create a noise model based on the specified type
    if noise_type == 'depolarizing':
        noise_model = noise.NoiseModel()
        error_1 = noise.depolarizing_error(0.01, 1)  # 1% error for single-qubit gates
        error_2 = noise.depolarizing_error(0.02, 2)  # 2% error for two-qubit gates
        noise_model.add_all_qubit_quantum_error(error_1, ['u1', 'u2', 'u3', 'h', 'x', 'y', 'z'])
        noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'swap'])
    elif noise_type == 'bit-flip':
        noise_model = noise.NoiseModel()
        # Define a 1-qubit bit-flip error
        p_flip = 0.1  # Probability of bit-flip
        bit_flip_kraus = [np.sqrt(1 - p_flip) * np.eye(2), np.sqrt(p_flip) * XGate().to_matrix()]
        error_1 = noise.QuantumError(Kraus(bit_flip_kraus))
        
        # Define a 2-qubit bit-flip error (for demonstration, using independent bit-flips on each qubit)
        bit_flip_kraus_2 = [np.sqrt(1 - p_flip) * np.eye(4), np.sqrt(p_flip) * np.kron(XGate().to_matrix(), XGate().to_matrix())]
        error_2 = noise.QuantumError(Kraus(bit_flip_kraus_2))
        
        # Add errors to the noise model
        noise_model.add_all_qubit_quantum_error(error_1, ['x', 'h'])  # Apply to 1-qubit gates
        noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'swap'])  # Apply to 2-qubit gates
    elif noise_type == 'phase-flip':
        noise_model = noise.NoiseModel()
        # Define a 1-qubit phase-flip error
        p_flip = 0.1  # Probability of phase-flip
        phase_flip_kraus = [np.sqrt(1 - p_flip) * np.eye(2), np.sqrt(p_flip) * ZGate().to_matrix()]
        error_1 = noise.QuantumError(Kraus(phase_flip_kraus))
        
        # Define a 2-qubit phase-flip error (for demonstration, using independent phase-flips on each qubit)
        phase_flip_kraus_2 = [np.sqrt(1 - p_flip) * np.eye(4), np.sqrt(p_flip) * np.kron(ZGate().to_matrix(), ZGate().to_matrix())]
        error_2 = noise.QuantumError(Kraus(phase_flip_kraus_2))
        
        # Add errors to the noise model
        noise_model.add_all_qubit_quantum_error(error_1, ['z', 'h'])  # Apply to 1-qubit gates
        noise_model.add_all_qubit_quantum_error(error_2, ['cx', 'cz', 'swap'])  # Apply to 2-qubit gates
    else:
        raise ValueError("Invalid noise type specified.")
    return noise_model

def quantum_circuit(marked_states):
    oracle = grover_oracle(marked_states)
    grover_op = GroverOperator(oracle)

    optimal_num_iterations = math.floor(
        math.pi / (4 * math.asin(math.sqrt(len(marked_states) / 2**grover_op.num_qubits)))
    )

    # Create the quantum circuit for Grover's search algorithm
    qc = QuantumCircuit(grover_op.num_qubits)
    qc.h(range(grover_op.num_qubits))
    qc.compose(grover_op.power(optimal_num_iterations), inplace=True)
    qc.measure_all()

    return qc

def grover(marked_states, noise_type):
    # Create the quantum circuit for Grover's search algorithm
    qc = quantum_circuit(marked_states)

    # # Display the quantum circuit
    # qc.draw(output="mpl", style="iqp")

    if noise_type == 'none':
        simulator = AerSimulator()
    else:
    # Create a noise model
        noise_model = noise_func(noise_type)

        # Use AerSimulator with noise
        simulator = AerSimulator(noise_model=noise_model)

    # Transpile the circuit for the simulator
    transpiled_qc = transpile(qc, simulator)

    # Run the simulation
    job = simulator.run(transpiled_qc, shots=10000)
    result = job.result()
    counts = result.get_counts()

    # Plot the results with noise
    plt.bar(counts.keys(), counts.values())
    plt.xlabel('States')
    plt.ylabel('Counts')
    plt.title('Measurement Results')
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Grover's Algorithm Simulator")
    parser.add_argument(
        '--marked-states', 
        type=str, 
        required=True, 
        help='Comma-separated list of marked states for Grover\'s algorithm'
    )
    parser.add_argument(
        "--noise-model",
        type = str,
        default= 'none',
        help='choose a noise type from depolarizing, phase-flip and bit-flip'     
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    marked_states = [str(state) for state in args.marked_states.split(',')]
    print(marked_states)
    grover(marked_states, args.noise_model)