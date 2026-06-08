from typing import Callable, Tuple
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler
from qiskit.quantum_info import Statevector
import jax
import jax.numpy as jnp
import flax.linen as nn


def angle_embedding(qc: QuantumCircuit, inputs):
    """Encodes input data into quantum states using RY rotations."""
    num_qubits = inputs.shape[-1]
    for j in range(num_qubits):
        qc.ry(inputs[j], j)

def phase_embedding(qc: QuantumCircuit, inputs):
    num_qubits = inputs.shape[-1]
    for j in range(num_qubits):
        qc.h(j)
        qc.rz(inputs[j], j)
        qc.h(j)

def amplitude_embedding(qc: QuantumCircuit, inputs):
    num_qubits = inputs.shape[-1]
    # Normalize inputs to unit norm
    norm = jnp.linalg.norm(inputs)
    normalized = inputs / norm
    qc.initialize(normalized, range(num_qubits))

def iqp_embedding(qc: QuantumCircuit, inputs):
    n = inputs.shape[-1]
    angles = [(x / 10) * jnp.pi for x in inputs]  # Assuming inputs are in [0, 10]
    for i in range(n):
        qc.h(i)
    for i in range(n):
        qc.rz(angles[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            qc.cx(i, j)
            qc.rz(angles[i] * angles[j], j)
            qc.cx(i, j)


# VQC Design

def basic_vqc(qc: QuantumCircuit, inputs, weights):
    """
    Standard VQC with RX rotations and CNOT entanglement.
    Weights shape: (num_layers, num_qubits)
    """
    num_qubits = inputs.shape[-1]
    num_qlayers = weights.shape[-2]

    for i in range(num_qlayers):
        for j in range(num_qubits):
            qc.rx(weights[i, j], j)
        if num_qubits == 2:
            qc.cnot(0, 1)
        elif num_qubits > 2:
            for j in range(num_qubits):
                qc.cnot(j, (j + 1) % num_qubits)

def get_quantum_layer_circuit(inputs, weights,
                              embedding: Callable = angle_embedding, vqc: Callable = basic_vqc):
    """
    Constructs the TensorCircuit object using the provided embedding and VQC functions.
    """
    num_qubits = inputs.shape[-1]
    qc = QuantumCircuit(num_qubits)
    embedding(qc, inputs)
    vqc(qc, inputs, weights)
    return qc


def get_circuit(embedding: Callable = angle_embedding, vqc: Callable = basic_vqc,
                torch_interface: bool = False):
    """
    Returns a vectorized JAX function that executes the quantum circuit.
    """
    def qpred(inputs, weights):
        qc = get_quantum_layer_circuit(inputs, weights, embedding, vqc)
        # We measure the expectation of Z on all qubits (conceptually similar to measurement)
        # If the weight shape is complex (e.g. Design 3), we need to ensure we map output correctly.
        # Here we just return Z expectation on all qubits, which matches input dimension (num_qubits).
        # We assume weights.shape[1] is NOT used for determining output size, 
        # but inputs.shape[-1] (num_qubits) is the output size.
        num_qubits = inputs.shape[-1]
        return jnp.real(jnp.array([qc.expectation_ps(z=[i]) for i in range(num_qubits)]))

    qpred_batch = jax.vmap(qpred, in_axes=(0, None))
    if torch_interface:
        qpred_batch = jax.interfaces.torch_interface(qpred_batch, jit=True)

    return qpred_batch


class QuantumLayer(nn.Module):
    """
    A Flax Linen Module wrapping the quantum circuit.
    """
    num_qubits: int
    w_shape: tuple = (1,) # Shape of weights (e.g., (layers,) or (layers, 3))
    circuit: Callable = get_circuit()

    @nn.compact
    def __call__(self, x):
        shape = x.shape
        # Flatten input to (batch * seq_len, hidden_size)
        x = jnp.reshape(x, (-1, shape[-1]))
        
        # Initialize weights. 
        # The final weight shape will be self.w_shape + (self.num_qubits,)
        # e.g., if w_shape is (2, 3), weights will be (2, 3, num_qubits)
        w = self.param('w', nn.initializers.xavier_normal(), self.w_shape + (self.num_qubits,))
        
        # Execute circuit
        x = self.circuit(x, w)
        
        # Reshape back to original dimensions
        # x comes out as (batch * seq_len, num_qubits)
        # We assume num_qubits == hidden_size
        x = jnp.reshape(x, tuple(shape))
        return x