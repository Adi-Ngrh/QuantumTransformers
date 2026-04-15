import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer.primitives import Sampler
from qiskit.quantum_info import Statevector


def angle_encoding(number: int, min_val: float = 0, max_val: float = 10) -> np.ndarray:
    angle = ((number - min_val) / (max_val - min_val)) * np.pi
    qc = QuantumCircuit(1)
    qc.ry(angle, 0)
    sv = Statevector.from_instruction(qc)
    return sv.data


def phase_encoding(number: int, min_val: float = 0, max_val: float = 10) -> np.ndarray:
    angle = ((number - min_val) / (max_val - min_val)) * np.pi
    qc = QuantumCircuit(1)
    qc.h(0)
    qc.rz(angle, 0)
    qc.h(0)
    sv = Statevector.from_instruction(qc)
    return sv.data


def amplitude_encoding(data: list) -> np.ndarray:
    n = len(data)
    num_qubits = int(np.ceil(np.log2(n)))
    size = 2 ** num_qubits
    padded = np.array(data, dtype=float)
    padded = np.pad(padded, (0, size - n))
    norm = np.linalg.norm(padded)
    normalized = padded / norm
    qc = QuantumCircuit(num_qubits)
    qc.initialize(normalized, range(num_qubits))
    sv = Statevector.from_instruction(qc)
    return sv.data


def iqp_encoding(data: list, min_val: float = 0, max_val: float = 10) -> np.ndarray:
    n = len(data)
    angles = [(x - min_val) / (max_val - min_val) * np.pi for x in data]
    qc = QuantumCircuit(n)
    for i in range(n):
        qc.h(i)
    for i in range(n):
        qc.rz(angles[i], i)
    for i in range(n):
        for j in range(i + 1, n):
            qc.cx(i, j)
            qc.rz(angles[i] * angles[j], j)
            qc.cx(i, j)
    sv = Statevector.from_instruction(qc)
    return sv.data


def main():
    numbers = [0,1,2,3,4,5,6,7,8,9,10]

    print("=== Angle Encoding ===")
    coefficients = angle_encoding(numbers[4])
    print(coefficients)
    recovered_angle_val = 2 * np.arcsin(np.abs(coefficients[1]))
    reconstructed_number = (recovered_angle_val / np.pi) * (10 - 0) + 0
    print(f"Reconstructed number: {reconstructed_number:.6f}")

    print("\n=== Phase Encoding ===")
    phase_coeffs = phase_encoding(numbers[4])
    print(phase_coeffs)
    probability_1 = np.abs(phase_coeffs[1]) ** 2
    recovered_angle = 2 * np.arcsin(np.sqrt(probability_1))
    print(f"P(|1>) = : {probability_1:.6f}")
    print(f"Recovered angle : {recovered_angle:.6f} rad")

    print("\n=== Amplitude Encoding ===")
    amp_coeffs = amplitude_encoding(numbers)
    norm = np.linalg.norm(np.array(numbers, dtype=float))
    print(f"{'Value':>6}  {'Amplitude':>12}  {'Reconstructed':>13}")
    print("-" * 36)
    for i, num in enumerate(numbers):
        amp = amp_coeffs[i]
        reconstructed = np.real(amp) * norm
        print(f"{num:>6}  {amp:>12.6f}  {reconstructed:>13.6f}")


    print("\n=== IQP Encoding ===")
    iqp_coeffs = iqp_encoding(numbers)
    n_qubits = int(np.log2(len(iqp_coeffs)))
    print(f"Qubits: {n_qubits}, Statevector size: {len(iqp_coeffs)}")
    probs = np.abs(iqp_coeffs) ** 2
    top_indices = np.argsort(probs)[::-1][:10]
    print(f"{'Index':>6}  {'Amplitude':>24}  {'Probability':>12}")
    print("-" * 48)
    for i in top_indices:
        print(f"{i:>6}  {iqp_coeffs[i]:>24}  {probs[i]:>12.6f}")

    print("\nIQP Reconstruction (via single-qubit marginals):")
    sv_size = len(iqp_coeffs)
    print(f"{'Original':>10}  {'Reconstructed':>14}")
    print("-" * 28)
    for q in range(n_qubits):
        prob_1 = sum(
            np.abs(iqp_coeffs[k]) ** 2
            for k in range(sv_size)
            if (k >> q) & 1
        )
        recovered_angle = np.arctan2(np.sqrt(prob_1), np.sqrt(1 - prob_1))
        reconstructed = (recovered_angle / (np.pi / 2)) * (10 - 0) + 0
        print(f"{numbers[q]:>10}  {reconstructed:>14.6f}")


if __name__ == '__main__':
    main()
