#!/usr/bin/env python
# coding: utf-8

# In[1]:


from qiskit import QuantumCircuit, converters
from qiskit import IBMQ, Aer, execute
from qiskit.providers.aer.noise import *
from qiskit.quantum_info import *
import qiskit.tools.jupyter
import random
import math
import numpy as np
# import basic plot tools
from qiskit.tools.visualization import *


# In[2]:


provider = IBMQ.load_account()
real_backend = provider.get_backend('ibmq_armonk')
properties = real_backend.properties()    
coupling_map = real_backend.configuration().coupling_map
noise_model = NoiseModel.from_backend(properties)
basis_gates = noise_model.basis_gates


# In[3]:


#%qiskit_version_table


# In[4]:


def create_random_circuit(num_qubits, depth=1, only_gates=None):
    circ = QuantumCircuit(num_qubits, num_qubits)
    
    # set the default gates if none were specified
    if only_gates is None:
        only_gates = [ 'h', 'x', 'y', 'z' ]
        if num_qubits > 1:
            only_gates.extend([ 'cx', 'cz' ])
    
    for d in range(depth):
        for q in range(num_qubits):
            other_qubits = list(range(num_qubits))
            other_qubits.remove(q)            
            q2 = None
            q3 = None
            
            if num_qubits >= 2:
                q2 = random.choice(other_qubits)
                other_qubits.remove(q2)
            if num_qubits >= 3:
                q3 = random.choice(other_qubits)
                other_qubits.remove(q3)
            
            theta = random.uniform(0, math.pi)
            phi = random.uniform(0, math.pi)
            lam = random.uniform(0, math.pi) 
            
            # build up gate list using closures w/ q for convenience
            add_gate = {
                'u1': lambda: circ.u1(theta, q),
                'u2': lambda: circ.u2(phi, lam, q),
                'u3': lambda: circ.u3(theta, phi, lam, q),
                
                'h': lambda: circ.h(q),
                'rx': lambda: circ.rx(theta, q),
                'ry': lambda: circ.ry(theta, q),
                'rz': lambda: circ.rz(phi, q),
                's': lambda: circ.s(q),
                'sdg': lambda: circ.sdg(q),
                't': lambda: circ.t(q),
                'tdg': lambda: circ.tdg(q),
                'x': lambda: circ.x(q),
                'y': lambda: circ.y(q),
                'z': lambda: circ.z(q),
            }
            
            if num_qubits > 1:
                add_gate.update({
                    'ch': lambda: circ.ch(q, q2),
                    'cnot': lambda: circ.cx(q, q2),
                    'crx': lambda: circ.crx(theta, q, q2),
                    'cry': lambda: circ.cry(theta, q, q2),
                    'crz': lambda: circ.crz(theta, q, q2),
                    'cu1': lambda: circ.cu1(theta, q, q2),
                    'cu3': lambda: circ.cu3(theta, phi, lam, q, q2),
                    'cx': lambda: circ.cx(q, q2),
                    'cy': lambda: circ.cy(q, q2),
                    'cz': lambda: circ.cz(q, q2),
                    'dcx': lambda: circ.dcx(q, q2), 
                    'swap': lambda: circ.swap(q, q2),
                })
                
            if num_qubits > 2:
                add_gate.update({
                    'ccx': lambda: circ.ccx(q, q2, q3),
                    'cswap': lambda: circ.cswap(q, q2, q3),
                    'fredkin': lambda: circ.cswap(q, q2, q3),
                    'toffoli': lambda: circ.ccx(q, q2, q3),
                })
            
            add_gate.get(only_gates[random.randrange(len(only_gates))])()
            
    return circ


# In[5]:


def run(circ, shots, backend=None, noisy=False, disp=False):
    if (backend == None):
        backend = Aer.get_backend('qasm_simulator')
#         backend = Aer.get_backend('statevector_simulator')
    
    using_statevector_backend = backend.name() == 'statevector_simulator'
    if (noisy and not using_statevector_backend):
        print("Using noisy simulator")
        job = execute(circ, backend, shots=shots, coupling_map=coupling_map, noise_model=noise_model, basis_gates=basis_gates)
    else:
        job = execute(circ, backend, shots=shots)
    results = job.result()
    counts = results.get_counts()
    statevector = results.get_statevector(decimals=6) if using_statevector_backend else '[no statevector]'
    if disp:
        print("%s %s" % (counts, statevector))
        if using_statevector_backend:
            plot_bloch_multivector(statevector)
        else:
            plot_histogram(counts)
    return counts

def run_statevector(circ, disp=False):
    statevector_circ = circ.remove_final_measurements(inplace=False)
    backend = Aer.get_backend('statevector_simulator')
    job = execute(statevector_circ, backend, shots=1)
    results = job.result()
    statevector = results.get_statevector()
    if disp:
        print("%s" % (statevector))
        iplot_bloch_multivector(statevector)
    return statevector

def run_unitary(circ, disp=False):
    unitary_circ = circ.remove_final_measurements(inplace=False)
    backend = Aer.get_backend('unitary_simulator')
    job = execute(unitary_circ, backend, shots=1)
    results = job.result()
    unitary = results.get_unitary()
    if disp:
        print("%s" % (unitary))
    return unitary


# In[6]:


def compare_statevector_fidelity(random_circ, circ):
    rsv = Statevector(run_statevector(random_circ))
    #print(rsv)
    sv = Statevector(run_statevector(circ))
    #print(sv)
    rsv_counts = rsv.probabilities_dict()
    sv_counts = sv.probabilities_dict()
    return hellinger_fidelity(rsv_counts, sv_counts)

def compare_statevector_equiv(random_circ, circ):
    rsv = Statevector(run_statevector(random_circ))
    sv = Statevector(run_statevector(circ))
    return rsv.equiv(sv)

def compare_statevector_with_phase(random_circ, circ):
    rsv = run_statevector(random_circ)
    sv = run_statevector(circ)
    return not np.any(np.abs(rsv - sv) > 1e-03) # if diff is > 0, then this will fail

def compare_circuit_norm(random_circ, circ):
    ru = run_unitary(random_circ)
    su = run_unitary(circ)
    return 1 - np.tanh(np.linalg.norm(ru - su, ord=2)) # convert L2 norm to a [0-1) value

def get_verification_level_text(level):
    if level == 0:
        return "0: Outcome probabilities must match (disregards phase)"
    elif level == 1:
        return "1: Statevectors must match (including phase)"
    elif level == 2:
        return "2: Circuit matrices must match"
    elif level == 3:
        return "3: Circuits must match exactly"

def check_circuit(random_circ, circ, level=0, log=True):
    # TODO: do we want to print each level's fidelity?
    debug = True and log
    
    # hellinger fidelity between the statevectors - is this the same as the next level though?
    fidelity = [compare_statevector_fidelity(random_circ, circ)]
    if debug:
        print("%s: %s (Hellinger fidelity)" % (get_verification_level_text(0), fidelity[0]))
    
    # full statevector equality (statevectors exactly match)
    if level >= 1:        
        fidelity.append(compare_statevector_with_phase(random_circ, circ))
        if debug:
            print("%s: %s (Statevector equality)" % (get_verification_level_text(1), fidelity[1]))
            if not np.isclose(fidelity[1], 1.0):
                with np.printoptions(precision=3, suppress=True):                
                    print(" current statevector: %s" % run_statevector(circ))
                    print(" target statevector: %s" % run_statevector(random_circ))
#                 print(" difference between statevectors: %s" % (run_statevector(random_circ) - run_statevector(circ)))

    # the norm of the difference between the two circuit (matrices)
    if level >= 2:
        fidelity.append(compare_circuit_norm(random_circ, circ))
        if debug:
            print("%s: %s (Norm of the difference between circuit matrices)" % (get_verification_level_text(2), fidelity[2]))
            if not np.isclose(fidelity[2], 1.0):
                with np.printoptions(precision=3, suppress=True):
                    print(" current unitary:\n %s" % run_unitary(circ))
                    print(" target unitary:\n %s" % run_unitary(random_circ))
#                     print(" difference between matrices:\n %s" % (run_unitary(random_circ) - run_unitary(circ)))
        
    # full equivalency
    if level >= 3:
        fidelity.append(random_circ == circ)
        if debug:
            print("%s: %s" % (get_verification_level_text(3), fidelity[3]))            
    
    total_fidelity = np.mean(fidelity)
    
    if np.isclose(total_fidelity, 1.0, rtol=1e-03, atol=1e-03):
        if log:
            print()
            print("Congratulations, your intuition has grown stronger. This was the hidden circuit:")
            print(random_circ)
        return True
    else:
        if log:
            print()
            print("Not quite there yet: %s (fidelity)" % total_fidelity)
        return False
        


# In[7]:


class HiddenCircuit:    
    def __init__(self, num_qubits, depth=1, gate_set=None, level=0):
        self.__random_circ = create_random_circuit(num_qubits, depth, gate_set)
        self.__unlocked = False
        self.__level = level
        
    def check(self, circ):
        self.__unlocked = check_circuit(self.__random_circ, circ, self.__level)
        return self.__unlocked
    
    def get_empty_circuit(self):
        rc = self.__random_circ
        return QuantumCircuit(rc.num_qubits, rc.num_clbits)

    def prepend(self, append_circ, plot_fn=plot_bloch_multivector):
        sv = run_statevector(append_circ.combine(self.__random_circ))
        display(plot_fn(sv))
        return
    
    def append(self, append_circ, plot_fn=plot_bloch_multivector):
        sv = run_statevector(self.__random_circ.combine(append_circ))
        display(plot_fn(sv))
        return
    
    def __str__(self):
        if self.__unlocked:
            return str(self.__random_circ)
        else:
            return "Wouldn't you like to know? Unlock this mystery via check()"

    def __eq__(self, other):
        return self.__random_circ == other


# In[8]:


class Sensei:
    def __init__(self, seed, stage=0, gate_set=None, verbosity=1, auto_submit=False):
        self.__seed = seed
        self.__stage = stage
        self.__hidden_circ = None
        self.__gate_set = gate_set
        self.__verbosity = verbosity
        self.__auto_submit = auto_submit
        
        if self.__verbosity >= 1:
            print("Welcome to the dojo. You have chosen a practice governed by the number '%s'" % seed)
            print()
            print("When you are ready to practice(), prepare to receive a hidden circuit.")
            print()
            print("You can get_empty_circuit() in order to probe the hidden circuit with prepend() or append().")
            print("You can also get_empty_circuit() to replicate the hidden circuit and check() to see how close you are.")
            print()
            print("When you are ready to move to the next stage submit(replicated_circuit) back to me.")
        
    def practice(self):
        if self.__auto_submit and self.__hidden_circ is not None             and self.__hidden_circ.__dict__['_HiddenCircuit__unlocked']:
            self.submit(self.__hidden_circ.__dict__['_HiddenCircuit__random_circ'])
            
        if self.__hidden_circ is None:
            stage = self.__stage
            # epochs increase with qubit first, then verification levels, and then depth
            exercises_per_depth = 3
            depths_per_epoch = 5
            stages_per_epoch = depths_per_epoch * exercises_per_depth
            epoch = math.floor(stage / stages_per_epoch)
            num_qubits = math.floor(stage / stages_per_epoch) + 1
            depth = math.floor((stage % stages_per_epoch) / (depths_per_epoch - 1)) + 1
            level = 0
            
            gate_set = self.__gate_set
            if gate_set is None:
                # 1 qubit
                if stage < 10:
                    gate_set = ['h', 'x', 'y', 'z']                    
                    level = 0
                elif stage < 15:
                    gate_set = ['h', 'x', 'y', 'z', 's', 't']
                    level = 1
                # 2 qubits
                elif stage < 20:
                    gate_set = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg']
                    level = 1
                elif stage < 30:
                    gate_set = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'cx']
                    level = 2
                # 3 qubits
                elif stage < 45:
                    gate_set = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'cx', 'rx', 'ry', 'rz']
                    level = 2
                # 4 qubits
                elif stage < 60:
                    gate_set = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'cx', 'rx', 'ry', 'rz', 'cy', 'cz']
                    level = 2
                # 5 qubits
                elif stage < 75:
                    gate_set = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'cx', 'rx', 'ry', 'rz', 'cy', 'cz', 'ccx']
                    level = 2
                else:
                    gate_set = ['h', 'x', 'y', 'z', 's', 't', 'sdg', 'tdg', 'cx', 'rx', 'ry', 'rz', 'cy', 'cz', 'ccx', 'u3']
                    level = 3
                    
            print("Stage %s: %s qubit(s), circuit depth %s, verification level %s" % (stage, num_qubits, depth, level))
            print("Using gate set: %s" % gate_set)
            print("NOTE: control gates only count towards the depth of the control wire(s)")
            if self.__verbosity >= 1:
                print("%s [must reach this to move on]" % get_verification_level_text(level))
            random.seed(self.__seed + stage)
            self.__hidden_circ = HiddenCircuit(num_qubits, depth=depth, gate_set=gate_set, level=level)
            return self.__hidden_circ
        else:
            print("You must complete your current practice before receiving a new one. Receive your hidden circuit.")
            return self.__hidden_circ
        
    def submit(self, solution_circ):
        if self.__hidden_circ is not None:
            show_output = True
            
            rc = self.__hidden_circ.__dict__['_HiddenCircuit__random_circ']
            if self.__auto_submit and solution_circ == rc:
                correct = True
                show_output = False
            else:
                correct = check_circuit(rc, solution_circ, self.__hidden_circ.__dict__['_HiddenCircuit__level'], log=False)
            if correct:              
                if show_output:
                    print("Well done. You are ready for more practice().")
                self.__stage = self.__stage + 1
                self.__hidden_circ = None
                return
            else:
                if show_output:
                    print("You are not ready for more practice(). Submit again when you have proven yourself.")
                return
        else:
            print("You must practice() first before submitting.")
            return
        
    def _test_all():
        # Test all levels
        sensei = Sensei(42)
        for i in range(100):
            hc = sensei.practice()
            rc = sensei.__dict__['_Sensei__hidden_circ'].__dict__['_HiddenCircuit__random_circ']
            sensei.submit(rc)
            print()
#             print(solution)


# In[ ]:




