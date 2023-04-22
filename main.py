import streamlit as st
import pandas as pd
import numpy as np
import time

import matplotlib.pyplot as plt
from qiskit import QuantumCircuit
from qiskit.algorithms import IterativeAmplitudeEstimation, EstimationProblem
from qiskit.circuit.library import LinearAmplitudeFunction
from qiskit_aer.primitives import Sampler
from qiskit_finance.circuit.library import LogNormalDistribution

############################################################################################################

# Global variables
# number of qubits to represent the uncertainty
num_uncertainty_qubits = 3
progress_text = "Operation in progress. Please wait."

st.title('Quantum Option Pricing')


st.write("This app uses the Quantum Amplitude Estimation (QAE) algorithm to price European options")


option_type = st.sidebar.selectbox('Option Type', ('Call', 'Put'))

if st.button('Run QAE'):
  my_bar = st.progress(0, text=progress_text)
  for percent_complete in range(100):
    time.sleep(0.1)
    my_bar.progress(percent_complete + 1, text=progress_text)

  
  if percent_complete == 99:
    my_bar.progress(100, text="Done!")


  
st.sidebar.markdown("Option Parameters:")
# parameters for considered random distribution
S = st.sidebar.number_input('Spot Price', min_value=0.0, max_value=100.0, value=2.0, step=0.01)
days = st.sidebar.number_input('Time to Maturity', min_value=0, max_value=365, value=40, step=1)
T = days / 365  # 40 days to maturity
vol = st.sidebar.slider('Volatility', 0.0, 1.0, 0.4, 0.01) # min, max, default, step
r = st.sidebar.slider('Annual Interest Rate', 0.0, 1.0, 0.05, 0.01) # min, max, default, step


# resulting parameters for log-normal distribution
mu = (r - 0.5 * vol**2) * T + np.log(S)
sigma = vol * np.sqrt(T)
mean = np.exp(mu + sigma**2 / 2)
variance = (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2)
stddev = np.sqrt(variance)

# lowest and highest value considered for the spot price; in between, an equidistant discretization is considered.
low = np.maximum(0, mean - 3 * stddev)
high = mean + 3 * stddev

# construct A operator for QAE for the payoff function by
# composing the uncertainty model and the objective
uncertainty_model = LogNormalDistribution(
    num_uncertainty_qubits, mu=mu, sigma=sigma**2, bounds=(low, high)
)

st.subheader("1. Uncertainty Model")

# plot probability distribution
x = uncertainty_model.values
y = uncertainty_model.probabilities
fig, ax = plt.subplots()
ax = plt.bar(x, y, width=0.2)
plt.xticks(x, size=15, rotation=90)
plt.yticks(size=15)
plt.grid()
plt.xlabel("Spot Price at Maturity $S_T$ (\$)", size=15)
plt.ylabel("Probability ($\%$)", size=15)
plt.title("Probability Distribution of Spot Price at Maturity", size=15)

st.pyplot(fig)


st.subheader("2. Payoff Function")
# set the strike price (should be within the low and the high value of the uncertainty)
strike_price = st.number_input('Strike Price (should be within the low and the high value of the uncertainty)'
                              , min_value=low, max_value=high, value=2.0, step=0.01)

# set the approximation scaling for the payoff function
c_approx = 0.25

if option_type == 'Call':

  # setup piecewise linear objective fcuntion
  breakpoints = [low, strike_price]
  slopes = [0, 1]
  offsets = [0, 0]
  f_min = 0
  f_max = high - strike_price
  european_call_objective = LinearAmplitudeFunction(
      num_uncertainty_qubits,
      slopes,
      offsets,
      domain=(low, high),
      image=(f_min, f_max),
      breakpoints=breakpoints,
      rescaling_factor=c_approx,
  )
  # construct A operator for QAE for the payoff function by
  # composing the uncertainty model and the objective
  num_qubits = european_call_objective.num_qubits
  european_call = QuantumCircuit(num_qubits)
  european_call.append(uncertainty_model, range(num_uncertainty_qubits))
  european_call.append(european_call_objective, range(num_qubits))

  # plot exact payoff function (evaluated on the grid of the uncertainty model)
  x = uncertainty_model.values
  y = np.maximum(0, x - strike_price)
  fig2, ax2 = plt.subplots()
  ax2 = plt.plot(x, y, "ro-")
  plt.grid()
  plt.title("Payoff Function", size=15)
  plt.xlabel("Spot Price", size=15)
  plt.ylabel("Payoff", size=15)
  plt.xticks(x, size=15, rotation=90)
  plt.yticks(size=15)

  st.pyplot(fig2)

elif option_type == 'Put':
  # setup piecewise linear objective fcuntion
  breakpoints = [low, strike_price]
  slopes = [-1, 0]
  offsets = [strike_price - low, 0]
  f_min = 0
  f_max = strike_price - low
  european_put_objective = LinearAmplitudeFunction(
      num_uncertainty_qubits,
      slopes,
      offsets,
      domain=(low, high),
      image=(f_min, f_max),
      breakpoints=breakpoints,
      rescaling_factor=c_approx,
  )
  # construct A operator for QAE for the payoff function by
  # composing the uncertainty model and the objective
  european_put = european_put_objective.compose(uncertainty_model, front=True)

  # plot exact payoff function (evaluated on the grid of the uncertainty model)
  x = uncertainty_model.values
  y = np.maximum(0, strike_price - x)
  fig2, ax2 = plt.subplots()
  ax2 = plt.plot(x, y, "bo-")
  plt.grid()
  plt.title("Payoff Function", size=15)
  plt.xlabel("Spot Price", size=15)
  plt.ylabel("Payoff", size=15)
  plt.xticks(x, size=15, rotation=90)
  plt.yticks(size=15)
  
  st.pyplot(fig2)





# evaluate exact expected value (normalized to the [0, 1] interval)
exact_value = np.dot(uncertainty_model.probabilities, y)
if option_type == 'Call':
  exact_delta = sum(uncertainty_model.probabilities[x >= strike_price])
elif option_type == 'Put':
  exact_delta = -sum(uncertainty_model.probabilities[x <= strike_price])
st.write("Exact expected value:\t%.4f" % exact_value)
st.write("Exact delta value:   \t%.4f" % exact_delta)


st.subheader("3. Evaluate the Expected Value of the Payoff Function")

# set target precision and confidence level
epsilon = st.slider('Target Precision', 0.0, 0.10, 0.01, 0.01) # min, max, default, step
alpha = st.slider('Confidence Level', 0.0, 0.10, 0.05, 0.01) # min, max, default, step

if option_type == 'Call':
  problem = EstimationProblem(
    state_preparation=european_call,
    objective_qubits=[3],
    post_processing=european_call_objective.post_processing,
  )
  # construct amplitude estimation
  ae = IterativeAmplitudeEstimation(
      epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 100})
  )

  # run amplitude estimation
  result = ae.estimate(problem)

  conf_int = np.array(result.confidence_interval_processed)
  st.write("Exact value:        \t%.4f" % exact_value)
  st.write("Estimated value:    \t%.4f" % (result.estimation_processed))
  st.write("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))

elif option_type == 'Put':
  problem = EstimationProblem(
    state_preparation=european_put,
    objective_qubits=[num_uncertainty_qubits],
    post_processing=european_put_objective.post_processing,
  )
  # construct amplitude estimation
  ae = IterativeAmplitudeEstimation(
      epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 100})
  )

  result = ae.estimate(problem)

  conf_int = np.array(result.confidence_interval_processed)
  st.write("Exact value:        \t%.4f" % exact_value)
  st.write("Estimated value:    \t%.4f" % (result.estimation_processed))
  st.write("Confidence interval:\t[%.4f, %.4f]" % tuple(conf_int))




from qiskit_finance.applications.estimation import EuropeanCallDelta
st.subheader("4. Evaluate the Delta of the Payoff Function")

if option_type == 'Call':
  european_call_delta = EuropeanCallDelta(
      num_state_qubits=num_uncertainty_qubits,
      strike_price=strike_price,
      bounds=(low, high),
      uncertainty_model=uncertainty_model,
  )

  problem = european_call_delta.to_estimation_problem()

  # construct amplitude estimation
  ae_delta = IterativeAmplitudeEstimation(
      epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 100})
  )

  result_delta = ae_delta.estimate(problem)

  conf_int = np.array(result_delta.confidence_interval_processed)
  st.write("Exact delta:    \t%.4f" % exact_delta)
  st.write("Esimated value: \t%.4f" % european_call_delta.interpret(result_delta))
  st.write("Confidence interval: \t[%.4f, %.4f]" % tuple(conf_int))


elif option_type == 'Put':
  # setup piecewise linear objective fcuntion
  breakpoints = [low, strike_price]
  slopes = [0, 0]
  offsets = [1, 0]
  f_min = 0
  f_max = 1

  european_put_delta_objective = LinearAmplitudeFunction(
      num_uncertainty_qubits,
      slopes,
      offsets,
      domain=(low, high),
      image=(f_min, f_max),
      breakpoints=breakpoints,
  )

  # construct circuit for payoff function
  european_put_delta = european_put_delta_objective.compose(uncertainty_model, front=True)

  problem = EstimationProblem(
    state_preparation=european_put_delta, objective_qubits=[num_uncertainty_qubits]
  )
  # construct amplitude estimation
  ae_delta = IterativeAmplitudeEstimation(
      epsilon_target=epsilon, alpha=alpha, sampler=Sampler(run_options={"shots": 100})
  )

  result_delta = ae_delta.estimate(problem)

  conf_int = -np.array(result_delta.confidence_interval)[::-1]
  st.write("Exact delta:    \t%.4f" % exact_delta)
  st.write("Esimated value: \t%.4f" % -result_delta.estimation)
  st.write("Confidence interval: \t[%.4f, %.4f]" % tuple(conf_int))

st.subheader("5. Resources")
st.write("1. [Qiskit Finance Documentation](https://qiskit.org/documentation/finance.html)")
