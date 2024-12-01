import numpy as np
import matplotlib.pyplot as plt

# Scenario is that a robber runs away at velocity 1.5/min with a 5 min headstart on the sheriff who has velocity 2.5
# Robber also has 5.5 units head start

t = np.linspace(0, 40, 1000)

d_robber = 1.5 * t + 5.5
d_sheriff = 2.5 * (t-5) + 0

#solve equation algebraically
# d_robber = 1.5t + 5.5 = 2.5*t - 5*2.5
# 5.5+5*2.5 = 1t
# t = (5.5+12.5)/1
intercept_t = (5.5+12.5)/1

fig, ax = plt.subplots()
plt.title("A bank robber being chased by the sheriff")
plt.xlabel("Time (min)")
plt.ylabel("Distance (units)")
ax.set_xlim([0,40])
ax.set_ylim([0, 100])
ax.plot(t, d_robber, c='green')
ax.plot(t, d_sheriff, c='blue')
plt.axvline(x=intercept_t, color = 'purple', linestyle = '--')
_ = plt.axhline(y=1.5*intercept_t+5.5, color = 'purple', linestyle = '--')

plt.show()