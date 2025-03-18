"""
A script for simulating the heat transfer for an n-dimensional cuboid.
"""

import sys
import numpy as np
import matplotlib.pyplot as plt

# Parameters

# Dimensions (cm)
THICKNESS = 2.5
WIDTH = 12.5 
LENGTH = 20

TEMP_COUNT = 100 # 200
TIME_RESOLUTION = 0.01 # s
MEAT_ALPHA = 0.001 # Thermal diffusivity for meat cm^2/s

MAX_TIME = 300.0

# How often to sample the time steps for a 200x200 heatmap grid
timeSampleInterval = int((MAX_TIME / TIME_RESOLUTION) / 200.0)
if timeSampleInterval == 0:
  timeSampleInterval = 1

# Double-check convergence for variables
if TIME_RESOLUTION > ((THICKNESS / TEMP_COUNT) ** 2) / (2 * MEAT_ALPHA):
  print("WARNING: The time steps may not be small enough for the solutions to converge!")

# Temps (K)
INITIAL_STEAK_TEMP = 293
PAN_TEMP = 473
OVEN_TEMP = 453
ROOM_TEMP = 293

"""A class for the simulation of a steak"""
class Steak():

  def __init__(self, initialCond, size, boundaryCondFunc, dims=1):

    # Check for dimensions of params
    if not (dims == 1 or dims == 2):
      print("Invalid dimensions for steak - 1D or 2D only.")
      return 

    self.isValid = False
    try:
      if len(size) != dims or initialCond.ndim != dims:
        print("Mismatched dimensions for initial conditions and size.")
        return
    except:
      return
    self.isValid = True

    # Set up parameters
    self.size = size

    # Add an extra "boundary temperature" around the edges
    if dims == 1:
      self.temperatures = []

      self.temperatures.append(0.0)
      for temp in initialCond:
        self.temperatures.append(temp)
      self.temperatures.append(0.0)

      self.temperatures = np.array(self.temperatures)

    elif dims == 2:
      self.temperatures = np.zeros((initialCond.shape[0] + 2, initialCond.shape[1] + 2))

      for index in np.ndindex(initialCond.shape):
        self.temperatures[(index[0] + 1, index[1] + 1)] = initialCond[index]

    # The temperature profile recorded at points in time
    self.heatmapSamples = [] 
    self.internalTemps = []

    self.maillardData = [0.0 for i in range(len(initialCond))]
    self.maillardHistory = []
    self.maillardTotal = 0.0

    self.timeIndex = 0

    self.ApplyBoundaryConds = boundaryCondFunc

  """Applies the RK4 method to obtain the second spacial derivative."""
  def RungeKutta(self, t1, t2, t3, dx2) -> float:

    dt = TIME_RESOLUTION

    k1 = t3 - 2 * t2 + t1
    k1 /= dx2

    k2 = t3 + (dt * k1) / 2.0 - 2 * (t2 + (dt / 2.0) * k1) + t1 + (dt / 2.0) * k1
    k2 /= dx2

    k3 = t3 + (dt / 2.0) * k2 - 2 * (t2 + (dt / 2.0) * k2) + t1 + (dt / 2.0) * k2
    k3 /= dx2

    k4 = t3 + dt * k3 - 2 * (t2 + dt * k3) + t1 + dt * k3
    k4 /= dx2

    return (k1 + 2 * k2 + 2 * k3 + k4) / 6.0

  """ Performs a single time step simulation of the heat flow.
  """
  def Simulate(self):

    if not self.isValid:
      return

    time = self.timeIndex * TIME_RESOLUTION

    # Apply boundary conditions
    self.ApplyBoundaryConds(self.temperatures, time)

    # # Save current state to heat map
    # if self.timeIndex % timeSampleInterval == 0:
    #   self.heatmapSamples.append((time, self.temperatures.copy()))
    #   self.maillardHistory.append((time, self.maillardTotal))
    self.timeIndex += 1

    # Calculate next temperatures
    # tempCount = len(self.temperatures)
    # dx = self.thickness / tempCount

    newTemps = np.zeros((self.temperatures.shape[0] - 1, self.temperatures.shape[1] - 1))
    for index in np.ndindex(self.temperatures.shape):
      
      # Ignore updating boundary temps (these are set by boundary conditions)
      # if x == 0 or x == tempCount - 1:
        # continue

      # Ignore updating boundary temps (these are set by boundary conditions)
      if index[0] == 0 or index[1] == 0 or index[0] == self.temperatures.shape[0] - 1 or index[1] == self.temperatures.shape[1] - 1:
        continue

      # Calculate next temperature value
      laplacian = 0

      # X deriv
      t1 = self.temperatures[(index[0] - 1, index[1])]
      t2 = self.temperatures[(index[0], index[1])]
      t3 = self.temperatures[(index[0] + 1, index[1])]

      dx2 = (THICKNESS / TEMP_COUNT) ** 2

      laplacian += self.RungeKutta(t1, t2, t3, dx2)

      # Y deriv
      t1 = self.temperatures[(index[0], index[1] - 1)]
      t2 = self.temperatures[(index[0], index[1])]
      t3 = self.temperatures[(index[0], index[1] + 1)]

      dx2 = (WIDTH / TEMP_COUNT) ** 2

      laplacian += self.RungeKutta(t1, t2, t3, dx2)

      newTemps[(index[0] - 1, index[1] - 1)] = self.temperatures[index] + TIME_RESOLUTION * MEAT_ALPHA * laplacian

      # # Check Maillard condition (140-165)
      # if t2 >= 423 and t2 <= 438:
      #   self.maillardData[x - 1] += TIME_RESOLUTION
      #   self.maillardTotal += dx * TIME_RESOLUTION

    # Update new temperatures
    for index in np.ndindex(newTemps.shape):
      self.temperatures[(index[0] + 1, index[1] + 1)] = newTemps[index]

# Air-steak constants
h = 0.002 # W/cm^2 K
k = 0.005  # W/cm K

"""Boundary conditions for an n-flip pan sear. Ideally n should be odd."""
def n_flip_sear(cookTime, flipCount, temps, time):

  flipTime = cookTime / (flipCount + 1)

  if int(time / flipTime) % 2 == 0:
    temps[0] = PAN_TEMP

    # Fix derivative
    diff = (h/k) * (ROOM_TEMP - temps[len(temps) - 2])
    temps[len(temps) - 1] = temps[len(temps) - 2] + diff * (THICKNESS / TEMP_COUNT)

  else:
    temps[len(temps) - 1] = PAN_TEMP

    # Fix derivative
    diff = (h/k) * (ROOM_TEMP - temps[1])
    temps[0] = temps[1] + diff * (THICKNESS / TEMP_COUNT)

"""Boundary conditions for a single-flip pan sear"""
def pan_sear_cond(cookTime, temps, time) -> None:
  n_flip_sear(cookTime, 1, temps, time)

"""Single-flip pan sear in two dimensions"""
def pan_sear_cond_2d(cookTime, temps, time) -> None:

  flipTime = cookTime / 2

  rowSize = temps.shape[0]
  colSize = temps.shape[1]

  if int(time / flipTime) % 2 == 0:
    temps[0, : ] = PAN_TEMP

    # Fix derivatives on air boundaries
    diff = (h/k) * (ROOM_TEMP - temps[1:rowSize - 1, colSize - 2])
    temps[1:rowSize - 1, colSize - 1] = temps[1:rowSize - 1, colSize - 2] + diff * (THICKNESS / TEMP_COUNT)

    diff = (h/k) * (ROOM_TEMP - temps[1:rowSize - 1, 1])
    temps[1:rowSize - 1, 0] = temps[1:rowSize - 1, 1] + diff * (THICKNESS / TEMP_COUNT)

    diff = (h/k) * (ROOM_TEMP - temps[rowSize - 2, 1:colSize - 1])
    temps[rowSize - 1, 1:colSize - 1] = temps[rowSize - 2, 1:colSize - 1] + diff * (THICKNESS / TEMP_COUNT)

  else:
    temps[rowSize - 1, :] = PAN_TEMP

    # Fix derivatives on air boundaries
    diff = (h/k) * (ROOM_TEMP - temps[1:rowSize - 1, colSize - 2])
    temps[1:rowSize - 1, colSize - 1] = temps[1:rowSize - 1, colSize - 2] + diff * (THICKNESS / TEMP_COUNT)

    diff = (h/k) * (ROOM_TEMP - temps[1:rowSize - 1, 1])
    temps[1:rowSize - 1, 0] = temps[1:rowSize - 1, 1] + diff * (THICKNESS / TEMP_COUNT)

    diff = (h/k) * (ROOM_TEMP - temps[1, 1:colSize - 1])
    temps[0, 1:colSize - 1] = temps[1, 1:colSize - 1] + diff * (THICKNESS / TEMP_COUNT)

"""Boundary conditions for a reverse sear"""
def reverse_sear_cond(ovenTime, ovenTemp, searTime, temps, time):
  
  #if temps[int((len(temps) - 1) / 2.0)] >= 318:
    #print("Time to internal temp: " + str(time))
    #sys.exit()

  if time < ovenTime:
    # Fix derivative (both sides)
    diff = (h/k) * (ovenTemp - temps[len(temps) - 2])
    temps[len(temps) - 1] = temps[len(temps) - 2] + diff * (THICKNESS / TEMP_COUNT)

    diff = (h/k) * (ovenTemp - temps[1])
    temps[0] = temps[1] + diff * (THICKNESS / TEMP_COUNT)
  else:
    pan_sear_cond(searTime, temps, time - ovenTime)

initialSteakTemp = np.full((TEMP_COUNT, TEMP_COUNT), INITIAL_STEAK_TEMP)

panSearSteak = Steak(initialSteakTemp, [THICKNESS, WIDTH], lambda temps, time : pan_sear_cond_2d(300.0, temps, time), 2)
# panSearSteak = Steak([INITIAL_STEAK_TEMP for i in range(TEMP_COUNT)], THICKNESS, lambda temps, time : n_flip_sear(300.0, 19, temps, time))
# panSearSteak = Steak([INITIAL_STEAK_TEMP for i in range(TEMP_COUNT)], THICKNESS, lambda temps, time : reverse_sear_cond(800.0, 453.0, 120.0, temps, time))
# panSearSteak = Steak(np.array([INITIAL_STEAK_TEMP for i in range(TEMP_COUNT)]), [THICKNESS], lambda temps, time : reverse_sear_cond(1600.0, 373.0, 120.0, temps, time))

timeSteps = int(MAX_TIME / TIME_RESOLUTION)
for t in range(timeSteps):
  panSearSteak.Simulate()

  if t % 1000 == 0:
    print("Time Steps: " + str(t))

# Save 2D temps

dataOut = "x y T\n"
for index in np.ndindex(panSearSteak.temperatures.shape):

  if index[0] == 0 or index[1] == 0 or index[0] == panSearSteak.temperatures.shape[0] - 1 or index[1] == panSearSteak.temperatures.shape[1] - 1:
    continue

  xPos = (index[0] - 1) * (THICKNESS / (TEMP_COUNT - 1))
  yPos = (index[1] - 1) * (WIDTH / (TEMP_COUNT - 1))

  dataOut += f"{xPos} {yPos} {panSearSteak.temperatures[index]}\n"

dataFile = open("2d_steak_data.txt", 'w')
dataFile.write(dataOut)
dataFile.close()

# Save to file
# pltValues = 0

# dataOutput = "x t T\n"

# for row in panSearSteak.heatmapSamples:
#   tValue = row[0]

#   tmp = []
#   for x in range(len(row[1][1: len(row[1]) - 1])):
#     xValue = x * (THICKNESS / (TEMP_COUNT - 1))
#     dataOutput += str(xValue) + ' ' + str(tValue) + ' ' + str(row[1][x]) + '\n'
#     tmp.append(row[1][x])

#   pltValues += 1

# print(f"Grid size: {TEMP_COUNT} x {pltValues}")

# FILE_OUTPUT = "data.txt"
# outFile = open(FILE_OUTPUT, 'w')
# outFile.write(dataOutput)
# outFile.close()

# # Maillard output (space)
# maillardData = "x M\n"
# for x in range(len(panSearSteak.maillardData)):
#   xValue = x * (THICKNESS / (TEMP_COUNT - 1))
#   maillardData += f"{xValue} {panSearSteak.maillardData[x]}\n"

# outFile = open("maillard-space.txt", 'w')
# outFile.write(maillardData)
# outFile.close()

# # Maillard output (time)
# maillardData = "t M\n"
# for data in panSearSteak.maillardHistory:
#   maillardData += f"{data[0]} {data[1]}\n"

# outFile = open("maillard-time.txt", 'w')
# outFile.write(maillardData)
# outFile.close()