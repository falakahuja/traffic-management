# LAG
# NO. OF VEHICLES IN SIGNAL CLASS
# stops not used
# DISTRIBUTION
# BUS TOUCHING ON TURNS
# Distribution using python class

# *** IMAGE XY COOD IS TOP LEFT
import random
import math
import time
import threading
# from vehicle_detection import detection
import pygame
import sys
import os

# options={
#    'model':'./cfg/yolo.cfg',     #specifying the path of model
#    'load':'./bin/yolov2.weights',   #weights
#    'threshold':0.3     #minimum confidence factor to create a box, greater than 0.3 good
# }

# tfnet=TFNet(options)    #READ ABOUT TFNET

# Default values of signal times
defaultRed = 150
defaultYellow = 5
defaultGreen = 20
defaultMinimum = 10
defaultMaximum = 60

signals = []
noOfSignals = 4
simTime = 300       # change this to change time of simulation
timeElapsed = 0

currentGreen = 0   # Indicates which signal is green
nextGreen = (currentGreen+1)%noOfSignals
currentYellow = 0   # Indicates whether yellow signal is on or off 

# Average times for vehicles to pass the intersection
carTime = 2
bikeTime = 1
rickshawTime = 2.25 
busTime = 2.5
truckTime = 2.5

# Count of cars at a traffic signal
noOfCars = 0
noOfBikes = 0
noOfBuses =0
noOfTrucks = 0
noOfRickshaws = 0
noOfLanes = 2

# Red signal time at which cars will be detected at a signal
detectionTime = 5

speeds = {'car':2.25, 'bus':1.8, 'truck':1.8, 'rickshaw':2, 'bike':2.5}  # average speeds of vehicles

# Coordinates of start
x = {'right':[0,0,0], 'down':[271,254,240], 'left':[1400,1400,1400], 'up':[200,210,225]}    
y = {'right':[223,232,250], 'down':[0,0,0], 'left':[300,285,268], 'up':[800,800,800]}

vehicles = {'right': {0:[], 1:[], 2:[], 'crossed':0}, 'down': {0:[], 1:[], 2:[], 'crossed':0}, 'left': {0:[], 1:[], 2:[], 'crossed':0}, 'up': {0:[], 1:[], 2:[], 'crossed':0}}
vehicleTypes = {0:'car', 1:'bus', 2:'truck', 3:'rickshaw', 4:'bike'}
directionNumbers = {0:'right', 1:'down', 2:'left', 3:'up'}

# Coordinates of signal image, timer, and vehicle count
'''Changes of the Singnal Coods'''
signalCoods = [(590,340),(675,260),(770,430),(675,510)]
signalTimerCoods = [(530,210),(810,210),(810,550),(530,550)]
vehicleCountCoods = [(480,210),(880,210),(880,550),(480,550)]
vehicleCountTexts = ["0", "0", "0", "0"]

# Coordinates of stop lines
stopLines = {'right': 210, 'down': 220, 'left': 270, 'up': 307}
defaultStop = {'right': 200, 'down': 210, 'left': 280, 'up': 317}
stops = {'right': [200,200,200], 'down': [320,320,320], 'left': [810,810,810], 'up': [545,545,545]}

mid = {'right': {'x':705, 'y':445}, 'down': {'x':695, 'y':450}, 'left': {'x':695, 'y':425}, 'up': {'x':695, 'y':400}}
rotationAngle = 3

# Gap between vehicles
gap = 7    # stopping gap
gap2 = 7   # moving gap

simulation = pygame.sprite.Group()

class TrafficSignal:
    def __init__(self, red, yellow, green, minimum, maximum):
        self.red = red
        self.yellow = yellow
        self.green = green
        self.minimum = minimum
        self.maximum = maximum
        self.signalText = "30"
        self.totalGreenTime = 0
        
class Vehicle(pygame.sprite.Sprite):
    def __init__(self, lane, vehicleClass, direction_number, direction, will_turn):
        pygame.sprite.Sprite.__init__(self)
        self.lane = lane
        self.vehicleClass = vehicleClass
        self.speed = speeds[vehicleClass]
        self.direction_number = direction_number
        self.direction = direction
        self.x = x[direction][lane]
        self.y = y[direction][lane]
        self.crossed = 0
        self.willTurn = will_turn
        self.turned = 0
        self.rotateAngle = 0
        vehicles[direction][lane].append(self)
        # self.stop = stops[direction][lane]
        self.index = len(vehicles[direction][lane]) - 1
        path = "images/" + direction + "/" + vehicleClass + ".png"
        self.originalImage = pygame.image.load(path)
        self.currentImage = pygame.image.load(path)

    
        if(direction=='right'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):    # if more than 1 vehicle in the lane of vehicle before it has crossed stop line
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().width - gap         # setting stop coordinate as: stop coordinate of next vehicle - width of next vehicle - gap
            else:
                self.stop = defaultStop[direction]
            # Set new starting and stopping coordinate
            temp = self.currentImage.get_rect().width + gap    
            x[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='left'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().width + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().width + gap
            x[direction][lane] += temp
            stops[direction][lane] += temp
        elif(direction=='down'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop - vehicles[direction][lane][self.index-1].currentImage.get_rect().height - gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] -= temp
            stops[direction][lane] -= temp
        elif(direction=='up'):
            if(len(vehicles[direction][lane])>1 and vehicles[direction][lane][self.index-1].crossed==0):
                self.stop = vehicles[direction][lane][self.index-1].stop + vehicles[direction][lane][self.index-1].currentImage.get_rect().height + gap
            else:
                self.stop = defaultStop[direction]
            temp = self.currentImage.get_rect().height + gap
            y[direction][lane] += temp
            stops[direction][lane] += temp
        simulation.add(self)

    def render(self, screen):
        screen.blit(self.currentImage, (self.x, self.y))
        
    def move(self):
        if(self.direction=='right'):
            if(self.crossed==0 and self.x+self.currentImage.get_rect().width>stopLines[self.direction]):   # if the image has crossed stop line now
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x+self.currentImage.get_rect().width<mid[self.direction]['x']):
                    if((self.x+self.currentImage.get_rect().width<=self.stop or (currentGreen==0 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 2
                        self.y += 1.8
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.image = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2)):
                            self.y += self.speed
            else: 
                if((self.x+self.currentImage.get_rect().width<=self.stop or self.crossed == 1 or (currentGreen==0 and currentYellow==0)) and (self.index==0 or self.x+self.currentImage.get_rect().width<(vehicles[self.direction][self.lane][self.index-1].x - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x += self.speed  # move the vehicle



        elif(self.direction=='down'):
            if(self.crossed==0 and self.y+self.currentImage.get_rect().height>stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y+self.currentImage.get_rect().height<mid[self.direction]['y']):
                    if((self.y+self.currentImage.get_rect().height<=self.stop or (currentGreen==1 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.y += self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 2.5
                        self.y += 2
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or self.y<(vehicles[self.direction][self.lane][self.index-1].y - gap2)):
                            self.x -= self.speed
            else: 
                if((self.y+self.currentImage.get_rect().height<=self.stop or self.crossed == 1 or (currentGreen==1 and currentYellow==0)) and (self.index==0 or self.y+self.currentImage.get_rect().height<(vehicles[self.direction][self.lane][self.index-1].y - gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y += self.speed
            
        elif(self.direction=='left'):
            if(self.crossed==0 and self.x<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.x>mid[self.direction]['x']):
                    if((self.x>=self.stop or (currentGreen==2 and currentYellow==0) or self.crossed==1) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):                
                        self.x -= self.speed
                else: 
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x -= 1.8
                        self.y -= 2.5
                        if(self.rotateAngle==90):
                            self.turned = 1
                            # path = "images/" + directionNumbers[((self.direction_number+1)%noOfSignals)] + "/" + self.vehicleClass + ".png"
                            # self.x = mid[self.direction]['x']
                            # self.y = mid[self.direction]['y']
                            # self.currentImage = pygame.image.load(path)
                    else:
                        if(self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or self.x>(vehicles[self.direction][self.lane][self.index-1].x + gap2)):
                            self.y -= self.speed
            else: 
                if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                # (if the image has not reached its stop coordinate or has crossed stop line or has green signal) and (it is either the first vehicle in that lane or it is has enough gap to the next vehicle in that lane)
                    self.x -= self.speed  # move the vehicle    
            # if((self.x>=self.stop or self.crossed == 1 or (currentGreen==2 and currentYellow==0)) and (self.index==0 or self.x>(vehicles[self.direction][self.lane][self.index-1].x + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width + gap2))):                
            #     self.x -= self.speed
        elif(self.direction=='up'):
            if(self.crossed==0 and self.y<stopLines[self.direction]):
                self.crossed = 1
                vehicles[self.direction]['crossed'] += 1
            if(self.willTurn==1):
                if(self.crossed==0 or self.y>mid[self.direction]['y']):
                    if((self.y>=self.stop or (currentGreen==3 and currentYellow==0) or self.crossed == 1) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height +  gap2) or vehicles[self.direction][self.lane][self.index-1].turned==1)):
                        self.y -= self.speed
                else:   
                    if(self.turned==0):
                        self.rotateAngle += rotationAngle
                        self.currentImage = pygame.transform.rotate(self.originalImage, -self.rotateAngle)
                        self.x += 1
                        self.y -= 1
                        if(self.rotateAngle==90):
                            self.turned = 1
                    else:
                        if(self.index==0 or self.x<(vehicles[self.direction][self.lane][self.index-1].x - vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().width - gap2) or self.y>(vehicles[self.direction][self.lane][self.index-1].y + gap2)):
                            self.x += self.speed
            else: 
                if((self.y>=self.stop or self.crossed == 1 or (currentGreen==3 and currentYellow==0)) and (self.index==0 or self.y>(vehicles[self.direction][self.lane][self.index-1].y + vehicles[self.direction][self.lane][self.index-1].currentImage.get_rect().height + gap2) or (vehicles[self.direction][self.lane][self.index-1].turned==1))):                
                    self.y -= self.speed

# Initialization of signals with default values
def initialize():
    ts1 = TrafficSignal(0, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts1)
    ts2 = TrafficSignal(ts1.red+ts1.yellow+ts1.green, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts2)
    ts3 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts3)
    ts4 = TrafficSignal(defaultRed, defaultYellow, defaultGreen, defaultMinimum, defaultMaximum)
    signals.append(ts4)
    repeat()

# Set time according to formula
def setTime():
    global noOfCars, noOfBikes, noOfBuses, noOfTrucks, noOfRickshaws, noOfLanes
    global carTime, busTime, truckTime, rickshawTime, bikeTime
    global currentGreen, nextGreen, signals

    #os.system("say detecting vehicles, " + directionNumbers[(currentGreen + 1) % noOfSignals])

    # Reset vehicle counts
    noOfCars = noOfBuses = noOfTrucks = noOfRickshaws = noOfBikes = 0

    # Count vehicles in the next direction
    for lane in range(3):  # assuming 3 lanes per direction
        for vehicle in vehicles[directionNumbers[nextGreen]][lane]:
            if vehicle.crossed == 0:
                vclass = vehicle.vehicleClass
                if vclass == 'car':
                    noOfCars += 1
                elif vclass == 'bus':
                    noOfBuses += 1
                elif vclass == 'truck':
                    noOfTrucks += 1
                elif vclass == 'rickshaw':
                    noOfRickshaws += 1
                elif vclass == 'bike':
                    noOfBikes += 1

    # Compute time based on weighted vehicle count
    greenTime = math.ceil(
        ((noOfCars * carTime) +
         (noOfRickshaws * rickshawTime) +
         (noOfBuses * busTime) +
         (noOfTrucks * truckTime) +
         (noOfBikes * bikeTime)) / (noOfLanes + 1)
    )

    # Check if NEAT AI suggests an override
    ai_green = get_neat_green_time()  # <- this should return an int or None
    if ai_green is not None:
        print(f"NEAT suggested green time: {ai_green}")
        greenTime = ai_green

    # Clamp green time
    if greenTime < defaultMinimum:
        greenTime = defaultMinimum
    elif greenTime > defaultMaximum:
        greenTime = defaultMaximum

    # Assign computed green time
    signals[(currentGreen + 1) % noOfSignals].green = greenTime
    print('Final Green Time:', greenTime)

def repeat():
    global currentGreen, currentYellow, nextGreen
    while(signals[currentGreen].green>0):   # while the timer of current green signal is not zero
        printStatus()
        updateValues()
        if(signals[(currentGreen+1)%(noOfSignals)].red==detectionTime):    # set time of next green signal 
            thread = threading.Thread(name="detection",target=setTime, args=())
            thread.daemon = True
            thread.start()
            # setTime()
        time.sleep(1)
    currentYellow = 1   # set yellow signal on
    vehicleCountTexts[currentGreen] = "0"
    # reset stop coordinates of lanes and vehicles 
    for i in range(0,3):
        stops[directionNumbers[currentGreen]][i] = defaultStop[directionNumbers[currentGreen]]
        for vehicle in vehicles[directionNumbers[currentGreen]][i]:
            vehicle.stop = defaultStop[directionNumbers[currentGreen]]
    while(signals[currentGreen].yellow>0):  # while the timer of current yellow signal is not zero
        printStatus()
        updateValues()
        time.sleep(1)
    currentYellow = 0   # set yellow signal off
    
    # reset all signal times of current signal to default times
    signals[currentGreen].green = defaultGreen
    signals[currentGreen].yellow = defaultYellow
    signals[currentGreen].red = defaultRed
       
    currentGreen = nextGreen # set next signal as green signal
    nextGreen = (currentGreen+1)%noOfSignals    # set next green signal
    signals[nextGreen].red = signals[currentGreen].yellow+signals[currentGreen].green    # set the red time of next to next signal as (yellow time + green time) of next signal
    repeat()     

# Print the signal timers on cmd
def printStatus():
    for i in range(noOfSignals):
        r = max(0, signals[i].red)
        y = max(0, signals[i].yellow)
        g = max(0, signals[i].green)
        
        if i == currentGreen:
            if currentYellow == 0:
                print(" GREEN TS", i+1, "-> r:", r, " y:", y, " g:", g)
            else:
                print("YELLOW TS", i+1, "-> r:", r, " y:", y, " g:", g)
        else:
            print("   RED TS", i+1, "-> r:", r, " y:", y, " g:", g)
    print()


# Update values of the signal timers after every second
# Global or class-level flag (initialize somewhere before simulation starts)
currentGreenChanged = True  # Initially True for the first green signal

def updateValues():
    global currentGreen, currentYellow,currentGreenChanged  # Ensure we can modify the flag

    for i in range(noOfSignals):
        if i == currentGreen:
            if currentYellow == 0:
                # Only decrement green if it's already set (AI value is applied once)
                if signals[i].green > 0:
                    signals[i].green -= 1
                    signals[i].totalGreenTime += 1
                # If the signal just turned green, we leave green as is (AI will set it)
            else:
                signals[i].yellow = max(0, signals[i].yellow - 1)
        else:
            signals[i].red = max(0, signals[i].red - 1)

    # Check if we need to switch to the next signal
    if signals[currentGreen].green == 0 and currentYellow == 0:
        currentYellow = 1  # Switch to yellow
    elif currentYellow == 1 and signals[currentGreen].yellow == 0:
        currentYellow = 0
        currentGreen = (currentGreen + 1) % noOfSignals
        currentGreenChanged = True  # <-- Set flag when new green starts

# Generating vehicles in the simulation
def generateVehicles():
    while(True):
        vehicle_type = random.randint(0,4)
        if(vehicle_type==4):
            lane_number = 0
        else:
            lane_number = random.randint(0,1) + 1
        will_turn = 0
        if(lane_number==2):
            temp = random.randint(0,4)
            if(temp<=2):
                will_turn = 1
            elif(temp>2):
                will_turn = 0
        temp = random.randint(0,999)
        direction_number = 0
        a = [400,800,900,1000]
        if(temp<a[0]):
            direction_number = 0
        elif(temp<a[1]):
            direction_number = 1
        elif(temp<a[2]):
            direction_number = 2
        elif(temp<a[3]):
            direction_number = 3
        Vehicle(lane_number, vehicleTypes[vehicle_type], direction_number, directionNumbers[direction_number], will_turn)
        time.sleep(0.25)

def simulationTime():
    global timeElapsed, simTime
    while(True):
        timeElapsed += 1
        time.sleep(1)
        if(timeElapsed==simTime):
            totalVehicles = 0
            print('Lane-wise Vehicle Counts')
            for i in range(noOfSignals):
                print('Lane',i+1,':',vehicles[directionNumbers[i]]['crossed'])
                totalVehicles += vehicles[directionNumbers[i]]['crossed']
            print('Total vehicles passed: ',totalVehicles)
            print('Total time passed: ',timeElapsed)
            print('No. of vehicles passed per unit time: ',(float(totalVehicles)/float(timeElapsed)))
            os._exit(1)

# ===============================================================
# =============== NEAT INTEGRATION SECTION ======================
# ===============================================================
import neat
import pickle
import sys

# ---------- NEAT Evaluation Function ----------
def eval_genomes(genomes, config):
    """
    Evaluate each genome's performance in the traffic simulation.
    Each genome predicts a green time, and fitness depends on how
    efficiently it clears vehicles from the intersection.
    """
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        # Input features (vehicle counts)
        inputs = [
            len(vehicles['right'][0]) + len(vehicles['right'][1]) + len(vehicles['right'][2]),
            len(vehicles['down'][0]) + len(vehicles['down'][1]) + len(vehicles['down'][2]),
            len(vehicles['left'][0]) + len(vehicles['left'][1]) + len(vehicles['left'][2]),
            len(vehicles['up'][0]) + len(vehicles['up'][1]) + len(vehicles['up'][2])
        ]
        inputs = [count / 50 for count in inputs]


        # Network output (predicted green time)
        output = net.activate(inputs)
        predicted_green = int(output[0] * defaultMaximum)

        # Clamp within safe range
        predicted_green = max(defaultMinimum, min(defaultMaximum, predicted_green))

        # Simulate basic traffic clearance (you can replace with real-time logic later)
        # Estimate how many vehicles could pass given this green time
        estimated_cleared = (predicted_green / defaultMaximum) * (
            noOfCars + noOfBikes + noOfBuses + noOfRickshaws + noOfTrucks
        )

        # Simple throughput vs waiting penalty
        try:
            throughput = sum(v['crossed'] for v in vehicles.values() if isinstance(v, dict))
        except Exception:
            throughput = estimated_cleared  # fallback if no detailed vehicle stats

        try:
            waiting_penalty = sum(
                len(v[lane]) for v in vehicles.values() if isinstance(v, dict) for lane in [0, 1, 2]
            )
        except Exception:
            waiting_penalty = len(vehicles)  # fallback if structure simpler

        # Base fitness: reward clearing vehicles, penalize waiting
        genome.fitness = max(1.0, throughput + estimated_cleared - 0.3 * waiting_penalty)

        # ✅ Optional reinforcement-like feedback (from dynamic simulation)
        try:
            fitness = run_simulation(net)  # optional function returning performance score
            genome.fitness = 0.5 * genome.fitness + 0.5 * fitness
        except Exception as e:
            print(f"[WARN] run_simulation failed for genome {genome_id}: {e}")

import random

def run_simulation(net):
  
    # Example: simulate how well the network might perform
    # You can replace this with actual simulation logic later.
    simulated_performance = random.uniform(0, 100)  # random fitness between 0–100

    return simulated_performance

# ---------- Run NEAT Training ----------
def run_neat(config_path="config.txt", generations=10):
    """
    Run NEAT evolution for the specified number of generations.
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    pop = neat.Population(config)
    pop.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)

    print("\n[🚦] Starting NEAT training...\n")
    winner = pop.run(eval_genomes, generations)

    with open("best_neat_model.pkl", "wb") as f:
        pickle.dump(winner, f)
    print("\n✅ NEAT training complete. Model saved as best_neat_model.pkl\n")


# ---------- Load Trained NEAT Model ----------
def get_neat_green_time():
    """
    Use the trained NEAT network to predict an adaptive green time.
    """
    try:
        with open("best_neat_model.pkl", "rb") as f:
            winner = pickle.load(f)
    except FileNotFoundError:
        print("[⚠️] NEAT model not trained yet — using rule-based timing.")
        return None  # fallback to old method

    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "config.txt"
    )
    net = neat.nn.FeedForwardNetwork.create(winner, config)

    inputs = [noOfCars, noOfBikes, noOfBuses, noOfRickshaws, noOfTrucks,noOfLanes]
    output = net.activate(inputs)
    green_time = int(output[0] * defaultMaximum)
    return max(defaultMinimum, min(defaultMaximum, green_time))


import pygame
import threading
import sys
class Main:
    def __init__(self):
        # --- Load trained NEAT model ---
        try:
            print("[🧠] Loading trained NEAT model: best_neat_model.pkl ...")
            self.net = self.load_best_model()
            print("[✅] Model loaded successfully. Using AI-controlled signals.")
        except Exception as e:
            print(f"[⚠️] Could not load trained model: {e}")
            self.net = None  # fallback to normal timing if model not found

        # --- Background threads ---
        self.thread4 = threading.Thread(name="simulationTime", target=simulationTime, daemon=True)
        self.thread4.start()

        self.thread2 = threading.Thread(name="initialization", target=initialize, daemon=True)
        self.thread2.start()

        self.thread3 = threading.Thread(name="generateVehicles", target=generateVehicles, daemon=True)
        self.thread3.start()

        # --- Colours and screen setup ---
        self.black = (0, 0, 0)
        self.white = (255, 255, 255)
        self.screenWidth, self.screenHeight = 1400, 800
        self.screenSize = (self.screenWidth, self.screenHeight)

        self.background = pygame.image.load("first.png")
        self.screen = pygame.display.set_mode(self.screenSize)
        pygame.display.set_caption("SIMULATION")

        # --- Load assets ---
        self.redSignal = pygame.image.load("images/signals/red.png")
        self.yellowSignal = pygame.image.load("images/signals/yellow.png")
        self.greenSignal = pygame.image.load("images/signals/green.png")
        self.font = pygame.font.Font(None, 30)

        # --- Input normalization caps (adjust these to realistic values) ---
        self.max_counts = {
            "cars": 50,
            "bikes": 50,
            "buses": 10,
            "rickshaws": 20,
            "trucks": 15,
            "lanes": 8
        }

        
       

    # ------------------------------------------------------------------
    def load_best_model(self):
        """Load trained NEAT model from file."""
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            "config.txt"
        )
        with open("best_neat_model.pkl", "rb") as f:
            winner = pickle.load(f)
        return neat.nn.FeedForwardNetwork.create(winner, config)

    # ------------------------------------------------------------------
    def run_simulation(self):
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            self.screen.blit(self.background, (0, 0))

            # --- AI Adaptive Signal Control ---
            global currentGreenChanged  # make sure we can modify the flag
            if self.net and currentGreenChanged:
                try:
                    inputs = [
                        noOfCars / self.max_counts["cars"],
                        noOfBikes / self.max_counts["bikes"],
                        noOfBuses / self.max_counts["buses"],
                        noOfRickshaws / self.max_counts["rickshaws"],
                        noOfTrucks / self.max_counts["trucks"],
                        noOfLanes / self.max_counts["lanes"]
                    ]
                    inputs = [max(0.0, min(1.0, i)) for i in inputs]

                    output = self.net.activate(inputs)
                    raw_output = output[0]
                    predicted_green = int(raw_output * defaultMaximum)
                    predicted_green = max(defaultMinimum, min(defaultMaximum, predicted_green))
                    
                    # Only apply once when the signal first turns green
                    signals[currentGreen].green = predicted_green
                    currentGreenChanged = False  # <- important: reset flag so we don’t overwrite every frame
                except Exception as e:
                    print(f"[WARN] AI update failed: {e}")

            # --- Draw signals ---
            for i in range(noOfSignals):
                if i == currentGreen:
                    if currentYellow == 1:
                        self.screen.blit(self.yellowSignal, signalCoods[i])
                    else:
                        self.screen.blit(self.greenSignal, signalCoods[i])
                else:
                    self.screen.blit(self.redSignal, signalCoods[i])

            # --- Draw vehicles ---
            for vehicle in simulation:
                self.screen.blit(vehicle.currentImage, [vehicle.x, vehicle.y])
                vehicle.move()

            # --- Draw timers and text ---
            for i in range(noOfSignals):
                text = self.font.render(str(signals[i].green if i == currentGreen and currentYellow == 0 else signals[i].red), True, self.white)
                self.screen.blit(text, signalTimerCoods[i])

            pygame.display.update()
            clock.tick(60)

    def run(self):
        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()

            # Draw background
            self.screen.blit(self.background, (0, 0))

            # Draw signals
            for i in range(noOfSignals):
                if i == currentGreen:
                    if currentYellow == 1:
                        self.screen.blit(self.yellowSignal, signalCoods[i])
                    else:
                        self.screen.blit(self.greenSignal, signalCoods[i])
                else:
                    self.screen.blit(self.redSignal, signalCoods[i])

            # Draw vehicles
            for vehicle_group in simulation:
                vehicle_group.move()
                vehicle_group.render(self.screen)

            # Draw timers
            for i in range(noOfSignals):
                timer_text = self.font.render(str(signals[i].green if i == currentGreen and currentYellow == 0 else signals[i].red), True, self.white)
                self.screen.blit(timer_text, signalTimerCoods[i])

            # Update display
            pygame.display.update()
            clock.tick(60)  # limit to 60 FPS


           

# ---------- Optional CLI helper ----------
if __name__ == "__main__":
    if "--train" in sys.argv:
        run_neat("config.txt", generations=20)
    else:
        pygame.init()
        main = Main()
        main.run_simulation()
