# ------------------------------------------------------------------------
# coding=utf-8
# ------------------------------------------------------------------------
#
#  Created by Martin J. Laubach on 2011-11-15
#
# ------------------------------------------------------------------------

from __future__ import absolute_import

import random
import math
import bisect

from draw import Maze

"""
# Smaller maze

maze_data = ( ( 2, 0, 1, 0, 0 ),
              ( 0, 0, 0, 0, 1 ),
              ( 1, 1, 1, 0, 0 ),
              ( 1, 0, 0, 0, 0 ),
              ( 0, 0, 2, 0, 1 ))
"""

# 0 - empty square, space can be travel
# 1 - filled square, block cant be travel
# 2 - filled square with a beacon at each corner, detectable by the robot(source)

#environment the turtle is placed
maze_data = ( ( 1, 1, 0, 0, 2, 0, 0, 0, 0, 1 ),
              ( 1, 2, 0, 0, 1, 1, 0, 0, 0, 0 ),
              ( 0, 1, 1, 0, 0, 0, 0, 1, 0, 1 ),
              ( 0, 0, 0, 0, 1, 0, 0, 1, 1, 2 ),
              ( 1, 1, 0, 1, 1, 2, 0, 0, 1, 0 ),
              ( 1, 1, 1, 0, 1, 1, 1, 0, 2, 0 ),
              ( 2, 0, 0, 0, 0, 0, 0, 0, 0, 0 ),
              ( 1, 2, 0, 1, 1, 1, 1, 0, 0, 0 ),
              ( 0, 0, 0, 0, 1, 0, 0, 0, 1, 0 ),
              ( 0, 0, 1, 0, 0, 2, 1, 1, 1, 0 ))

PARTICLE_COUNT = 2000    # Total number of particles, these particles are randomly distribute in the map at first intance,
                         # more particles  means the robot have more information to trace the true position of the robot(turtle)
                         

                    #whole concept first:   #btw, robot and turtle are the same for convention :> 
                         # by observing how likely each particle is relate to the robot/turtle, giving it weight(higher weight means more alike)
                         # and resampling, dropout the unlikely particle and resample particles in the segment/region that have higher weight.
                         # At this phase, our turtle move, so particles now move with same vector as turtle does.(updating each particles' weights)
                         # After finished moving, we get new particles distribution in a new segment/region, calculate its' weights and resampling,
                         # the turtle moving again, then repeat the step until the distribution of particles converge, at this time the gery cluster
                         # can find the turtle's location. *reminder: The particles distribution may be diverge again, but not in this program i guess.

     
ROBOT_HAS_COMPASS = True # Does the robot know where north is? If so, it
# makes orientation a lot easier since it knows which direction it is facing.
# If not -- and that is really fascinating -- the particle filter can work
# out its heading too, it just takes more particles and more time. Try this
# with 3000+ particles, it obviously needs lots more hypotheses as a particle
# now has to correctly match not only the position but also the heading.

# I guess the heading will change after turtle hit the wall 

# ------------------------------------------------------------------------
# Some utility functions for adding noise to control the error between agent to particle(environment), these functions are 
# all defined by user, as long as the bias and variance wrong getting too big then should be fine

def add_noise(level, *coords):
    return [x + random.uniform(-level, level) for x in coords]

def add_little_noise(*coords):
    return add_noise(0.02, *coords)

def add_some_noise(*coords):
    return add_noise(0.1, *coords)

# This is just a gaussian kernel I pulled out of my hat, to transform
# values near to robbie's measurement => 1, further away => 0
sigma2 = 0.9 ** 2
def w_gauss(a, b): #gaussian distribution formula, greater sigma makes the curve looks flat, lower sigma makes the curve more peak
    error = a - b
    g = math.e ** -(error ** 2 / (2 * sigma2))
    return g

# ------------------------------------------------------------------------
def compute_mean_point(particles):
    """
    Compute the mean for all particles that have a reasonably good weight.
    This is not part of the particle filter algorithm but rather an
    addition to show the "best belief" for current position.
    """

    m_x, m_y, m_count = 0, 0, 0
    for p in particles:
        m_count += p.w
        m_x += p.x * p.w
        m_y += p.y * p.w

    if m_count == 0:
        return -1, -1, False

    m_x /= m_count #compute x_mean
    m_y /= m_count #compute y_mean

    # Now compute how good that mean is -- check how many particles
    # actually are in the immediate vicinity
    m_count = 0
    for p in particles:
        if world.distance(p.x, p.y, m_x, m_y) < 1:
            m_count += 1

    return m_x, m_y, m_count > PARTICLE_COUNT * 0.95 # i dun know why using 0.95 as threshold, but here just use 95%
                                                     # as the confidence represent how reliable these particles are locate vicinity(cluster)

# ------------------------------------------------------------------------
class WeightedDistribution(object):
    def __init__(self, state):
        accum = 0.0
        self.state = [p for p in state if p.w > 0]
        self.distribution = []
        for x in self.state:
            accum += x.w
            self.distribution.append(accum)

    def pick(self):
        try:
            return self.state[bisect.bisect_left(self.distribution, random.uniform(0, 1))]
        except IndexError:
            # Happens when all particles are improbable w=0
            return None

# ------------------------------------------------------------------------
class Particle(object):
    def __init__(self, x, y, heading=None, w=1, noisy=False):
        if heading is None:
            heading = random.uniform(0, 360) #randomly give a direction, same as what I assume at the beginning of the program
        if noisy:
            x, y, heading = add_some_noise(x, y, heading)

        self.x = x
        self.y = y
        self.h = heading
        self.w = w

    def __repr__(self):
        return "(%f, %f, w=%f)" % (self.x, self.y, self.w) #particle object x_coordinate, y_coordinate, weight object

    @property #encapsulation, only callable
    def xy(self):
        return self.x, self.y

    @property #encapsulation, only callable
    def xyh(self):
        return self.x, self.y, self.h

    @classmethod #特殊寫法 我也不是很懂 只知道cls可以取代self
    def create_random(cls, count, maze):
        return [cls(*maze.random_free_place()) for _ in range(0, count)]

    def read_sensor(self, maze):
        """
        Find distance to nearest beacon.
        """
        return maze.distance_to_nearest_beacon(*self.xy)

    def advance_by(self, speed, checker=None, noisy=False): #moving turtle and particle with same direction
        h = self.h
        if noisy: #create noisy when moving, moving may cause error so here adding error into particles
            speed, h = add_little_noise(speed, h)
            h += random.uniform(-3, 3) # needs more noise to disperse better
        r = math.radians(h) #sensor radian
        dx = math.sin(r) * speed
        dy = math.cos(r) * speed
        if checker is None or checker(self, dx, dy): #if checker is none, won't move
            self.move_by(dx, dy)
            return True
        return False

    def move_by(self, x, y): # update particles
        self.x += x
        self.y += y

# ------------------------------------------------------------------------
class Robot(Particle): #initialize robot with speed 0.2, random_direction, distance to nearest beacon. Although move() 
                       #function say the movement is stochastic, but I think particles still follow turtle's movement. 
    speed = 0.2

    def __init__(self, maze):
        super(Robot, self).__init__(*maze.random_free_place(), heading=90)
        self.chose_random_direction()
        self.step_count = 0

    def chose_random_direction(self):
        heading = random.uniform(0, 360)
        self.h = heading

    def read_sensor(self, maze):
        """
        Poor robot, it's sensors are noisy and pretty strange,
        it only can measure the distance to the nearest beacon(!)
        and is not very accurate at that too!
        """
        return add_little_noise(super(Robot, self).read_sensor(maze))[0] #distance between turtle and beacon

    def move(self, maze):
        """
        Move the robot. Note that the movement is stochastic too.
        """
        while True:
            self.step_count += 1
            if self.advance_by(self.speed, noisy=True,
                checker=lambda r, dx, dy: maze.is_free(r.x+dx, r.y+dy)):
                break
            # Bumped into something or too long in same direction,
            # chose random new direction
            self.chose_random_direction() #well, i didn't see any sentimental functionality in here, I guess it 
                                          #will be triggered when bumped into wall right? as author mention above

# ------------------------------------------------------------------------

world = Maze(maze_data)
world.draw()

# initial distribution assigns each particle an equal probability
particles = Particle.create_random(PARTICLE_COUNT, world)
robbie = Robot(world)

while True:
    # Read robbie's sensor
    r_d = robbie.read_sensor(world)

    # Update particle weight according to how good every particle matches
    # robbie's sensor reading
    for p in particles:
        if world.is_free(*p.xy): #not sure what is world.is_free() actually means in real word and in the program
            p_d = p.read_sensor(world)
            p.w = w_gauss(r_d, p_d) #compute Euclidean distance between particle and robot
        else:
            p.w = 0

    # ---------- Try to find current best estimate for display ----------
    m_x, m_y, m_confident = compute_mean_point(particles)

    # ---------- Show current state ----------
    world.show_particles(particles)
    world.show_mean(m_x, m_y, m_confident)
    world.show_robot(robbie)

    # ---------- Shuffle particles ----------
    new_particles = []

    # Normalise weights
    nu = sum(p.w for p in particles)
    if nu:
        for p in particles:
            p.w = p.w / nu

    # create a weighted distribution, for fast picking
    dist = WeightedDistribution(particles)

    for _ in particles:
        p = dist.pick()
        if p is None:  # No pick b/c all totally improbable
            new_particle = Particle.create_random(1, world)[0]
        else:
            new_particle = Particle(p.x, p.y,
                    heading=robbie.h if ROBOT_HAS_COMPASS else p.h,
                    noisy=True)
        new_particles.append(new_particle)

    particles = new_particles

    # ---------- Move things ----------
    old_heading = robbie.h
    robbie.move(world)
    d_h = robbie.h - old_heading #delta h, move particles with same speed and distance respect to turtle/robot

    # Move particles according to my belief of movement (this may
    # be different than the real movement, but it's all I got)
    for p in particles:
        p.h += d_h # in case robot changed heading, swirl particle heading too
        p.advance_by(robbie.speed)
