import pygame
import sys
import random
import math
import tensorflow as tf

class Simulation:
    def __init__(self):
        # Initialize Pygame
        pygame.init()

        # Constants
        self.WIDTH, self.HEIGHT = 800, 600
        self.Color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.BLACK = (0, 0, 0)
        self.FPS = 600

        # Create the window
        self.screen = pygame.display.set_mode((self.WIDTH, self.HEIGHT), pygame.NOFRAME)
        pygame.display.set_caption('Simple Pygame Simulation')

        # Character (drone) properties
        self.character_width, self.character_height = 20, 20
        self.character_x, self.character_y = self.WIDTH // 2, self.HEIGHT // 2
        self.character_speed = 5
        self.last_direction = 0  # Store the last moving direction
        self.score = 0
        self.lasers = []
        self.lastpos = [self.character_x,self.character_y]
        self.reward = 0

        self.clock = pygame.time.Clock()

        self.objective = self.new_objective()

        for i in range(-90, 91, 45):
            updated_angle = self.last_direction + i
            laser_distance = self.draw_laser(
                self.character_x + self.character_width // 2,
                self.character_y + self.character_height // 2,
                updated_angle,
                100
            )
            self.lasers.append(round(laser_distance, 2))

    def new_objective(self):
        print("OJECTIVE REACHED")
        print("OJECTIVE REACHED")
        print("OJECTIVE REACHED")
        print("OJECTIVE REACHED")
        print("SCORE = ",self.score)
        X = random.randint(0, self.WIDTH)
        y = random.randint(0, self.HEIGHT)
        return [X, y]

    def draw_laser(self, x, y, angle, length):
        end_x = x + length * math.cos(math.radians(angle))
        end_y = y - length * math.sin(math.radians(angle))

        end_x = max(0, min(end_x, self.WIDTH))
        end_y = max(0, min(end_y, self.HEIGHT))

        for i in range(int(length)):
            sample_x = int(x + i * math.cos(math.radians(angle)))
            sample_y = int(y - i * math.sin(math.radians(angle)))

            if 0 <= sample_x < self.WIDTH and 0 <= sample_y < self.HEIGHT:
                if self.screen.get_at((sample_x, sample_y)) == (255, 255, 255, 255):
                    pygame.draw.line(self.screen, (255, 0, 0), (x, y), (sample_x, sample_y), 2)
                    return math.sqrt((sample_x - x) ** 2 + (sample_y - y) ** 2)

        pygame.draw.line(self.screen, (255, 0, 0), (x, y), (end_x, end_y), 2)
        return length

    def check_objective(self):
        if (
            self.character_x >= self.objective[0] - 20
            and self.character_x <= self.objective[0] + 20
            and self.character_y >= self.objective[1] - 20
            and self.character_y <= self.objective[1] + 20
        ):
            self.objective = self.new_objective()
            self.score += 1

    def take_action(self, action):
        if action == 0:  # Move left
            self.lastpos[0] = self.character_x
            self.character_x -= self.character_speed
            self.last_direction = 180
        elif action == 1:  # Move right
            self.lastpos[0] = self.character_x
            self.character_x += self.character_speed
            self.last_direction = 0
        elif action == 2:  # Move up
            self.lastpos[1] = self.character_y
            self.character_y -= self.character_speed
            self.last_direction = 90
        elif action == 3:  # Move down
            self.lastpos[1] = self.character_y
            self.character_y += self.character_speed
            self.last_direction = 270

        self.character_x = max(0, min(self.WIDTH - self.character_width, self.character_x))
        self.character_y = max(0, min(self.HEIGHT - self.character_height, self.character_y))

    def get_state(self):
        state = (
            self.lasers[0],
            self.lasers[1],
            self.lasers[2],
            self.lasers[3],
            self.lasers[4],
            self.character_x,
            self.character_y,
            self.objective[0],
            self.objective[1]
        )
        return tf.convert_to_tensor(state, dtype=tf.float32)[tf.newaxis, :]  # Add a new axis
    
    def get_reward(self):
        self.reward = 0
        #if the distance between character now is smaller than distance character before, good
        try:
            self.distance = math.sqrt((self.character_x - self.objective[0])**2 + (self.character_y - self.objective[1])**2)
            if math.sqrt((self.character_x - self.objective[0])**2 + (self.character_y - self.objective[1])**2) < math.sqrt((self.lastpos[0] - self.objective[0])**2 + (self.lastpos[1] - self.objective[1])**2):
                self.reward += (1/self.distance) * 100
            elif self.distance == math.sqrt((self.lastpos[0] - self.objective[0])**2 + (self.lastpos[1] - self.objective[1])**2):
                self.reward = 0
            else:
                self.reward -= (1/self.distance) * 100
        except:
            self.reward = 0
            
        return self.reward
        

    def step(self, action):
        self.screen.fill(self.BLACK)


        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        self.take_action(action)  # Move the character
        self.check_objective()  # Check if the objective is reached

        # Calculate the direction of the lasers based on the character's last movement direction
        self.lasers = []
        for i in range(-90, 91, 45):
            updated_angle = self.last_direction + i
            laser_distance = self.draw_laser(
                self.character_x + self.character_width // 2,
                self.character_y + self.character_height // 2,
                updated_angle,
                100
            )
            self.lasers.append(round(laser_distance, 2))
    

        pygame.draw.rect(self.screen, self.Color, (self.character_x, self.character_y, self.character_width, self.character_height))
        pygame.draw.circle(self.screen, (0, 100, 200), (self.objective[0], self.objective[1]), 10)

        pygame.display.update()
        self.clock.tick(self.FPS)
        return self.get_state(), self.get_reward(), False