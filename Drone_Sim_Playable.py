import pygame
import sys
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
Color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
BLACK = (0, 0, 0)
FPS = 60

# Create the window
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption('Simple Pygame Simulation')

# Character (drone) properties
character_width, character_height = 20, 20
character_x, character_y = WIDTH // 2, HEIGHT // 2
character_speed = 5
last_direction = 0  # Store the last moving direction
score = 0

clock = pygame.time.Clock()

# Function to convert degrees to radians
def new_objective():
    X = random.randint(0,800)
    y = random.randint(0,600)
    objective = [X,y]
    return objective

def degrees_to_radians(degrees):
    return math.radians(degrees)

# Function to draw a line (laser) from character in a specific direction
def draw_laser(x, y, angle, length):
    end_x = x + length * math.cos(degrees_to_radians(angle))
    end_y = y - length * math.sin(degrees_to_radians(angle))  # Negative because y increases downward in Pygame

    # Check the window boundaries for the laser drawing
    end_x = max(0, min(end_x, WIDTH))
    end_y = max(0, min(end_y, HEIGHT))

    # Iterate through each point on the line
    for i in range(int(length)):
        sample_x = int(x + i * math.cos(degrees_to_radians(angle)))
        sample_y = int(y - i * math.sin(degrees_to_radians(angle)))

        # Check for collision with white pixels
        if 0 <= sample_x < WIDTH and 0 <= sample_y < HEIGHT:
            if screen.get_at((sample_x, sample_y)) == (255, 255, 255, 255):
                # Draw the laser up to the intersection point
                pygame.draw.line(screen, (255, 0, 0), (x, y), (sample_x, sample_y), 2)
                return math.sqrt((sample_x - x) ** 2 + (sample_y - y) ** 2)

    # Draw the laser to the end of the line if no intersection is found
    pygame.draw.line(screen, (255, 0, 0), (x, y), (end_x, end_y), 2)
    return length  # No intersection found, return the length of the laser

# Main game loop
running = True
lasers = []  # List to store the laser distances
objective = new_objective()
while running:
    screen.fill(BLACK)

    pygame.draw.rect(screen, (255,255,255), (200,200,100,100))

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    keys = pygame.key.get_pressed()
    character_direction = 0

    # Capture the distances of intersections for all lasers
    distances = []
    for angle in range(-90, 91, 45):
        updated_angle = last_direction + angle  # Use the last direction
        laser_distance = draw_laser(character_x + character_width // 2, character_y + character_height // 2, updated_angle, 100)
        distances.append(round(laser_distance, 2))
    lasers = distances
    print(lasers[0])


    if keys[pygame.K_LEFT]:
        if lasers[3]-3 > 5:
            character_x -= character_speed
            last_direction = 180  # Store the direction
        else:
             character_x += character_speed*5
    elif keys[pygame.K_RIGHT]:
        if lasers[3]-3 > 5:
            character_x += character_speed
            last_direction = 0  # Store the direction
        else:
             character_x -= character_speed*5
    elif keys[pygame.K_UP]:
        if lasers[3]-3 > 5:
            character_y -= character_speed
            last_direction = 90  # Store the direction
        else:
             character_y += character_speed*5
    elif keys[pygame.K_DOWN]:
        if lasers[3]-3 > 5:
            character_y += 5
            last_direction = 270  # Store the direction
        else:
             character_y -= character_speed*5

    # Ensure character stays within the window boundaries
    character_x = max(0, min(WIDTH - character_width, character_x))
    character_y = max(0, min(HEIGHT - character_height, character_y))

    if character_x >=objective[0]-20 and character_x <= objective[0]+20 and character_y >= objective[1]-20 and character_y <=objective[1]+20:
        objective = new_objective()
        score +=1

    

    # Draw the character (drone)
    
    pygame.draw.rect(screen, Color, (character_x, character_y, character_width, character_height))
    pygame.draw.circle(screen, (0,100,200), (objective[0], objective[1]), 10)

    # Display the values of the lasers
    for i, distance in enumerate(distances):
        text = f"Laser {i + 1}: {distance}"
        font = pygame.font.Font(None, 24)
        text_surface = font.render(text, True, (255, 255, 255))
        screen.blit(text_surface, (10, 10 + i * 20))

    pygame.display.update()
    clock.tick(FPS)

pygame.quit()
sys.exit()