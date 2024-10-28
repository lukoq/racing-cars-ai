import numpy as np
import pygame
import time
import math
from utils import *
import neat
import torch


pygame.font.init()

TILE = scale_image(pygame.image.load("imgs/Soil_Tile.png"), 0.3)
TRACK = scale_image(pygame.image.load("imgs/track.png"), 1.0)

TRACK_BORDER = scale_image(pygame.image.load("imgs/track-border.png"), 1.0)
TRACK_BORDER_MASK = pygame.mask.from_surface(TRACK_BORDER)

FINISH = scale_image(pygame.image.load("imgs/start.png"), 0.075)
FINISH_MASK = pygame.mask.from_surface(FINISH)
FINISH_POSITION = (35, 250)

RED_CAR = scale_image(pygame.image.load("imgs/red-car.png"), 0.6)
GREEN_CAR = scale_image(pygame.image.load("imgs/green-car.png"), 0.6)

WIDTH, HEIGHT = 1100, 600
X_CENTER, Y_CENTER = WIDTH / 2, HEIGHT / 2
WIN = pygame.display.set_mode((WIDTH, HEIGHT), pygame.DOUBLEBUF)
pygame.display.set_caption("Racing Game!")

MAIN_FONT = pygame.font.SysFont("comicsans", 44)

FPS = 30
PATH = []


class AbstractCar:
    def __init__(self, max_vel, rotation_vel):
        self.img = self.IMG
        self.max_vel = max_vel
        self.vel = 0
        self.rotation_vel = rotation_vel
        self.angle = 0
        self.x, self.y = self.START_POS
        self.x, self.y = self.START_POS
        self.acceleration = 0.1
        self.crashed = False

    def rotate(self, left=False, right=False):
        if left:
            self.angle += self.rotation_vel
        elif right:
            self.angle -= self.rotation_vel

    def draw(self, win):
        rotated_image = pygame.transform.rotate(self.img, self.angle)
        # Get the rectangle of the rotated image and set its center
        new_rect = rotated_image.get_rect(center=self.img.get_rect(topleft=(self.x, self.y)).center)
        # Draw the rotated image onto the window
        win.blit(rotated_image, new_rect.topleft)

    def move_forward(self):
        self.vel = min(self.vel + self.acceleration, self.max_vel)
        self.move()

    def move_backward(self):
        self.vel = max(self.vel - self.acceleration, -self.max_vel / 2)
        self.move()

    def move(self):
        radians = math.radians(self.angle)
        vertical = math.cos(radians) * self.vel
        horizontal = math.sin(radians) * self.vel

        self.y -= vertical
        self.x -= horizontal

    def collide(self, mask, x=0, y=0):
        car_mask = pygame.mask.from_surface(self.img)
        offset = (int(self.x - x), int(self.y - y))
        poi = mask.overlap(car_mask, offset)
        return poi

    def get_img_pos(self):
        offset = (13, 24)
        img_x, img_y = self.x + offset[0], self.y + offset[1]
        return img_x, img_y

    def reset(self):
        self.x, self.y = self.START_POS
        self.angle = 0
        self.vel = 0


class PlayerCar(AbstractCar):
    IMG = RED_CAR
    START_POS = (100, 200)

    def reduce_speed(self):
        self.vel = max(self.vel - self.acceleration / 2, 0)
        self.move()

    def bounce(self):
        self.vel = -self.vel
        self.move()
        self.crashed = True

    def get_angle(self):
        return math.radians(self.angle)

    def point_to_mask_distance(self, mask, direction, max_distance=1000):
        if direction == 'FORWARD':
            angle_rad = self.get_angle() + 90 / 2 * math.pi
        elif direction == 'RIGHT':
            angle_rad = self.get_angle() + 45 / 2 * math.pi
        elif direction == 'LEFT':
            angle_rad = self.get_angle()
        elif direction == 'LIGHT_LEFT':
            angle_rad = self.get_angle() + 45 * math.pi / 180
        elif direction == 'LIGHT_RIGHT':
            angle_rad = self.get_angle() + 135 * math.pi / 180
        else:
            angle_rad = self.get_angle()
        direction_x = math.cos(-angle_rad)
        direction_y = math.sin(-angle_rad)

        # Step size for moving along the ray
        step_size = 1

        x, y = self.get_img_pos()

        # Iterate through the ray to find the first collision point
        for distance in range(0, max_distance, step_size):
            # Move the point along the ray
            check_x = int(x + direction_x * distance)
            check_y = int(y + direction_y * distance)

            # Check if the point is within the mask and collides with it
            if 0 <= check_x < mask.get_size()[0] and 0 <= check_y < mask.get_size()[1]:
                if mask.get_at((check_x, check_y)):
                    # We found a collision, return the distance and the collision point
                    return distance, (check_x, check_y)

        return max_distance, (int(x + direction_x * max_distance), int(y + direction_y * max_distance))


def draw(win, track, tile, finish):
    for x in range(0, WIDTH, 32):
        for y in range(0, HEIGHT, 32):
            win.blit(tile, (x, y))

    win.blit(track, (0, 0))
    win.blit(finish, FINISH_POSITION)


def move_player(player_car, action_index):
    # 0 - Move forward
    # 1 - Steer left
    # 2 - Steer right
    # 3 - Move backward (optional)

    moved = False

    if action_index == 0:  # Accelerate / Move forward
        moved = True
        player_car.move_forward()
    elif action_index == 1:  # Steer left
        player_car.rotate(left=True)
    elif action_index == 2:  # Steer right
        player_car.rotate(right=True)
    elif action_index == 3:  # Move backward (if applicable)
        moved = True
        player_car.move_backward()

    # If no movement (i.e., action did not move forward or backward), reduce speed
    if not moved:
        player_car.reduce_speed()


def handle_collision(player_car):
    return player_car.collide(TRACK_BORDER_MASK) is not None


def handle_finish_line_collision(player_car):
    collision_point = player_car.collide(FINISH_MASK, *FINISH_POSITION)
    return collision_point is not None


def get_car_input(car):
    car_input = []
    for direction in ('FORWARD', 'LEFT', 'RIGHT', 'LIGHT_LEFT', 'LIGHT_RIGHT'):
        distance_to_border, collision_point = car.point_to_mask_distance(TRACK_BORDER_MASK, direction)
        car_input.append(distance_to_border)
    return np.array(car_input) / 1000  # Normalize the inputs


def draw_input(car):
    for direction in ('FORWARD', 'LEFT', 'RIGHT', 'LIGHT_LEFT', 'LIGHT_RIGHT'):
        distance_to_border, collision_point = car.point_to_mask_distance(TRACK_BORDER_MASK, direction)
        pygame.draw.line(WIN, (255, 0, 0), car.get_img_pos(), collision_point, 2)
        pygame.draw.circle(WIN, (0, 0, 255), collision_point, 3)
    pygame.draw.circle(WIN, (0, 255, 0), car.get_img_pos(), 3)


def compute_progress(car):
    f_min = -math.pi
    f_max = math.pi
    position = car.get_img_pos()
    theta = math.atan2(position[1] - Y_CENTER, position[0] - X_CENTER)
    return (theta - f_min) / (f_max - f_min)  # Normalize


def eval_genomes(genomes, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # cars handling
    checkpoints = [0.1, 0.2, 0.4, 0.6, 0.8]
    cars = []
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        car = PlayerCar(4, 4)
        progress = compute_progress(car)
        cars.append({'network': net,
                     'car': car,
                     'fitness': 0,
                     'alive': True,
                     'last_progress': progress,
                     'stagnation_counter': 0,
                     'checkpoints_passed': [],
                     })

    clock = pygame.time.Clock()
    run = True

    while run:
        WIN.fill((0, 0, 0))
        draw(WIN, TRACK, TILE, FINISH)  # Draw background and track

        all_crashed = True
        for car_info in cars:
            car = car_info['car']
            net = car_info['network']

            if car_info['alive']:
                car_input = torch.tensor(get_car_input(car), device=device)
                output = net.activate(car_input.cpu().numpy())
                action_index = np.argmax(output)
                move_player(car, action_index)

                if handle_collision(car):
                    car_info['fitness'] -= 1
                    car_info['fitness'] = max(car_info['fitness'], 0)
                    car_info['alive'] = False
                elif handle_finish_line_collision(car):
                    car_info['alive'] = False
                else:
                    all_crashed = False

                current_progress = compute_progress(car)
                if current_progress > car_info['last_progress']:  # forward
                    car_info['fitness'] += (current_progress - car_info['last_progress']) * 5
                    car_info['stagnation_counter'] = 0
                    for checkpoint in checkpoints:
                        if current_progress >= checkpoint and checkpoint not in car_info['checkpoints_passed']:
                            car_info['fitness'] += 1  # bonus
                            car_info['checkpoints_passed'].append(checkpoint)

                elif current_progress < car_info['last_progress']:  # backward
                    car_info['fitness'] -= (car_info['last_progress'] - current_progress) * 0.5
                    car_info['fitness'] = max(car_info['fitness'], 0)
                else:
                    car_info['stagnation_counter'] += 1
                    if car_info['stagnation_counter'] > 30:
                        car_info['fitness'] *= 0.5
                        car_info['stagnation_counter'] = 0

                car_info['last_progress'] = current_progress
            car.draw(WIN)


        # Check for end of generation if all cars have crashed
        if all_crashed:
            run = False



        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_n:  # Press 'n' for new generation
                    run = False

        pygame.display.update()
        clock.tick(FPS)

    # Assign fitness to each genome after the generation ends
    for i, (genome_id, genome) in enumerate(genomes):
        genome.fitness = cars[i]['fitness']


def run_neat(config_path):
    # Correct the typo by using 'neat.Config' with a capital 'C'
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)
    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    # Start the NEAT evolution process
    winner = population.run(eval_genomes, 100)  # Run for up to 100 generations


if __name__ == "__main__":
    config_path = "config-feedforward"
    run_neat(config_path)
