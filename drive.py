import neat
import pygame
import os


# Define a fitness function to evaluate the performance of the cars
def eval_genomes(genomes, config):
    cars = []
    for genome_id, genome in genomes:
        # Create a neural network for each genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = 0
        cars.append((net, AbstractCar(4, 4), genome))

    run = True
    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                break

        # Update car positions based on neural network outputs
        for net, car, genome in cars:
            inputs = car.get_data()  # Get sensor data
            output = net.activate(inputs)  # Get the network's output

            # Use output to steer and move the car
            car.act(output)

            # Calculate fitness: e.g., distance traveled without collision
            genome.fitness += 1  # You can improve this formula to consider time, distance, etc.

            # Check for collisions or other stopping conditions
            if car.collide(mask):
                genome.fitness -= 100  # Penalize for crashing
                cars.remove((net, car, genome))  # Remove crashed cars

        # Redraw cars, track, etc.
        # e.g., screen.fill(BLACK), draw track, car.draw(screen), pygame.display.flip()


def run(config_file):
    # Load NEAT configuration
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_file)

    # Create NEAT population
    p = neat.Population(config)

    # Add reporters to monitor progress
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # Run NEAT
    winner = p.run(eval_genomes, 50)  # Run for 50 generations

    # Save and visualize the best network (winner)
    print('Best genome:', winner)


if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config-feedforward.txt")
    run(config_path)
