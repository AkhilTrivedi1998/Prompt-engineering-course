import pygame
import time
import random

pygame.init()

# Colors
BACKGROUND_COLOR = (50, 153, 213)  # Background color
SNAKE_COLOR = (0, 0, 0)  # Snake color
FOOD_COLOR = (0, 255, 0)  # Food color
TEXT_COLOR = (213, 50, 80)  # End game text color
SCORE_COLOR = (255, 255, 255)  # Score color

# Screen dimensions
SCREEN_WIDTH = 600
SCREEN_HEIGHT = 400

# Game parameters
snake_block = 10

# Set up display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Snake Game")

# Set up clock
clock = pygame.time.Clock()

# Font
font_style = pygame.font.SysFont("comicsansms", 35)  # Modify the font style and size here
score_font = pygame.font.SysFont("comicsansms", 25)  # Modify the font style and size for the score

def Your_score(score):
    value = score_font.render("Your Score: " + str(score), True, SCORE_COLOR)
    screen.blit(value, [0, 0])

def our_snake(snake_block, snake_list):
    for x in snake_list:
        pygame.draw.rect(screen, SNAKE_COLOR, [x[0], x[1], snake_block, snake_block])

def message(msg, color):
    mesg = font_style.render(msg, True, color)
    text_rect = mesg.get_rect(center=(SCREEN_WIDTH / 2, SCREEN_HEIGHT / 3))  # Center text
    screen.blit(mesg, text_rect)

def gameLoop():
    game_over = False
    game_close = False

    x1 = SCREEN_WIDTH / 2
    y1 = SCREEN_HEIGHT / 2

    x1_change = 0
    y1_change = 0

    snake_List = []
    Length_of_snake = 1

    foodx = round(random.randrange(0, SCREEN_WIDTH - snake_block) / 10.0) * 10.0
    foody = round(random.randrange(0, SCREEN_HEIGHT - snake_block) / 10.0) * 10.0

    snake_speed = 15  # Define snake_speed inside the game loop

    while not game_over:

        while game_close == True:
            screen.fill(BACKGROUND_COLOR)
            message("Game Over! Press Q to Quit or C to Play Again", TEXT_COLOR)
            Your_score(Length_of_snake - 1)
            pygame.display.update()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_q:
                        game_over = True
                        game_close = False
                    if event.key == pygame.K_c:
                        gameLoop()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                game_over = True
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT:
                    x1_change = -snake_block
                    y1_change = 0
                elif event.key == pygame.K_RIGHT:
                    x1_change = snake_block
                    y1_change = 0
                elif event.key == pygame.K_UP:
                    y1_change = -snake_block
                    x1_change = 0
                elif event.key == pygame.K_DOWN:
                    y1_change = snake_block
                    x1_change = 0

        if x1 >= SCREEN_WIDTH or x1 < 0 or y1 >= SCREEN_HEIGHT or y1 < 0:
            game_close = True
        x1 += x1_change
        y1 += y1_change
        screen.fill(BACKGROUND_COLOR)
        pygame.draw.rect(screen, FOOD_COLOR, [foodx, foody, snake_block, snake_block])
        snake_Head = []
        snake_Head.append(x1)
        snake_Head.append(y1)
        snake_List.append(snake_Head)
        if len(snake_List) > Length_of_snake:
            del snake_List[0]

        for x in snake_List[:-1]:
            if x == snake_Head:
                game_close = True

        our_snake(snake_block, snake_List)
        Your_score(Length_of_snake - 1)
        pygame.display.update()

        if x1 == foodx and y1 == foody:
            foodx = round(random.randrange(0, SCREEN_WIDTH - snake_block) / 10.0) * 10.0
            foody = round(random.randrange(0, SCREEN_HEIGHT - snake_block) / 10.0) * 10.0
            Length_of_snake += 1
            if Length_of_snake % 5 == 0:
                snake_speed += 1  # Increase snake_speed as the length of the snake increases

        clock.tick(snake_speed)

    pygame.quit()
    quit()

gameLoop()
