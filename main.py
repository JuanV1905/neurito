from pathlib import Path

import numpy as np
import pygame

LANGUAGE = "en"  # Change to "es" for Spanish interface text.
TEXT = {
    "en": {
        "title": "NEURITO - MY ARTIFICIAL BRAIN",
        "hunger": "Hunger", "energy": "Energy", "boredom": "Boredom",
        "eat": "EAT", "sleep": "SLEEP", "play": "PLAY", "neutral": "CALM",
        "accuracy": "Test accuracy against synthetic policy",
        "epoch": "Epoch", "loss": "Loss",
        "final_loss": "Final loss",
        "baseline": "Baseline (always the most common action)",
        "near_ties": "Accuracy in near-ties (gap < 0.02)",
        "confusion": "Confusion matrix (rows = real, columns = predicted)",
        "confidence": "Confidence", "unsure": "unsure",
        "controls1": "H/J: hunger +/-    E/D: energy +/-",
        "controls2": "A/S: boredom +/-    R: random",
        "controls3": "N: calm state    ESC: quit",
    },
    "es": {
        "title": "NEURITO - MI CEREBRO ARTIFICIAL",
        "hunger": "Hambre", "energy": "Energía", "boredom": "Aburrimiento",
        "eat": "COMER", "sleep": "DORMIR", "play": "JUGAR", "neutral": "NEUTRAL",
        "accuracy": "Precisión frente a la política sintética",
        "epoch": "Época", "loss": "Pérdida",
        "final_loss": "Pérdida final",
        "baseline": "Línea base (siempre la acción más común)",
        "near_ties": "Precisión en casi empates (diferencia < 0.02)",
        "confusion": "Matriz de confusión (filas = real, columnas = predicción)",
        "confidence": "Confianza", "unsure": "dudando",
        "controls1": "H/J: hambre +/-    E/D: energía +/-",
        "controls2": "A/S: aburrimiento +/-    R: aleatorio",
        "controls3": "N: estado tranquilo    ESC: salir",
    },
}
labels = TEXT[LANGUAGE]
actions = ("eat", "sleep", "play")

# Training settings. Results from our experiments (10 seeds, 1200 epochs):
# 3 hidden neurons is unstable (~79%), 6 is enough (~97%), 10-20 only add stability.
# Learning rate: <= 0.01 is too slow, 0.05-0.5 works, >= 1 oscillates, >= 5 collapses.
HIDDEN_NEURONS = 6
EPOCHS = 1200
LEARNING_RATE = 0.1
NEUTRAL_LIMIT = 0.4  # Display rule: all needs below this value -> calm pet.
UNSURE_LIMIT = 0.6   # Below this top probability the network is "unsure".

# The ideal action is the biggest need. Hunger and boredom grow with the need,
# but energy works the other way around, so tiredness = 1 - energy puts the
# three values in the same direction and on the same [0, 1] scale.
def priorities(data):
    hunger, energy, boredom = data[:, 0], data[:, 1], data[:, 2]
    tiredness = 1 - energy
    return np.column_stack((hunger, tiredness, boredom))


def synthetic_policy(data):
    return np.argmax(priorities(data), axis=1)


# Synthetic examples follow a simple priority policy.
np.random.seed(42)
X = np.random.rand(3000, 3)
targets = synthetic_policy(X)
# Weights start random so neurons differ from each other; biases can start at 0.
# Without biases the accuracy dropped from ~98% to ~70% in our tests.
W1 = np.random.randn(3, HIDDEN_NEURONS) * 0.5
b1 = np.zeros((1, HIDDEN_NEURONS))
W2 = np.random.randn(HIDDEN_NEURONS, 3) * 0.5
b2 = np.zeros((1, 3))


def relu(values):
    # Keeps positive values and turns negative ones into 0 (non-linearity).
    return np.maximum(0, values)


def softmax(values):
    # Turns the 3 outputs into probabilities that add up to 1.
    exp_values = np.exp(values - np.max(values, axis=1, keepdims=True))
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def predict(data):
    hidden = relu(data @ W1 + b1)
    return softmax(hidden @ W2 + b2)


# Full-batch gradient descent.
for epoch in range(EPOCHS):
    z1 = X @ W1 + b1
    hidden = relu(z1)
    probabilities = softmax(hidden @ W2 + b2)
    count = len(X)
    loss = -np.mean(np.log(probabilities[np.arange(count), targets] + 1e-10))
    delta2 = probabilities.copy()
    delta2[np.arange(count), targets] -= 1
    delta2 /= count
    grad_W2 = hidden.T @ delta2
    grad_b2 = np.sum(delta2, axis=0, keepdims=True)
    delta1 = (delta2 @ W2.T) * (z1 > 0)
    grad_W1 = X.T @ delta1
    grad_b1 = np.sum(delta1, axis=0, keepdims=True)
    W1 -= LEARNING_RATE * grad_W1
    b1 -= LEARNING_RATE * grad_b1
    W2 -= LEARNING_RATE * grad_W2
    b2 -= LEARNING_RATE * grad_b2
    if epoch % 200 == 0:
        print(f"{labels['epoch']} {epoch} | {labels['loss']}: {loss:.4f}")
print(f"{labels['final_loss']}: {loss:.4f}")

# Evaluation. A single accuracy number can hide problems, so we also print a
# baseline, the confusion matrix and the accuracy where the decision is hardest.
test = np.random.rand(1000, 3)
test_targets = synthetic_policy(test)
guesses = np.argmax(predict(test), axis=1)
accuracy = np.mean(guesses == test_targets)
print(f"{labels['accuracy']}: {accuracy:.2%}")

baseline = np.bincount(test_targets).max() / len(test_targets)
print(f"{labels['baseline']}: {baseline:.2%}")

sorted_priorities = np.sort(priorities(test), axis=1)
gap = sorted_priorities[:, 2] - sorted_priorities[:, 1]
near_tie = gap < 0.02
if near_tie.any():
    near_accuracy = np.mean(guesses[near_tie] == test_targets[near_tie])
    print(f"{labels['near_ties']}: {near_accuracy:.2%} ({near_tie.sum()})")

confusion = np.zeros((3, 3), dtype=int)
np.add.at(confusion, (test_targets, guesses), 1)
print(labels["confusion"])
for name, row in zip(actions, confusion):
    print(f"  {labels[name]:>8}: {row}")

pygame.init()
screen = pygame.display.set_mode((900, 600))
pygame.display.set_caption("Neurito")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 30)
small_font = pygame.font.SysFont(None, 23)
asset_dir = Path(__file__).resolve().parent / "assets"
pet_size = (330, 330)


def load_frames(names):
    return [
        pygame.transform.smoothscale(
            pygame.image.load(str(asset_dir / name)).convert_alpha(), pet_size
        )
        for name in names
    ]


frames = {
    "neutral": load_frames(["neutral_1.png", "neutral_2.png", "neutral_3.png"]),
    "eat": load_frames(["comer_1.png", "comer_2.png"]),
    "sleep": load_frames(["dormir_1.png", "dormir_2.png"]),
    "play": load_frames(["jugar_1.png", "jugar_2.png"]),
}
needs = [0.2, 0.9, 0.2]
current_state = "neutral"
frame_index = 0
last_frame_time = pygame.time.get_ticks()
running = True

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_h:
                needs[0] = min(1, needs[0] + 0.1)
            elif event.key == pygame.K_j:
                needs[0] = max(0, needs[0] - 0.1)
            elif event.key == pygame.K_e:
                needs[1] = min(1, needs[1] + 0.1)
            elif event.key == pygame.K_d:
                needs[1] = max(0, needs[1] - 0.1)
            elif event.key == pygame.K_a:
                needs[2] = min(1, needs[2] + 0.1)
            elif event.key == pygame.K_s:
                needs[2] = max(0, needs[2] - 0.1)
            elif event.key == pygame.K_r:
                needs = list(np.random.rand(3))
            elif event.key == pygame.K_n:
                needs = [0.2, 0.9, 0.2]

    probabilities = predict(np.array([needs]))[0]
    action = actions[int(np.argmax(probabilities))]
    confidence = float(np.max(probabilities))
    if (
        needs[0] < NEUTRAL_LIMIT
        and needs[1] > 1 - NEUTRAL_LIMIT
        and needs[2] < NEUTRAL_LIMIT
    ):
        state = "neutral"  # Display rule; the network has three outputs.
    else:
        state = action

    now = pygame.time.get_ticks()
    if state != current_state:
        current_state, frame_index, last_frame_time = state, 0, now
    elif now - last_frame_time >= 500:
        frame_index = (frame_index + 1) % len(frames[state])
        last_frame_time = now

    screen.fill((250, 245, 252))
    title = font.render(labels["title"], True, (55, 45, 75))
    screen.blit(title, (45, 28))

    colors = [(235, 125, 75), (70, 175, 115), (105, 125, 225)]
    for index, key in enumerate(("hunger", "energy", "boredom")):
        value = needs[index]
        y = 125 + index * 95
        text = small_font.render(
            f"{labels[key]}: {value * 100:.0f}%", True, (35, 40, 55)
        )
        screen.blit(text, (50, y - 27))
        pygame.draw.rect(screen, (215, 220, 230), (50, y, 290, 22), border_radius=7)
        pygame.draw.rect(
            screen, colors[index], (50, y, int(290 * value), 22), border_radius=7
        )

    pet = frames[state][frame_index]
    screen.blit(pet, pet.get_rect(center=(650, 235)))
    screen.blit(font.render(labels[state], True, (65, 50, 85)), (585, 420))
    # Close priorities give a spread-out Softmax, so we show when the network is unsure.
    confidence_text = f"{labels['confidence']}: {confidence * 100:.0f}%"
    if confidence < UNSURE_LIMIT:
        confidence_text += f" ({labels['unsure']})"
    screen.blit(small_font.render(confidence_text, True, (70, 70, 90)), (585, 452))
    for index, action_name in enumerate(actions):
        score = small_font.render(
            f"{labels[action_name]}: {probabilities[index] * 100:.1f}%",
            True, (55, 50, 75)
        )
        screen.blit(score, (50, 390 + index * 28))

    screen.blit(small_font.render(labels["controls1"], True, (70, 70, 90)), (50, 495))
    screen.blit(small_font.render(labels["controls2"], True, (70, 70, 90)), (50, 525))
    screen.blit(small_font.render(labels["controls3"], True, (70, 70, 90)), (50, 555))
    pygame.display.flip()
    clock.tick(30)

pygame.quit()
