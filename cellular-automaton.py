import numpy as np
from PIL import Image, ImageDraw, ImageFont
import imageio
import os
import random

GRID_WIDTH = 50
GRID_HEIGHT = 50
CELL_SIZE = 32
NUM_FRAMES = 200
OUTPUT_DIR = "frames"
SCENARIO_TEMP = 1.5
IMAGE_PATH = "Figure2.6WorldMap1.5Celsius.png"
OCEAN_STATE = -1



# - EMOJI_MAP: dictionary mapping ecosystem states to emoji icons
#       0 → healthy (🌿)
#       1 → mild decline (🍃)
#       2 → moderate decline (🍂)
#       3 → heavy decline (🔥)
#       4 → collapse (💀)
EMOJI_MAP = {
    0: "🌿",
    1: "🍃",
    2: "🍂",
    3: "🔥",
    4: "💀"
}


# List defining pressure values at which a cell transitions
THRESHOLDS = [0.4, 0.9, 1.8, 3.2]

START_TEMP = 1.0 
END_TEMP = 4.0 
START_YEAR = 2020
END_YEAR = 2100

# Takes an RGB pixel and classifies it into an ecosystem state
def classify_pixel(pixel):
    r, g, b = map(int, pixel)
    
    # Detect neutral gray (ocean)
    if abs(r - g) < 8 and abs(r - b) < 8 and abs(g - b) < 8:
        return OCEAN_STATE

    centers = np.array([
        [242.7835, 239.4078, 183.3189],  # cluster 0
        [243.9324, 200.3258, 101.5311],  # cluster 1
        [240.1163, 239.8456, 230.8848],  # cluster 2
        [219.6588, 217.4619, 193.4606],  # cluster 3
        [239.9613, 222.3625, 145.1170]   # cluster 4
    ])
    
    dists = np.sum((centers - np.array([r, g, b]))**2, axis=1)
    cluster = np.argmin(dists)

    cluster_to_state = {
        2: 0,  # very pale → healthy
        0: 0,  # light yellow → healthy
        3: 1,  # yellow-brown → mild decline
        4: 2,  # orange → moderate decline
        1: 3   # deep orange/red → severe decline
    }
    
    return cluster_to_state[cluster]

# Converts the input image to a grid of ecosystem states
def convert_image_to_states():
    img = Image.open(IMAGE_PATH).convert("RGB")
    img_small = img.resize((GRID_WIDTH, GRID_HEIGHT), Image.LANCZOS)
    pixels = np.array(img_small)
    states = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=int)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            states[y, x] = classify_pixel(pixels[y, x])
    return states

# Initializes pressures based on initial states
def init_pressures_from_states(states):
    p = np.zeros((GRID_HEIGHT, GRID_WIDTH), dtype=float)
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            s = states[y, x]
            if s <= 0:
                p[y, x] = 0.0
            else:
                p[y, x] = THRESHOLDS[s - 1] * 0.1
    return p

emojiStates = convert_image_to_states()
land_mask = (emojiStates != OCEAN_STATE)
emojiStates[land_mask] = np.maximum(emojiStates[land_mask] - 1, 0)
pressures = init_pressures_from_states(emojiStates)

# - Ensure OUTPUT_DIR exists
os.makedirs(OUTPUT_DIR, exist_ok=True)
font_path = "fonts/seguiemj.ttf"
font = ImageFont.truetype(font_path, CELL_SIZE - 4)


def get_temperature(frame):
    return START_TEMP + (END_TEMP - START_TEMP) * (frame / NUM_FRAMES)

def get_year(frame):
    return int(START_YEAR + (END_YEAR - START_YEAR) * (frame / NUM_FRAMES))

def temp_to_pressure(T):
    # tune these values as needed
    min_rate = 0.00006
    max_rate = 0.00035
    return min_rate + (max_rate - min_rate) * ((T - START_TEMP) / (END_TEMP - START_TEMP))

def step(current_temp):
    new_states = np.copy(emojiStates)
    new_pressures = np.copy(pressures)

    temp_pressure = temp_to_pressure(current_temp)

    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            if emojiStates[y, x] == OCEAN_STATE:
                new_states[y, x] = OCEAN_STATE
                new_pressures[y, x] = 0.0
                continue
            # Apply temperature-driven pressure
            new_pressures[y, x] += temp_pressure

            # Neighbor diffusion effect
            for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
                ny, nx = y + dy, x + dx
                if 0 <= ny < GRID_HEIGHT and 0 <= nx < GRID_WIDTH:
                    if emojiStates[ny, nx] >= 2:
                        new_pressures[y, x] += 0.005
            # Random disturbances
            if random.random() < 0.0002:
                new_pressures[y, x] += random.uniform(0.2, 0.5)

            # Threshold transitions
            for level, threshold in enumerate(THRESHOLDS):
                if new_pressures[y, x] >= threshold and new_states[y, x] == level:
                    new_states[y, x] = level + 1

    # Update global states
    emojiStates[:, :] = new_states
    pressures[:, :] = new_pressures


# ------------------------------------------------
# 5. RENDERING FUNCTION
# ------------------------------------------------
def render(frame_number, current_temp, current_year):
    legend_height = 200
    horizontal_padding = 40 

    img_width = GRID_WIDTH * CELL_SIZE + horizontal_padding
    img_height = GRID_HEIGHT * CELL_SIZE + legend_height

    img = Image.new("RGB", (img_width, img_height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # ===== Draw the grid of emojis =====
    for y in range(GRID_HEIGHT):
        for x in range(GRID_WIDTH):
            state = emojiStates[y, x]
            if state == OCEAN_STATE:
                continue
            emoji = EMOJI_MAP[state]
            draw.text((x * CELL_SIZE + 10, y * CELL_SIZE + 10), emoji, font=font, fill=(0,0,0))

    # ===== Draw labels (year + temperature) =====
    overlay_y = GRID_HEIGHT * CELL_SIZE + 10
    draw.text((10, overlay_y), f"Year: {current_year}", font=font, fill=(0, 0, 0))
    draw.text((200, overlay_y), f"Warming: {current_temp:.2f} °C", font=font, fill=(0, 0, 0))

    # ===== Legend =====
    legend_items = [
        ("🌿", "Healthy / <25% loss"),
        ("🍃", "Mild decline"),
        ("🍂", "Moderate decline"),
        ("🔥", "Severe decline"),
        ("💀", "Collapse / >75% loss")
    ]

    legend_start_y = GRID_HEIGHT * CELL_SIZE + 50
    legend_x = 10

    for emoji, label in legend_items:
        draw.text((legend_x, legend_start_y), f"{emoji}  {label}",
                  font=font, fill=(0, 0, 0))
        legend_start_y += 28

    img.save(os.path.join(OUTPUT_DIR, f"frame_{frame_number:04d}.png"))

# 6. MAIN SIMULATION LOOP
for frame in range(NUM_FRAMES):
    current_temp = get_temperature(frame)
    current_year = get_year(frame)

    print(f"Frame {frame+1}/{NUM_FRAMES} — Temp: {current_temp:.2f} °C — Year: {current_year}")

    render(frame, current_temp, current_year)
    step(current_temp)


# ------------------------------------------------
# 7. EXPORT ANIMATION
# ------------------------------------------------
images = []
for frame in range(NUM_FRAMES):
    filename = os.path.join(OUTPUT_DIR, f"frame_{frame:04d}.png")
    images.append(imageio.imread(filename))
imageio.mimsave('biodiversity_ca.gif', images, fps=10)


# ------------------------------------------------
# 8. OPTIONAL ENHANCEMENTS (FOR LATER)
# ------------------------------------------------
# - Import a world map mask to simulate only land cells
# - Add noise textures to simulation for more organic behavior
# - Add borders or gridlines for a more stylized appearance
# - Export per-step statistics (average decline, collapse ratio, etc.)
# - Add labels to images (year, temp scenario, timestep)
#
