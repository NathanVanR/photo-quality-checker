# %%
import cv2
import numpy as np
import os
import random
# %%
g def create_bad_dataset(input_path, output_path):

    images = [f for f in os.listdir(input_path) if f.endswith(('.jpg', '.png', '.jpeg'))]

    for filename in images:
        img = cv2.imread(os.path.join(input_path, filename))

        failure_mode = random.choice(['blur', 'lighting', 'saturation'])

        if failure_mode == 'blur':
            processed = cv2.GaussianBlur(img, (15, 15), 0)
            label = "BLURRY"

        elif failure_mode == 'lighting':
            processed = np.clip(img.astype(float) * 0.2, 0, 255).astype(np.uint8)
            label = "DARK"

        elif failure_mode == 'saturation':
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            s = np.clip(s.astype(float) * 0.1, 0, 255).astype(np.uint8)
            processed = cv2.cvtColor(cv2.merge([h, s, v]), cv2.COLOR_HSV2BGR)
            label = "DULL"

        new_name = f"bad_{label}_{filename}"
        cv2.imwrite(os.path.join(output_path, new_name), processed)
        print(f"Created {new_name}")

create_bad_dataset('../data/test/bad', '../data/test/bad_test')