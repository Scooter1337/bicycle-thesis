#!/usr/bin/env python3
"""
Display the NLL comparison graph.
"""

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the saved image
img = mpimg.imread('/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/nll_comparison.png')

plt.figure(figsize=(12, 8))
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.title('NLL Comparison Graph with Trend Line')
plt.tight_layout()
plt.savefig('/Users/luca/Developer/Universiteit/leiden-university/bachelor-project/nll_display.png', dpi=300, bbox_inches='tight')
plt.close()

print("Graph display saved to: /Users/luca/Developer/Universiteit/leiden-university/bachelor-project/nll_display.png")
