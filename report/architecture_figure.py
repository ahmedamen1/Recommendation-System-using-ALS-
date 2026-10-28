"""Build a publication-ready vector diagram of the complete serving system."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch

plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42,
                     "svg.fonttype": "none"})
HERE = Path(__file__).resolve().parent
fig, ax = plt.subplots(figsize=(7.0, 3.9))
ax.set(xlim=(0, 16), ylim=(0, 8.9))
ax.axis("off")
navy, teal, gold, ink = "#254b66", "#246454", "#a57823", "#203341"

def panel(x, y, w, h, fill, edge, lw=.8):
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                 boxstyle="round,pad=0.04,rounding_size=0.12",
                 facecolor=fill, edgecolor=edge, linewidth=lw, zorder=1))

def node(x, y, number, title, body, color, fill="#ffffff"):
    panel(x, y, 3.45, 1.85, fill, color)
    ax.add_patch(Circle((x+.30, y+1.48), .19, facecolor=color, edgecolor="none", zorder=3))
    ax.text(x+.30, y+1.48, str(number), ha="center", va="center", fontsize=7,
            weight="bold", color="white", zorder=4)
    ax.text(x+.61, y+1.48, title, fontsize=8.3, weight="bold", color=color, va="center")
    ax.text(x+.18, y+1.12, body, fontsize=7.2, color=ink, va="top", linespacing=1.55)

def arrow(a, b, color, style="-", size=9):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=size,
                                color=color, lw=1.15, linestyle=style, zorder=4))

# Left-to-right reading order is the same in both lanes.
ax.text(.25, 8.45, "OFFLINE", fontsize=9, weight="bold", color=navy)
ax.text(2.3, 8.45, "Train and evaluate once; export a compact model", fontsize=8, color=ink)
node(.25, 6.05, 1, "MovieLens 32M", "32 million ratings\nMovie titles + genres", navy, "#f2f6fa")
node(4.25, 6.05, 2, "Prepare data", "Map IDs; sparse indexes\nSeeded 80/20 split", navy, "#f2f6fa")
node(8.25, 6.05, 3, "Learn & evaluate", "ALS / Genre ALS\nTrain-only updates\nTest RMSE + cold start", navy, "#f2f6fa")
node(12.25, 6.05, 4, "Refit & export", "Refit on all ratings\nVectors, biases, counts\nCatalog + settings", gold, "#fff8e9")
for left in (.25, 4.25, 8.25):
    arrow((left+3.5, 6.95), (left+3.94, 6.95), navy)

# Dedicated handoff lane prevents crossing the training or request paths.
ax.plot([13.95, 13.95, 8.0], [5.99, 5.20, 5.20], color=gold, lw=1.2, zorder=4)
arrow((8.0, 5.20), (8.0, 4.43), gold)
ax.text(10.2, 5.37, "Load bundle at startup", ha="center", fontsize=7.4, color=gold)
ax.text(.25, 4.84, "ONLINE", fontsize=9, weight="bold", color=teal)
ax.text(.25, 4.43, "Respond to each rating", fontsize=7.4, color=ink)

# The engine boundary clarifies that both fitting and ranking use the saved bundle.
panel(4.05, 1.75, 7.85, 2.63, "#eef6f2", "#bdd1c6", .7)
ax.text(7.98, 4.09, "LOCAL PYTHON + NUMPY SERVICE", ha="center", fontsize=7.4,
        weight="bold", color=teal)
node(.25, 1.98, 5, "Your taste", "Rate films + set filters\nBrowser saves profile", teal)
node(4.25, 1.98, 6, "Fit your profile", "Solve new user vector\nand user bias\nMovie parameters fixed", teal)
node(8.25, 1.98, 7, "Score & select", "Score; exclude rated\nTitle/year/genre/support\nOptional genre variety", teal)
node(12.25, 1.98, 8, "Your next films", "Ranked recommendations\nPredicted stars + reasons\nLocal poster artwork", teal, "#f0f7ed")
for left in (.25, 4.25, 8.25):
    arrow((left+3.5, 2.89), (left+3.94, 2.89), teal)

# Return path: the displayed results lead to another browser rating.
ax.plot([13.97, 13.97, 1.97], [1.92, .91, .91], color=teal, lw=1.0, zorder=3)
arrow((1.97, .91), (1.97, 1.92), teal)
ax.text(8.0, 1.08, "Results return to the browser; rate another film to refine your picks",
        ha="center", fontsize=7.3, color=teal)
ax.text(8.0, .30, "Saved movie model stays fixed  |  Only the new user profile is fitted online",
        ha="center", fontsize=7.1, color=ink)
fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.01)
for extension in ("pdf", "svg"):
    fig.savefig(HERE / "figures" / f"system_architecture.{extension}", bbox_inches="tight")
plt.close(fig)
