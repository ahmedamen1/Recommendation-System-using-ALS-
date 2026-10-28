"""Regenerate vector figures and LaTeX tables from audited run artifacts."""
from pathlib import Path
import textwrap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"
AUDIT = HERE / "audit"
FIGURES = HERE / "figures"
TABLES = HERE / "tables"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
BLUE, ORANGE = "#245c8a", "#bd5a28"


def save(fig, name):
    fig.tight_layout(pad=.6)
    fig.savefig(FIGURES / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def history_plot(frame, name, bias=False):
    fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.65), sharex=True)
    axes[0].plot(frame.sweep, frame.loss, ".-", color=BLUE, markersize=3)
    axes[0].set_ylabel("Negative log joint", fontsize=8)
    axes[0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    axes[0].yaxis.get_offset_text().set_fontsize(7)
    axes[1].plot(frame.sweep, frame.train_rmse, color=BLUE, label="Train")
    axes[1].plot(frame.sweep, frame.test_rmse, color=ORANGE, label="Test")
    axes[1].set_ylabel("RMSE (stars)", fontsize=8)
    span = max(frame.train_rmse.max(), frame.test_rmse.max()) - min(frame.train_rmse.min(), frame.test_rmse.min())
    tick_step = next(step for step in [.01, .02, .05, .1, .2, .5, 1, 2, 5] if step >= span / 4)
    axes[1].yaxis.set_major_locator(MultipleLocator(tick_step))
    axes[1].yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    axes[1].legend(frameon=False, fontsize=7)
    axes[1].set_xlabel("Cumulative update" if bias else "ALS sweep", fontsize=8)
    for ax in axes:
        ax.tick_params(labelsize=7)
        ax.grid(alpha=.15)
    save(fig, name)


history_plot(pd.read_csv(AUDIT / "bias_history_corrected.csv"), "bias_learning_curves", True)
for stem, output in [("als_k10", "practical3_learning_curves"), ("genre", "genre_learning_curves")]:
    history_plot(pd.read_csv(RESULTS / f"{stem}_history.csv"), output)
fig, axes = plt.subplots(2, 1, figsize=(3.25, 2.65), sharex=True)
for k, color in [(10, BLUE), (20, ORANGE)]:
    history = pd.read_csv(RESULTS / f"als_k{k}_history.csv")
    for ax, column in zip(axes, ["train_rmse", "test_rmse"]):
        ax.plot(history.sweep, history[column], color=color, label=f"K = {k}")
for ax, label in zip(axes, ["Training", "Test"]):
    ax.set_ylabel(f"{label} RMSE", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.legend(frameon=False, fontsize=7, handlelength=1.5, labelspacing=.2)
    ax.grid(alpha=.15)
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
axes[1].set_xlabel("ALS sweep", fontsize=8)
save(fig, "capacity_comparison")

ratings = pd.read_csv(AUDIT / "rating_counts.csv")
fig, ax = plt.subplots(figsize=(3.25, 2.15))
ax.bar(ratings.rating, ratings["count"] / 1e6, width=.38, color=BLUE)
ax.set(xlabel="Rating (stars)", ylabel="Ratings (millions)", xticks=[1, 2, 3, 4, 5])
save(fig, "rating_distribution")
fig, ax = plt.subplots(figsize=(3.25, 2.15))
for name, color in [("users", BLUE), ("movies", ORANGE)]:
    frame = pd.read_csv(AUDIT / f"{name}_degrees.csv")
    frame = frame[frame.degree > 0]
    ax.loglog(frame.degree, frame.fraction, ".", color=color, label=name.title(), markersize=2)
ax.set(xlabel="Number of ratings", ylabel="Fraction of catalog entities")
ax.legend(frameon=False)
save(fig, "degree_distribution")

for stem in ["movies_2d", "genres_2d"]:
    coordinates = pd.read_csv(RESULTS / f"{stem}_coordinates.csv")
    is_movies = stem == "movies_2d"
    fig, (ax, key) = plt.subplots(2, 1, figsize=(3.25, 3.65 if is_movies else 3.15),
                                  gridspec_kw={"height_ratios": [1.9, 1.4 if is_movies else .9]})
    points = coordinates[["coordinate_1", "coordinate_2"]].to_numpy()
    ax.scatter(points[:, 0], points[:, 1], s=15, color=BLUE)
    offsets = ({3: (6, 17), 5: (2, -5), 6: (-17, -9), 9: (7, 1),
                13: (8, -3), 14: (-9, 7), 15: (7, -7), 16: (-20, -8),
                17: (-10, -13), 18: (12, 4), 19: (0, 14)}
               if stem == 'genres_2d'
               else {9: (8, -2), 12: (-17, -9), 2: (-10, 7), 7: (5, -12)})
    for number, point in enumerate(points, 1):
        ax.annotate(str(number), point, xytext=offsets.get(number, (5, 6)),
                    textcoords="offset points", fontsize=7,
                    arrowprops={"arrowstyle": "-", "lw": .4, "color": "#888888"})
    ax.set_xlabel("Latent coordinate 1", fontsize=8)
    ax.set_ylabel("Latent coordinate 2", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.margins(.13); ax.grid(alpha=.12)
    key.axis("off")
    columns = 2 if is_movies else 3
    per_column = int(np.ceil(len(coordinates) / columns))
    short_titles = {
        "Star Wars: Episode V - The Empire Strikes Back (1980)": "Empire Strikes Back (1980)",
        "Lord of the Rings: The Fellowship of the Ring, The (2001)": "Fellowship of the Ring (2001)",
        "Star Wars: Episode IV - A New Hope (1977)": "A New Hope (1977)",
        "Silence of the Lambs, The (1991)": "Silence of the Lambs (1991)",
        "Matrix, The (1999)": "The Matrix (1999)",
        "Shawshank Redemption, The (1994)": "Shawshank Redemption (1994)",
    }
    labels = [textwrap.fill(f"{i}. {short_titles.get(label, label) if is_movies else label}",
                            24 if is_movies else 18, subsequent_indent="   ")
              for i, label in enumerate(coordinates.label, 1)]
    for column in range(columns):
        group = labels[column * per_column:(column + 1) * per_column]
        key.text(column / columns, 1, "\n".join(group), transform=key.transAxes,
                 va="top", fontsize=7, linespacing=1.12)
    save(fig, stem)


def tex(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def table(name, spec, header, rows):
    lines = [r"\begin{tabular*}{\linewidth}{@{\extracolsep{\fill}}"+spec+r"}", r"\toprule", " & ".join(header)+r" \\", r"\midrule"]
    lines += [" & ".join(map(str, row))+r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular*}"]
    (TABLES / f"{name}.tex").write_text("\n".join(lines)+"\n", encoding="utf-8")


final = pd.read_csv(RESULTS / "final_test_results.csv").set_index("model")
comparison = pd.read_csv(RESULTS / "als_test_comparison.csv").set_index("name")
table("final_results", "lrr", ["Model", "Updates", "Test RMSE"], [
    ["Training mean", "--", f"{final.loc['Training mean','test_rmse']:.2f}"],
    ["Biases only", "28", f"{final.loc['Biases only','test_rmse']:.2f}"],
    ["ALS, $K=10$", "20", f"{final.loc['Configured ALS','test_rmse']:.2f}"],
    ["ALS, $K=20$", "20", f"{comparison.loc['als_k20','test_rmse']:.2f}"],
    ["Genre ALS, $K=10$", "20", f"{final.loc['Genre ALS','test_rmse']:.2f}"],
])
rows=[]
labels=["Baseline", r"$K=20$", r"$\tau=0.01$", r"$\tau=1$", r"$\lambda=0.1$"]
for label,(_, row) in zip(labels,comparison.iterrows()):
    rows.append([label,str(int(row.factors)),f"{row.lam:g}",f"{row.tau:g}",f"{row.train_rmse:.2f}",f"{row.test_rmse:.2f}",f"{row.seconds:.2f}"])
table("experiments", "lrrrrrr", ["Setting", "$K$", r"$\lambda$", r"$\tau$", "Train", "Test", "Time (s)"], rows)
bench=pd.read_csv(RESULTS/'update_benchmark.csv')
reference=bench.median_seconds.iloc[0]
table("benchmark", "lrr", ["Method", "Median (ms)", "Speedup"], [[tex(row.method),f"{1000 * row.median_seconds:.2f}",f"{reference/row.median_seconds:.2f}$\\times$"] for row in bench.itertuples()])
groups=pd.read_csv(RESULTS/'user_activity_errors.csv')
rows=[]
for label in ['1-19','20-99','100+']:
    group=groups[groups.user_train_ratings==label].set_index('model')
    rows.append([label.replace('-','--'),f"{int(group.loc['als_k10','test_ratings']):,}",f"{group.loc['als_k10','rmse']:.2f}",f"{group.loc['als_k20','rmse']:.2f}"])
table('activity','lrrr',['Train ratings','Test rows','$K=10$','$K=20$'],rows)
natural=pd.read_csv(RESULTS/'natural_cold_start_errors.csv')
controlled=pd.read_csv(RESULTS/'controlled_cold_start.csv')
rows=[]
for label, group_name in [('Warm items','warm user / warm item'),('Natural cold items','warm user / cold item')]:
    frame=natural[natural.group==group_name].set_index('model')
    rows.append([label,f"{int(frame.loc['Configured ALS','test_ratings']):,}",f"{frame.loc['Configured ALS','rmse']:.2f}",f"{frame.loc['Genre ALS','rmse']:.2f}"])
rows.append(['Controlled cold items',f"{int(controlled.test_ratings.iloc[0]):,}",f"{controlled.test_rmse.iloc[0]:.2f}",f"{controlled.test_rmse.iloc[1]:.2f}"])
table('cold_start','lrrr',['Test group','Ratings','ALS','Genre ALS'],rows)
recommendations=pd.read_csv(RESULTS/'recommendations_1.0_100.csv').head(5)
short_titles=['The Return of the King (2003)','The Two Towers (2002)','A New Hope (1977)','The Empire Strikes Back (1980)','Return of the Jedi (1983)']
table('recommendations','lrr',['Recommended movie','Prediction','Support'],[[tex(title),f'{row.predicted_rating:.2f}',f'{row.observed_ratings:,}'] for title,row in zip(short_titles,recommendations.itertuples())])
polar=pd.read_csv(RESULTS/'polarizing_movies.csv').head(5)
titles=[r'Dumb \& Dumber (1994)','The Return of the King (2003)','The Two Towers (2002)','The Fellowship of the Ring (2001)','Deathly Hallows: Part 1 (2010)']
table('polarization',r'p{0.52\linewidth}rrr',['Movie','Norm','SD','Support'],[[title,f'{row.vector_norm:.2f}',f'{row.rating_std:.2f}',f'{row.ratings:,}'] for title,row in zip(titles,polar.itertuples())])
print('Prepared 8 vector PDF figures and 7 LaTeX tables.')
