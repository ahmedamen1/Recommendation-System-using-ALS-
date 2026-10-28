"use strict";

const $ = (id) => document.getElementById(id);
const number = new Intl.NumberFormat("en-US");
const compact = new Intl.NumberFormat("en-US", { notation: "compact", maximumFractionDigits: 1 });
const state = { info: null, profile: {}, movies: new Map(), view: "discover", genre: "", query: "", queries: { discover: "", "for-you": "" }, offset: 0, total: 0, request: 0, controller: null };
const palettes = {
  Drama: ["#465a53", "#e5e6bf", "#91a77c"],
  Comedy: ["#b5965c", "#292e2b", "#dccc92"],
  Action: ["#833f37", "#fae0b2", "#b56e52"],
  Adventure: ["#4f6449", "#eee5b4", "#98ac65"],
  "Sci-Fi": ["#304752", "#dee3bb", "#718e91"],
  Crime: ["#655563", "#f0d6b5", "#a28585"],
  Romance: ["#9c7771", "#ffe9c7", "#d6a08a"],
  Horror: ["#384a42", "#e4d7a1", "#7c935d"],
  Animation: ["#887341", "#fcf0c8", "#c4ac68"],
  Thriller: ["#4c5f67", "#f1e8cd", "#859b97"]
};
const escapeHtml = (value) => String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[char]));
const cleanTitle = (title) => title.replace(/\s*\(\d{4}\)\s*$/, "");
const year = (title) => title.match(/\((\d{4})\)\s*$/)?.[1] || "CINEMA";
const colors = (movie) => palettes[movie.genres.find((genre) => palettes[genre])] || palettes.Drama;
const colorStyle = (movie) => { const [bg, fg, ink] = colors(movie); return `--cover-bg:${bg};--cover-fg:${fg};--cover-ink:${ink}`; };
function posterImage(movie) {
  const url = movie.poster_url || "";
  return /^\/posters\/\d+\.(?:jpg|jpeg|png|webp)$/.test(url)
    ? `<img class="poster-image" src="${escapeHtml(url)}" alt="" loading="lazy" decoding="async" width="300" height="450">` : "";
}
document.addEventListener("load", (event) => {
  if (event.target.matches?.(".poster-image")) event.target.parentElement.classList.add("has-poster");
}, true);
document.addEventListener("error", (event) => {
  if (event.target.matches?.(".poster-image")) event.target.remove();
}, true);
let toastTimer, searchTimer;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body;
  try { body = await response.json(); }
  catch { throw new Error("The local server returned an unexpected response."); }
  if (!response.ok) throw new Error(body.error || "The request could not be completed.");
  return body;
}

function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 3500);
}

function saveProfile() {
  try { localStorage.setItem(`frame-profile-${state.info.bundle_id}`, JSON.stringify(state.profile)); }
  catch { toast("Ratings work for this session, but browser storage is unavailable."); }
}

function restoreProfile() {
  try {
    const saved = JSON.parse(localStorage.getItem(`frame-profile-${state.info.bundle_id}`) || "{}");
    for (const [id, entry] of Object.entries(saved).slice(0, 200)) {
      if (entry && entry.movie && String(entry.movie.movieId) === id && typeof entry.movie.title === "string"
          && Array.isArray(entry.movie.genres) && typeof entry.rating === "number" && entry.rating >= .5 && entry.rating <= 5) {
        state.profile[id] = entry;
        state.movies.set(Number(id), entry.movie);
      }
    }
  } catch { /* A malformed or unavailable local store simply starts a fresh profile. */ }
}

function renderProfile() {
  const entries = Object.values(state.profile);
  const count = entries.length;
  $("taste-count").textContent = `${count} ${count === 1 ? "movie" : "movies"} rated`;
  $("nav-count").textContent = count;
  $("taste-hint").textContent = count === 0 ? "Start with one" : count < 5 ? "Try a few more" : "A taste of you";
  $("taste-progress-fill").style.width = `${Math.min(count / 5, 1) * 100}%`;
  $("profile-empty").hidden = count > 0;
  $("reset-button").disabled = count === 0;
  $("recommend-button").disabled = count === 0;
  $("profile-list").innerHTML = entries.map(({ movie, rating }) => `
    <div class="profile-item">
      <div class="mini-cover" style="${colorStyle(movie)}" aria-hidden="true"><span>${escapeHtml(cleanTitle(movie.title).charAt(0))}</span>${posterImage(movie)}</div>
      <div class="profile-info"><strong title="${escapeHtml(movie.title)}">${escapeHtml(cleanTitle(movie.title))}</strong>
        <select aria-label="Your rating for ${escapeHtml(movie.title)}" data-profile-rating="${movie.movieId}">
          ${Array.from({ length: 10 }, (_, i) => (i + 1) / 2).map(value => `<option value="${value}" ${value === rating ? "selected" : ""}>${value.toFixed(1)} ★</option>`).join("")}
        </select>
      </div><button class="remove-rating" data-remove="${movie.movieId}" aria-label="Remove ${escapeHtml(movie.title)} from your ratings">×</button>
    </div>`).join("");
}

function renderGenres() {
  const featured = ["Drama", "Comedy", "Action", "Sci-Fi", "Romance", "Thriller", "Animation", "Adventure", "Crime", "Horror"];
  const choices = [...new Set([...featured, ...state.info.genres])].filter(g => state.info.genres.includes(g));
  $("genres").innerHTML = ["", ...choices].map(genre => `<button class="genre ${state.genre === genre ? "active" : ""}" data-genre="${escapeHtml(genre)}" aria-pressed="${state.genre === genre}">${escapeHtml(genre || "All genres")}</button>`).join("");
}

function scoreDetails(movie) {
  if (!movie.score_parts) return "";
  const parts = movie.score_parts;
  return `<details class="score-details"><summary>Why this score?</summary><p>Your rating baseline ${parts.user_bias.toFixed(2)} + movie bias ${parts.movie_bias.toFixed(2)} + learned taste match ${parts.taste_match.toFixed(2)} = ${movie.raw_prediction.toFixed(2)} raw stars (rounded).</p><p>Displayed stars are limited to 0.5–5. Rating counts describe support, not confidence. Genre notes describe shared metadata.</p></details>`;
}

function filterValues() {
  return { q: state.query, genre: state.genre, year_from: $("year-from").value,
    year_to: $("year-to").value, minimum_ratings: Number($("support").value) };
}

function filterSummary() {
  const filters = filterValues();
  const labels = [filters.q ? `“${filters.q}”` : "", filters.genre,
    filters.year_from || filters.year_to ? `${filters.year_from || "Any year"} – ${filters.year_to || "present"}` : "",
    filters.minimum_ratings ? `${number.format(filters.minimum_ratings)}+ ratings` : ""].filter(Boolean);
  $("filter-summary").textContent = labels.length ? `Showing: ${labels.join(" · ")}` : "Showing the full eligible catalog";
}

function card(movie) {
  state.movies.set(movie.movieId, movie);
  const rating = state.profile[movie.movieId]?.rating || 0;
  const predicted = typeof movie.predicted_rating === "number";
  return `<article class="movie-card" data-movie="${movie.movieId}" aria-label="${escapeHtml(movie.title)}">
    <div class="cover" style="${colorStyle(movie)}" data-pattern="${movie.movieId % 4}" aria-hidden="true">
      ${posterImage(movie)}
      <div class="cover-top"><span>Movie Recommender</span><span>${year(movie.title)}</span></div>
      <h3 class="cover-title">${escapeHtml(cleanTitle(movie.title))}</h3><div class="cover-line"></div>
      <span class="cover-bottom">${escapeHtml(movie.genres[0] || "THE MOVIELENS COLLECTION")}</span>
      ${predicted ? `<span class="prediction-badge">✦ ${movie.predicted_rating.toFixed(1)}</span>` : rating ? `<span class="rated-badge">★ ${rating.toFixed(1)}</span>` : ""}
    </div>
    <h3 class="movie-title" title="${escapeHtml(movie.title)}">${escapeHtml(cleanTitle(movie.title))}</h3>
    <div class="movie-meta">${escapeHtml(year(movie.title))} · ${escapeHtml(movie.genres.slice(0, 2).join(" / ") || "Unclassified")}</div>
    ${predicted ? `<p class="reason">${escapeHtml(movie.reason)}</p>${scoreDetails(movie)}<div class="movie-meta">Predicted ${movie.predicted_rating.toFixed(1)} ★ · ${compact.format(movie.rating_count)} ratings</div>` : ""}
    <div class="stars" role="group" aria-label="Rate ${escapeHtml(movie.title)}">
      ${[1, 2, 3, 4, 5].map(value => `<button class="star ${value <= rating ? "filled" : ""}" data-rate="${movie.movieId}" data-value="${value}" aria-label="Rate ${escapeHtml(movie.title)} ${value} ${value === 1 ? "star" : "stars"}" aria-pressed="${value === rating}">★</button>`).join("")}
      <span class="star-caption">${rating ? "Rated" : "Rate it"}</span>
    </div></article>`;
}

function setEmpty(title, message, browse = false) {
  $("movie-grid").innerHTML = "";
  $("empty-state").hidden = false;
  $("empty-title").textContent = title;
  $("empty-copy").textContent = message;
  $("empty-browse").hidden = !browse;
  $("load-more").hidden = true;
}

function skeleton() {
  $("movie-grid").innerHTML = Array.from({ length: 6 }, () => '<div class="skeleton" aria-hidden="true"><div class="cover"></div><div class="skeleton-line"></div><div class="skeleton-line"></div></div>').join("");
}

async function loadMovies(append = false) {
  if (!state.info) return;
  filterSummary();
  state.controller?.abort();
  const controller = new AbortController();
  state.controller = controller;
  const request = ++state.request;
  $("movie-grid").setAttribute("aria-busy", "true");
  $("notice").hidden = true;
  $("series-section").hidden = true;
  $("series-grid").innerHTML = "";
  $("empty-state").hidden = true;
  $("load-more").disabled = true;
  if (!append) { state.offset = 0; skeleton(); }
  try {
    if (state.view === "for-you" && Object.keys(state.profile).length === 0) {
      $("result-count").textContent = "Waiting for your taste";
      setEmpty("Your taste is the missing piece.", "Rate a movie and your personal recommendations will appear here.", true);
      return;
    }
    let result;
    if (state.view === "discover") {
      const params = new URLSearchParams({ ...filterValues(), sort: $("sort").value, offset: state.offset, limit: 12 });
      result = await api(`/api/movies?${params}`, { signal: controller.signal });
    } else {
      result = await api("/api/recommend", {
        method: "POST", headers: { "Content-Type": "application/json" }, signal: controller.signal,
        body: JSON.stringify({ ratings: Object.fromEntries(Object.entries(state.profile).map(([id, entry]) => [id, entry.rating])), ...filterValues(), diversity: Number($("diversity").value), limit: 12 })
      });
    }
    if (request !== state.request) return;
    if (state.view === "for-you") $("results-caption").textContent = result.ranking === "diverse"
      ? "Model picks with extra genre variety. Predicted stars stay unchanged; the series row keeps model order."
      : "Ranked by your model score. Predicted stars are estimates, not audience scores.";
    const series = result.series?.[0];
    if (state.view === "for-you" && series) {
      $("series-heading").textContent = `More from ${series.title}`;
      $("series-grid").innerHTML = series.movies.map(movie => card({ ...movie, reason: `Series match · overall model rank #${movie.overall_rank}` })).join("");
      $("series-section").hidden = false;
    }
    state.total = state.view === "discover" ? result.total : result.movies.length;
    state.offset += result.movies.length;
    $("result-count").textContent = state.view === "discover" ? `${number.format(state.total)} films` : `${result.movies.length} picks / ${number.format(result.eligible_movies)} matches`;
    if (!result.movies.length && !append) {
      setEmpty("No films match these filters.", "Try a wider year range, another title or genre, or a lower minimum rating count.");
    } else {
      const html = result.movies.map(card).join("");
      if (append) $("movie-grid").insertAdjacentHTML("beforeend", html);
      else $("movie-grid").innerHTML = html;
      $("load-more").hidden = state.view !== "discover" || state.offset >= state.total;
    }
  } catch (error) {
    if (error.name === "AbortError" || request !== state.request) return;
    $("notice").textContent = error.message;
    $("notice").hidden = false;
    setEmpty("We couldn’t load your films.", "Check that the local Python server is running, then switch tabs to retry.");
  } finally {
    if (request === state.request) {
      $("movie-grid").setAttribute("aria-busy", "false");
      $("load-more").disabled = false;
    }
  }
}

function setView(view, scroll = false) {
  clearTimeout(searchTimer);
  state.queries[state.view] = $("search").value.trim();
  state.view = view;
  state.query = state.queries[view];
  $("search").value = state.query;
  const personal = view === "for-you";
  for (const [id, active] of [["browse-tab", !personal], ["for-you-tab", personal], ["discover-nav", !personal], ["recommend-nav", personal]]) {
    $(id).classList.toggle("active", active);
    $(id).setAttribute("aria-pressed", String(active));
  }
  $("breadcrumb-view").textContent = personal ? "For you" : "Discover";
  $("section-eyebrow").textContent = personal ? "YOUR NEXT MOVIE NIGHT" : "MAKE IT PERSONAL";
  $("collection-title").textContent = personal ? "You may also like" : "A good place to start";
  $("search").placeholder = personal ? "Search within your recommendations…" : "Search for a movie…";
  $("sort-filter").hidden = personal;
  $("diversity-filter").hidden = !personal;
  $("results-caption").textContent = personal ? "Computed from your ratings. Predicted stars are estimates, not audience scores." : "Popular films from your model’s catalog. Tap a star to add your rating.";
  loadMovies();
  if (scroll) $("collection").scrollIntoView({ behavior: "smooth", block: "start" });
}

function rateMovie(id, rating) {
  const movie = state.movies.get(id);
  if (!movie) return;
  if (!state.profile[id] && Object.keys(state.profile).length >= 200) {
    toast("Your profile can hold 200 movies. Remove a rating to add another."); return;
  }
  state.profile[id] = { movie, rating };
  saveProfile(); renderProfile();
  toast(`Rated ${cleanTitle(movie.title)} ${rating.toFixed(1)} ★ — updating your picks`);
  setView("for-you");
}

function showModel() {
  if (!state.info) return;
  const info = state.info;
  const facts = [["Dataset", info.dataset], ["Model", info.model_type], ["Training ratings", number.format(info.training_ratings)], ["Movies", number.format(info.movies)], ["Latent dimensions", info.factors], ["Training sweeps", info.sweeps], ["Fit", info.full_refit ? "All ratings refit" : "Training split only"], ["Inference", "Local · no retraining"]];
  $("model-details").innerHTML = facts.map(([name, value]) => `<dt>${escapeHtml(name)}</dt><dd>${escapeHtml(value)}</dd>`).join("");
  $("model-dialog").showModal();
}

$("collection").addEventListener("click", event => {
  const button = event.target.closest("[data-rate]");
  if (button) rateMovie(Number(button.dataset.rate), Number(button.dataset.value));
});
$("profile-list").addEventListener("change", event => {
  if (event.target.matches("[data-profile-rating]")) rateMovie(Number(event.target.dataset.profileRating), Number(event.target.value));
});
$("profile-list").addEventListener("click", event => {
  const button = event.target.closest("[data-remove]");
  if (!button) return;
  delete state.profile[button.dataset.remove];
  saveProfile(); renderProfile(); loadMovies();
  toast("Rating removed. Your recommendations have been updated.");
});
$("genres").addEventListener("click", event => {
  const button = event.target.closest("[data-genre]");
  if (!button) return;
  state.genre = button.dataset.genre;
  for (const genre of $("genres").children) {
    const active = genre === button;
    genre.classList.toggle("active", active); genre.setAttribute("aria-pressed", String(active));
  }
  loadMovies();
});
$("search").addEventListener("input", () => {
  clearTimeout(searchTimer);
  // Cancel immediately so an older result cannot replace the current search.
  state.controller?.abort();
  searchTimer = setTimeout(() => { state.query = $("search").value.trim(); loadMovies(); }, 220);
});
for (const id of ["support", "year-from", "year-to", "sort", "diversity"]) {
  $(id).addEventListener("change", () => {
    clearTimeout(searchTimer); state.query = $("search").value.trim(); loadMovies();
  });
}
$("reset-filters").addEventListener("click", () => {
  clearTimeout(searchTimer);
  state.query = ""; state.queries[state.view] = ""; state.genre = "";
  for (const id of ["search", "year-from", "year-to"]) $(id).value = "";
  $("support").value = "0"; $("sort").value = "popular"; $("diversity").value = "0";
  renderGenres(); loadMovies();
});
$("load-more").addEventListener("click", () => loadMovies(true));
for (const id of ["discover-nav", "browse-tab", "empty-browse", "start-button"]) $(id).addEventListener("click", () => setView("discover", id === "start-button"));
for (const id of ["recommend-nav", "for-you-tab", "recommend-button"]) $(id).addEventListener("click", () => setView("for-you", id === "recommend-button"));
$("taste-nav").addEventListener("click", () => { $("taste-panel").scrollIntoView({ behavior: "smooth", block: "center" }); $("recommend-button").focus({ preventScroll: true }); });
$("reset-button").addEventListener("click", () => { state.profile = {}; saveProfile(); renderProfile(); loadMovies(); toast("A fresh start. Your ratings have been cleared."); });
$("model-button").addEventListener("click", showModel);
$("footer-model").addEventListener("click", showModel);
$("close-dialog").addEventListener("click", () => $("model-dialog").close());
$("model-dialog").addEventListener("click", event => { if (event.target === $("model-dialog")) { const rect = event.target.getBoundingClientRect(); if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) event.target.close(); } });
document.addEventListener("keydown", event => { if (event.key === "/" && !["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName) && !$("model-dialog").open) { event.preventDefault(); $("search").focus(); } });

async function init() {
  skeleton();
  try {
    state.info = await api("/api/model", { signal: AbortSignal.timeout(10000) });
    const info = state.info;
    $("stat-movies").textContent = number.format(info.movies);
    $("stat-ratings").textContent = compact.format(info.training_ratings);
    $("stat-factors").textContent = info.factors;
    $("dataset-label").textContent = info.small_data ? "Small-data demo model" : "MovieLens 32M model";
    $("dataset-detail").textContent = `${info.model_type} · ${info.sweeps} sweeps · ${info.full_refit ? "all ratings refit" : "training split"}`;
    $("sidebar-model").textContent = `${info.model_type} connected`;
    restoreProfile(); renderProfile(); renderGenres();
    await loadMovies();
  } catch (error) {
    $("dataset-label").textContent = "Model unavailable";
    $("sidebar-model").textContent = "Not connected";
    $("notice").textContent = "Start the local app with: python dashboard/server.py --bundle path/to/inference_bundle.zip. Then open http://127.0.0.1:8000.";
    $("notice").hidden = false;
    setEmpty("Your model is waiting backstage.", "This dashboard needs the local Python server. Open its localhost address rather than opening the HTML file directly.");
    $("movie-grid").setAttribute("aria-busy", "false");
  }
}
init();
