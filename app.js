// ======================================================
// HackClubAI Web (GitHub Pages static)
// ✅ Animated galaxy background (CSS)
// ✅ Chat bubbles
// ✅ Model cards grid
// ✅ Filter that DOES NOT break selection
// ✅ Enter = Run, Shift+Enter = newline
// ======================================================

const AI_BASE_URL = "aiproxy.jwhughes2012.workers.dev/proxy";
const CHAT_COMPLETIONS_URL = `${AI_BASE_URL}/chat/completions`;
const SEARCH_BASE = "https://search.hackclub.com/res/v1";

const LS = {
  aiKey: "hc_ai_key",
  searchKey: "hc_search_key",
  theme: "hc_theme",
  sysPrompt: "hc_system_prompt",
  textModel: "hc_text_model",
  temp: "hc_temp",
  imageModel: "hc_image_model",
  aspect: "hc_aspect",
  imgCount: "hc_img_count",
  useSearch: "hc_use_search",
  searchType: "hc_search_type",
  searchCount: "hc_search_count",
  newsFresh: "hc_news_freshness",
};

const TEXT_MODELS = [
  { id:"qwen/qwen3-32b", name:"Qwen 3 32B", ctx:"41K", best:"Fast daily driver for chat + coding.", icon:"assets/icons/qwen.png" },
  { id:"moonshotai/kimi-k2-thinking", name:"Kimi K2 Thinking", ctx:"262K", best:"Long-context reasoning + big docs.", icon:"assets/icons/moonshot.png" },
  { id:"openai/gpt-oss-120b", name:"gpt-oss-120b", ctx:"131K", best:"Heavy prompts (slower).", icon:"assets/icons/openai.png" },
  { id:"moonshotai/kimi-k2-0905", name:"Kimi K2 0905", ctx:"262K", best:"Long-context general model.", icon:"assets/icons/moonshot.png" },
  { id:"google/gemini-3-flash-preview", name:"Gemini 3 Flash Preview", ctx:"1.0M", best:"Huge context + fast.", icon:"assets/icons/gemini.png" },
  { id:"qwen/qwen3-next-80b-a3b-instruct", name:"Qwen 3 Next 80B", ctx:"262K", best:"Stronger reasoning / harder coding.", icon:"assets/icons/qwen.png" },
  { id:"z-ai/glm-4.7", name:"GLM 4.7", ctx:"203K", best:"Strong general reasoning.", icon:"assets/icons/zai.png" },
  { id:"deepseek/deepseek-v3.2-speciale", name:"DeepSeek V3.2 Speciale", ctx:"164K", best:"Strong technical/code tasks.", icon:"assets/icons/deepseek.png" },
  { id:"deepseek/deepseek-v3.2", name:"DeepSeek V3.2", ctx:"164K", best:"Good balance.", icon:"assets/icons/deepseek.png" },
  { id:"x-ai/grok-4.1-fast", name:"Grok 4.1 Fast", ctx:"2.0M", best:"Massive context, fast.", icon:"assets/icons/grok.png" },
  { id:"google/gemini-2.5-flash", name:"Gemini 2.5 Flash", ctx:"1.0M", best:"Huge context + speed.", icon:"assets/icons/gemini.png" },
  { id:"openai/gpt-5.1", name:"GPT-5.1", ctx:"400K", best:"Higher quality reasoning/writing.", icon:"assets/icons/openai.png" },
  { id:"openai/gpt-5-mini", name:"GPT-5 Mini", ctx:"400K", best:"Fast general Q&A.", icon:"assets/icons/openai.png" },
];

const IMAGE_MODELS = [
  { id:"google/gemini-2.5-flash-image", name:"Gemini 2.5 Flash Image", ctx:"33K", best:"Fast image iterations.", icon:"assets/icons/gemini.png" },
  { id:"google/gemini-3-pro-image-preview", name:"Gemini 3 Pro Image Preview", ctx:"66K", best:"Higher quality, slower.", icon:"assets/icons/gemini.png" },
];

const ASPECTS = ["1:1","2:3","3:2","3:4","4:3","4:5","5:4","9:16","16:9","21:9"];

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

const runBtn = $("runBtn");
const settingsBtn = $("settingsBtn");
const promptEl = $("prompt");
const chatEl = $("chat");
const previewImg = $("previewImg");
const previewPlaceholder = $("previewPlaceholder");
const downloadLink = $("downloadLink");

const tabs = Array.from(document.querySelectorAll(".tab"));
const tabpages = {
  text: $("tab-text"),
  image: $("tab-image"),
  search: $("tab-search"),
};

const textModelGrid = $("textModelGrid");
const imageModelGrid = $("imageModelGrid");
const textDetail = $("textDetail");
const imageDetail = $("imageDetail");

const temp = $("temp");
const tempVal = $("tempVal");
const aspect = $("aspect");
const imgCount = $("imgCount");

const textFilter = $("textFilter");

const useSearch = $("useSearch");
const searchType = $("searchType");
const searchCount = $("searchCount");
const newsFreshness = $("newsFreshness");
const searchOut = $("searchOut");

const copyBtn = $("copyBtn");
const clearBtn = $("clearBtn");

// Modal
const modalBg = $("modalBg");
const modal = $("modal");
const closeModal = $("closeModal");
const aiKey = $("aiKey");
const searchKey = $("searchKey");
const theme = $("theme");
const systemPrompt = $("systemPrompt");
const saveSettings = $("saveSettings");
const resetSettings = $("resetSettings");

// ---------- State ----------
let state = {
  activeTab: "text",
  aiKey: localStorage.getItem(LS.aiKey) || "",
  searchKey: localStorage.getItem(LS.searchKey) || "",
  theme: localStorage.getItem(LS.theme) || "dark",
  systemPrompt: localStorage.getItem(LS.sysPrompt) || "You are a helpful assistant. Be accurate. If unsure, say so.",
  textModel: localStorage.getItem(LS.textModel) || TEXT_MODELS[0].id,
  temp: parseFloat(localStorage.getItem(LS.temp) || "0.7"),
  imageModel: localStorage.getItem(LS.imageModel) || IMAGE_MODELS[0].id,
  aspect: localStorage.getItem(LS.aspect) || "1:1",
  imgCount: parseInt(localStorage.getItem(LS.imgCount) || "1", 10),

  useSearch: (localStorage.getItem(LS.useSearch) || "false") === "true",
  searchType: localStorage.getItem(LS.searchType) || "web",
  searchCount: parseInt(localStorage.getItem(LS.searchCount) || "5", 10),
  newsFresh: localStorage.getItem(LS.newsFresh) || "pw",
};

function applyTheme() {
  document.body.classList.toggle("light", state.theme === "light");
}

// ---------- Chat bubbles ----------
function addBubble(kind, text) {
  const div = document.createElement("div");
  div.className = `bubble ${kind}`;
  div.textContent = text;
  chatEl.appendChild(div);
  chatEl.scrollTop = chatEl.scrollHeight;
}

function clearChat() {
  chatEl.innerHTML = "";
  setPreview("");
  searchOut.textContent = "(none)";
}

function copyChat() {
  const text = Array.from(chatEl.querySelectorAll(".bubble"))
    .map(b => {
      const who = b.classList.contains("user") ? "You" :
                  b.classList.contains("assistant") ? "AI" : "Note";
      return `${who}: ${b.textContent}`;
    })
    .join("\n\n");
  return text;
}

// ---------- Model cards ----------
function cardHTML(m) {
  const safeIcon = m.icon || "assets/icons/default.png";
  return `
    <div class="card" data-id="${m.id}" title="${m.id}">
      <img src="${safeIcon}" onerror="this.src='assets/icons/default.png'"/>
      <div class="meta">
        <div class="name">${m.name}</div>
        <div class="sub">${m.ctx} • ${m.id}</div>
      </div>
    </div>
  `;
}

// IMPORTANT: selection must be reapplied after every render
function selectCard(container, id) {
  container.querySelectorAll(".card").forEach(el => el.classList.remove("selected"));
  const match = container.querySelector(`.card[data-id="${CSS.escape(id)}"]`);
  if (match) match.classList.add("selected");
}

function updateTextDetail(id) {
  const m = TEXT_MODELS.find(x => x.id === id);
  if (!m) return;
  textDetail.textContent = `${m.name}\nModel: ${m.id}\nContext: ${m.ctx}\n\nBest for: ${m.best}`;
}

function updateImageDetail(id) {
  const m = IMAGE_MODELS.find(x => x.id === id);
  if (!m) return;
  imageDetail.textContent = `${m.name}\nModel: ${m.id}\nContext: ${m.ctx}\n\nBest for: ${m.best}`;
}

function renderTextModels(filter = "") {
  const f = filter.trim().toLowerCase();
  const list = TEXT_MODELS.filter(m =>
    !f || m.name.toLowerCase().includes(f) || m.id.toLowerCase().includes(f)
  );

  textModelGrid.innerHTML = list.map(cardHTML).join("");

  // Re-apply highlight ONLY if the selected model is visible
  selectCard(textModelGrid, state.textModel);

  // Keep details tied to selection in state (not filtered list)
  updateTextDetail(state.textModel);
}

function renderImageModels() {
  imageModelGrid.innerHTML = IMAGE_MODELS.map(cardHTML).join("");
  selectCard(imageModelGrid, state.imageModel);
  updateImageDetail(state.imageModel);
}

// ---------- Preview ----------
function setPreview(dataUrl) {
  if (!dataUrl) {
    previewImg.style.display = "none";
    previewPlaceholder.style.display = "block";
    downloadLink.style.display = "none";
    return;
  }
  previewImg.src = dataUrl;
  previewImg.style.display = "block";
  previewPlaceholder.style.display = "none";
  downloadLink.href = dataUrl;
  downloadLink.style.display = "inline-flex";
}

// ---------- Tabs ----------
tabs.forEach(btn => {
  btn.addEventListener("click", () => {
    tabs.forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    state.activeTab = btn.dataset.tab;
    Object.values(tabpages).forEach(p => p.classList.remove("show"));
    tabpages[state.activeTab].classList.add("show");
  });
});

// ---------- Modal ----------
function openModal() {
  modalBg.classList.remove("hidden");
  modal.classList.remove("hidden");
  aiKey.value = state.aiKey;
  searchKey.value = state.searchKey;
  theme.value = state.theme;
  systemPrompt.value = state.systemPrompt;
}
function closeModalFn() {
  modalBg.classList.add("hidden");
  modal.classList.add("hidden");
}
settingsBtn.addEventListener("click", openModal);
closeModal.addEventListener("click", closeModalFn);
modalBg.addEventListener("click", closeModalFn);

saveSettings.addEventListener("click", () => {
  state.aiKey = aiKey.value.trim();
  state.searchKey = searchKey.value.trim();
  state.theme = theme.value;
  state.systemPrompt = systemPrompt.value.trim() || "You are a helpful assistant. Be accurate. If unsure, say so.";

  localStorage.setItem(LS.aiKey, state.aiKey);
  localStorage.setItem(LS.searchKey, state.searchKey);
  localStorage.setItem(LS.theme, state.theme);
  localStorage.setItem(LS.sysPrompt, state.systemPrompt);

  applyTheme();
  closeModalFn();
});

resetSettings.addEventListener("click", () => {
  localStorage.clear();
  location.reload();
});

// ---------- Controls sync ----------
temp.value = String(state.temp);
tempVal.textContent = state.temp.toFixed(1);
temp.addEventListener("input", () => {
  state.temp = parseFloat(temp.value);
  tempVal.textContent = state.temp.toFixed(1);
  localStorage.setItem(LS.temp, String(state.temp));
});

ASPECTS.forEach(a => {
  const opt = document.createElement("option");
  opt.value = a;
  opt.textContent = a;
  aspect.appendChild(opt);
});
aspect.value = state.aspect;
aspect.addEventListener("change", () => {
  state.aspect = aspect.value;
  localStorage.setItem(LS.aspect, state.aspect);
});

imgCount.value = String(state.imgCount);
imgCount.addEventListener("change", () => {
  state.imgCount = Math.max(1, Math.min(4, parseInt(imgCount.value || "1", 10)));
  imgCount.value = String(state.imgCount);
  localStorage.setItem(LS.imgCount, String(state.imgCount));
});

useSearch.checked = state.useSearch;
searchType.value = state.searchType;
searchCount.value = String(state.searchCount);
newsFreshness.value = state.newsFresh;

useSearch.addEventListener("change", () => {
  state.useSearch = useSearch.checked;
  localStorage.setItem(LS.useSearch, String(state.useSearch));
});
searchType.addEventListener("change", () => {
  state.searchType = searchType.value;
  localStorage.setItem(LS.searchType, state.searchType);
});
searchCount.addEventListener("change", () => {
  state.searchCount = Math.max(1, Math.min(10, parseInt(searchCount.value || "5", 10)));
  searchCount.value = String(state.searchCount);
  localStorage.setItem(LS.searchCount, String(state.searchCount));
});
newsFreshness.addEventListener("change", () => {
  state.newsFresh = newsFreshness.value;
  localStorage.setItem(LS.newsFresh, state.newsFresh);
});

// Filter: renders but does NOT change selection
textFilter.addEventListener("input", () => {
  renderTextModels(textFilter.value);
});

// ---------- Model selection click (event delegation = survives re-render) ----------
textModelGrid.addEventListener("click", (e) => {
  const card = e.target.closest(".card");
  if (!card) return;

  state.textModel = card.dataset.id;
  localStorage.setItem(LS.textModel, state.textModel);

  selectCard(textModelGrid, state.textModel);
  updateTextDetail(state.textModel);
});

imageModelGrid.addEventListener("click", (e) => {
  const card = e.target.closest(".card");
  if (!card) return;

  state.imageModel = card.dataset.id;
  localStorage.setItem(LS.imageModel, state.imageModel);

  selectCard(imageModelGrid, state.imageModel);
  updateImageDetail(state.imageModel);
});

// ---------- Enter to Run ----------
promptEl.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    runBtn.click();
  }
});

// ---------- Search ----------
async function doSearch(query) {
  if (!state.searchKey) return "";

  const headers = { "Authorization": `Bearer ${state.searchKey}` };
  let url = "";

  if (state.searchType === "news") {
    const params = new URLSearchParams({ q: query, freshness: state.newsFresh, count: String(state.searchCount) });
    url = `${SEARCH_BASE}/news/search?${params.toString()}`;
  } else {
    const params = new URLSearchParams({ q: query, count: String(state.searchCount) });
    url = `${SEARCH_BASE}/web/search?${params.toString()}`;
  }

  // NOTE: If CORS is blocked, this will fail on GitHub Pages.
  const res = await fetch(url, { headers });
  if (!res.ok) throw new Error(`Search failed (${res.status})`);
  const data = await res.json();

  const bucket = (state.searchType === "news" ? data.news : data.web) || {};
  const results = bucket.results || [];
  const lines = results.slice(0, state.searchCount).map((r, i) => {
    const t = r.title || "Untitled";
    const u = r.url || "";
    const d = r.description || r.snippet || "";
    return `${i+1}. ${t}\n${u}\n${d}`.trim();
  });

  return lines.join("\n\n").trim();
}

// ---------- AI Calls ----------
async function callText(prompt, searchContext="") {
  const headers = {
    "Authorization": `Bearer ${state.aiKey}`,
    "Content-Type": "application/json"
  };

  const messages = [{ role: "system", content: state.systemPrompt }];

  if (searchContext) {
    messages.push({
      role: "system",
      content: "Use these search results to answer accurately. If you use a fact, cite the URL.\n\nSEARCH RESULTS:\n" + searchContext
    });
  }

  messages.push({ role: "user", content: prompt });

  const payload = {
    model: state.textModel,
    messages,
    temperature: state.temp,
    stream: false
  };

  const res = await fetch(CHAT_COMPLETIONS_URL, { method:"POST", headers, body: JSON.stringify(payload) });
  if (!res.ok) throw new Error(`Text request failed (${res.status})`);
  const data = await res.json();
  return (data?.choices?.[0]?.message?.content || "").trim();
}

async function callImage(prompt) {
  const headers = {
    "Authorization": `Bearer ${state.aiKey}`,
    "Content-Type": "application/json"
  };

  const payload = {
    model: state.imageModel,
    messages: [{ role:"user", content: prompt }],
    modalities: ["image","text"],
    image_config: { aspect_ratio: state.aspect },
    stream: false,
    n: state.imgCount
  };

  const res = await fetch(CHAT_COMPLETIONS_URL, { method:"POST", headers, body: JSON.stringify(payload) });
  if (!res.ok) throw new Error(`Image request failed (${res.status})`);
  const data = await res.json();

  const message = data?.choices?.[0]?.message || {};
  const assistantText = (message.content || "").trim();
  const images = message.images || [];
  if (!images.length) throw new Error("No images returned");

  const url = images[0]?.image_url?.url || "";
  if (!url) throw new Error("Image URL missing");

  const dataUrl = url.includes("data:") ? url : `data:image/png;base64,${url}`;
  return { assistantText, dataUrl };
}

// ---------- Run ----------
function requireKey() {
  if (!state.aiKey) {
    openModal();
    addBubble("sys", "Paste your Hack Club AI key in Settings first.");
    return false;
  }
  return true;
}

async function run() {
  if (!requireKey()) return;

  const p = promptEl.value.trim();
  if (!p) {
    addBubble("sys", "Type a prompt first.");
    return;
  }

  runBtn.disabled = true;
  runBtn.textContent = "Running…";

  addBubble("user", p);

  try {
    if (state.activeTab === "image") {
      addBubble("sys", `Image: ${state.imageModel} • aspect=${state.aspect} • n=${state.imgCount}`);
      const { assistantText, dataUrl } = await callImage(p);
      if (assistantText) addBubble("assistant", assistantText);
      setPreview(dataUrl);
      searchOut.textContent = "(search is for Text mode)";
    } else {
      let ctx = "";
      if (state.useSearch && state.searchKey) {
        try {
          ctx = await doSearch(p);
          searchOut.textContent = ctx || "(no search results)";
        } catch (e) {
          searchOut.textContent = `(search failed: ${e.message})`;
          ctx = "";
        }
      } else {
        searchOut.textContent = "(search disabled or no search key)";
      }

      addBubble("sys", `Text: ${state.textModel} • temp=${state.temp.toFixed(1)}`);
      const ans = await callText(p, ctx);
      addBubble("assistant", ans || "(empty response)");
      setPreview("");
    }
  } catch (e) {
    addBubble("sys", `Error: ${e.message}`);
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "Run";
  }
}

runBtn.addEventListener("click", run);

// ---------- Copy/Clear ----------
copyBtn.addEventListener("click", async () => {
  try {
    await navigator.clipboard.writeText(copyChat());
    addBubble("sys", "Copied chat to clipboard.");
  } catch {
    addBubble("sys", "Copy failed (browser blocked clipboard).");
  }
});

clearBtn.addEventListener("click", () => clearChat());

// ---------- Init ----------
function init() {
  applyTheme();

  renderTextModels("");
  renderImageModels();

  updateTextDetail(state.textModel);
  updateImageDetail(state.imageModel);

  if (!state.aiKey) openModal();
  addBubble("sys", "Ready. Enter runs. Shift+Enter makes a new line.");
}
init();
