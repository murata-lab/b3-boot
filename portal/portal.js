"use strict";

const menu = document.querySelector("#menu");
const sidebar = document.querySelector("#sidebar");
const home = document.querySelector("#home");
const lessons = document.querySelector("#lessons");
const status = document.querySelector("#status");
const statusMessage = document.querySelector("#status-message");
const retry = document.querySelector("#retry");
const frames = new Map();
const pending = new Map();
let chapters = [];
let routeVersion = 0;

menu.addEventListener("click", () => {
  sidebar.showModal();
  menu.setAttribute("aria-expanded", "true");
});
document.querySelector("#close-menu").addEventListener("click", () => sidebar.close());
sidebar.addEventListener("close", () => {
  menu.setAttribute("aria-expanded", "false");
  menu.focus();
});
sidebar.addEventListener("click", (event) => {
  if (event.target.closest("a")) sidebar.close();
  if (event.target === sidebar && event.clientX > sidebar.getBoundingClientRect().right) sidebar.close();
});

function showError(message) {
  status.hidden = false;
  statusMessage.textContent = message;
  retry.hidden = false;
}

async function openChapter(chapter) {
  if (frames.has(chapter.id)) return frames.get(chapter.id);
  if (pending.has(chapter.id)) return pending.get(chapter.id);
  const request = (async () => {
    const response = await fetch(`/api/chapters/${chapter.id}`, {method: "POST"});
    const data = await response.json();
    if (!response.ok) throw new Error(data.error);
    const frame = document.createElement("iframe");
    frame.title = `${chapter.number} ${chapter.title}`;
    frame.hidden = true;
    frame.src = data.url;
    lessons.append(frame);
    frames.set(chapter.id, frame);
    return frame;
  })();
  pending.set(chapter.id, request);
  try { return await request; }
  finally { pending.delete(chapter.id); }
}

async function route() {
  const version = ++routeVersion;
  const id = location.hash.replace(/^#\/?/, "");
  const chapter = chapters.find((item) => item.id === id && item.mode);
  for (const frame of frames.values()) frame.hidden = true;
  home.hidden = Boolean(id);
  lessons.hidden = true;
  status.hidden = true;
  retry.hidden = true;
  const title = chapter ? `${chapter.number} ${chapter.title}` : "教材一覧";
  document.querySelector("#current-title").textContent = title;
  document.title = chapter ? `${title} | 機械学習入門` : "機械学習入門";
  for (const link of sidebar.querySelectorAll("a")) {
    if (link.hash === (id ? `#/${id}` : "#/")) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  }
  if (!id) return;
  if (!chapter) {
    status.hidden = false;
    statusMessage.textContent = "この教材はまだ公開されていません。左上のメニューから教材を選んでください。";
    return;
  }
  status.hidden = false;
  statusMessage.textContent = `${title} を開いています…`;
  try {
    const frame = await openChapter(chapter);
    if (version !== routeVersion) return;
    frame.hidden = false;
    lessons.hidden = false;
    status.hidden = true;
  } catch (error) {
    if (version === routeVersion) showError(error.message || "接続できませんでした。起動したターミナルを確認してください。");
  }
}

async function initialize() {
  try {
    const response = await fetch("/api/chapters");
    if (!response.ok) throw new Error("教材一覧を取得できませんでした。");
    chapters = await response.json();
    for (const chapter of chapters) {
      const card = document.createElement(chapter.mode ? "a" : "article");
      card.className = `chapter ${chapter.mode ? "available" : "pending"}`;
      if (chapter.mode) card.href = `#/${chapter.id}`;
      const number = document.createElement("span");
      number.className = "chapter-number";
      number.textContent = chapter.number;
      const body = document.createElement("div");
      const title = document.createElement("h3");
      title.textContent = chapter.title;
      const description = document.createElement("p");
      description.textContent = chapter.description;
      body.append(title, description);
      if (!chapter.mode) {
        const badge = document.createElement("span");
        badge.className = "badge";
        badge.textContent = "準備中";
        body.append(badge);
      }
      card.append(number, body);
      document.querySelector("#chapters").append(card);
      const link = document.createElement(chapter.mode ? "a" : "div");
      if (chapter.mode) link.href = `#/${chapter.id}`;
      else link.className = "nav-pending";
      const navNumber = document.createElement("span");
      navNumber.className = "nav-number";
      navNumber.textContent = chapter.number;
      link.append(navNumber, document.createTextNode(chapter.title));
      if (!chapter.mode) {
        const label = document.createElement("small");
        label.textContent = "準備中";
        link.append(label);
      }
      document.querySelector("#navigation").append(link);
    }
    await route();
  } catch (_error) {
    home.hidden = true;
    showError("教材一覧に接続できませんでした。起動したターミナルを確認して、ページを再読み込みしてください。");
  }
}

retry.addEventListener("click", () => chapters.length ? route() : location.reload());
window.addEventListener("hashchange", route);
initialize();
