(function () {
  const PANEL_ID = "gsxt-assistant-panel";
  const TASKS_KEY = "gsxtTasks";
  const LOGS_KEY = "gsxtFlowLogs";
  const TAB_STATE_KEY = "__gsxt_send_report_assistant__";
  const MAX_AUTO_STEPS = 40;
  const MAX_AUTO_MS = 5 * 60 * 1000;
  const MAX_WAIT_MS = 2 * 60 * 1000;
  const MAX_LOG_ROWS = 2000;
  const CAPTCHA_RETRY_COOLDOWN_MS = 8 * 1000;
  const CAPTCHA_DEBUG_STEP_DELAY_MS = 1000;
  const CAPTCHA_EXPECTED_POINTS = 3;
  const MAX_CAPTCHA_IMAGE_REFRESHES = 5;
  const MAX_PAGE_RECOVERY_REFRESHES = 3;
  const PAGE_RECOVERY_REFRESH_MS = 3000;
  const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  let busy = false;
  let running = false;
  let submittedKeyword = "";
  let runGeneration = 0;
  const scheduledTimers = new Set();

  class CaptchaRefreshRequested extends Error {
    constructor(message, details = {}) {
      super(message);
      this.name = "CaptchaRefreshRequested";
      this.details = details;
    }
  }

  function makeRunId(prefix = "run") {
    const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    return `${prefix}-${stamp}-${Math.random().toString(36).slice(2, 7)}`;
  }

  function readTabState() {
    try {
      const parsed = JSON.parse(window.name || "{}");
      return parsed && parsed[TAB_STATE_KEY] ? parsed[TAB_STATE_KEY] : {};
    } catch {
      return {};
    }
  }

  function writeTabState(patch) {
    let parsed = {};
    try {
      parsed = JSON.parse(window.name || "{}") || {};
    } catch {
      parsed = {};
    }
    const next = { ...(parsed[TAB_STATE_KEY] || {}), ...patch };
    parsed[TAB_STATE_KEY] = next;
    window.name = JSON.stringify(parsed);
    return next;
  }

  function clearTabState() {
    try {
      const parsed = JSON.parse(window.name || "{}") || {};
      delete parsed[TAB_STATE_KEY];
      window.name = Object.keys(parsed).length ? JSON.stringify(parsed) : "";
    } catch {
      window.name = "";
    }
  }

  function clearScheduledTimers() {
    for (const timer of scheduledTimers) {
      clearTimeout(timer);
    }
    scheduledTimers.clear();
  }

  function currentRunToken() {
    return runGeneration;
  }

  function isRunTokenActive(token) {
    return running && token === runGeneration && autoStateIsValid();
  }

  function assertRunNotStopped(token) {
    if (token == null) return;
    if (!running || token !== runGeneration) {
      throw new Error("自动流程已停止");
    }
  }

  function textOf(el) {
    return (el && (el.innerText || el.textContent || "")).replace(/\s+/g, " ").trim();
  }

  function isVisible(el) {
    if (!el) return false;
    const style = window.getComputedStyle(el);
    const rect = el.getBoundingClientRect();
    return style.display !== "none" && style.visibility !== "hidden" && rect.width > 0 && rect.height > 0;
  }

  function isInPanel(el) {
    return Boolean(el && el.closest && el.closest(`#${PANEL_ID}`));
  }

  function clickableTarget(el) {
    let current = el;
    for (let depth = 0; current && current !== document.body && depth < 6; depth += 1) {
      const tag = (current.tagName || "").toLowerCase();
      const className = String(current.className || "");
      if (
        tag === "a"
        || tag === "button"
        || current.onclick
        || current.hasAttribute("onclick")
        || current.getAttribute("role") === "button"
        || current.hasAttribute("data-toggle")
        || /(btn|button|dropdown|menu|more|cursor|pointer|operate)/i.test(className)
      ) {
        return current;
      }
      current = current.parentElement;
    }
    return el;
  }

  function byText(text) {
    const nodes = Array.from(document.querySelectorAll("button, a, div, span, li, p"))
      .filter((el) => !isInPanel(el));
    return nodes.find((el) => isVisible(el) && textOf(el) === text)
      || nodes.find((el) => isVisible(el) && textOf(el).includes(text));
  }

  function clickElement(el, useClickableAncestor = true) {
    if (!el) throw new Error("element not found");
    const target = useClickableAncestor ? clickableTarget(el) : el;
    target.scrollIntoView({ block: "center", inline: "center" });
    const rect = target.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2;
    for (const type of ["mouseover", "mousemove", "mousedown", "mouseup", "click"]) {
      target.dispatchEvent(new MouseEvent(type, { bubbles: true, cancelable: true, view: window, clientX: x, clientY: y }));
    }
    if (typeof target.click === "function") target.click();
  }

  function setInputValue(input, value) {
    input.focus();
    const setter = Object.getOwnPropertyDescriptor(HTMLInputElement.prototype, "value").set;
    setter.call(input, value);
    for (const type of ["input", "change", "propertychange", "keyup"]) {
      input.dispatchEvent(new Event(type, { bubbles: true, cancelable: true }));
    }
  }

  function getTextarea() {
    return document.querySelector(`#${PANEL_ID} [data-ga-tasks]`);
  }

  function getTasks() {
    return (getTextarea()?.value || "").split(/\r?\n/).map((line) => line.trim()).filter(Boolean);
  }

  function setTasks(tasks) {
    const text = tasks.join("\n");
    const textarea = getTextarea();
    if (textarea) textarea.value = text;
    chrome.storage.local.set({ [TASKS_KEY]: text });
  }

  function setStatus(message) {
    const status = document.querySelector(`#${PANEL_ID} .ga-status`);
    if (status) status.textContent = message;
  }

  async function debugStatus(message, delayMs = CAPTCHA_DEBUG_STEP_DELAY_MS) {
    setStatus(message);
    if (delayMs > 0) await sleep(delayMs);
  }

  function pickCaptchaField(source, patterns) {
    if (!source) return "";
    for (const [key, value] of Object.entries(source)) {
      if (patterns.some((pattern) => pattern.test(key))) return String(value || "");
    }
    return "";
  }

  function getRunId() {
    const state = readTabState();
    if (state.runId) return state.runId;
    return running ? makeRunId("run") : makeRunId("manual");
  }

  function collectPageIdentifiers(pageType) {
    const identifiers = {
      url: location.href,
      title: document.title || "",
      keyword: document.querySelector("#keyword")?.value?.trim() || currentTask() || "",
      pageType,
      domIds: []
    };

    for (const id of [
      "keyword",
      "btn_query",
      "moreActionsToggle",
      "moreActionsMenu",
      "btn_send_pdf",
      "entNameForSubscribe",
      "entTypeForHistoryScan",
      "captcha",
      "captchaId",
      "risk_id",
      "riskType"
    ]) {
      const el = document.getElementById(id);
      if (!el || isInPanel(el)) continue;
      identifiers.domIds.push(id);
      if ("value" in el && el.value) identifiers[id] = shortValue(el.value);
      if (el.href) identifiers[`${id}_href`] = shortValue(el.href);
    }

    const text = pageText();
    const codeMatch = text.match(/统一社会信用代码[:：\s]*([0-9A-Z]{15,20})/);
    if (codeMatch) identifiers.unifiedSocialCreditCode = codeMatch[1];

    const companyName = document.querySelector("#entNameForSubscribe")?.value
      || Array.from(document.querySelectorAll("h1, .companyDetail, .nameBox")).map(textOf).find(Boolean)
      || "";
    if (companyName) identifiers.companyName = shortValue(companyName);

    const captchaMeta = collectCaptchaMeta();
    if (Object.keys(captchaMeta).length) identifiers.captchaMeta = captchaMeta;

    return identifiers;
  }

  function appendLog(eventName, pageType, extra = {}) {
    const row = {
      time: new Date().toLocaleString("zh-CN", { hour12: false }),
      isoTime: new Date().toISOString(),
      runId: getRunId(),
      pageType,
      event: eventName,
      task: currentTask(),
      identifiers: collectPageIdentifiers(pageType),
      ...extra
    };

    chrome.storage.local.get([LOGS_KEY], (data) => {
      const logs = Array.isArray(data[LOGS_KEY]) ? data[LOGS_KEY] : [];
      logs.push(row);
      chrome.storage.local.set({ [LOGS_KEY]: logs.slice(-MAX_LOG_ROWS) });
    });
  }

  function csvCell(value) {
    const text = typeof value === "string" ? value : JSON.stringify(value ?? "");
    return `"${text.replace(/"/g, '""')}"`;
  }

  function exportLogs() {
    chrome.storage.local.get([LOGS_KEY], (data) => {
      const logs = Array.isArray(data[LOGS_KEY]) ? data[LOGS_KEY] : [];
      if (!logs.length) {
        setStatus("暂无日志可导出。");
        return;
      }
      const header = [
        "time",
        "runId",
        "pageType",
        "event",
        "task",
        "url",
        "companyName",
        "unifiedSocialCreditCode",
        "captcha_id",
        "risk_type",
        "risk_id",
        "lot_number",
        "process_token",
        "identifiers_json"
      ];
      const lines = [header.map(csvCell).join(",")];
      for (const row of logs) {
        const identifiers = row.identifiers || {};
        const captchaMeta = identifiers.captchaMeta || {};
        lines.push([
          row.time || row.isoTime || "",
          row.runId || "",
          row.pageType || "",
          row.event || "",
          row.task || "",
          identifiers.url || "",
          identifiers.companyName || "",
          identifiers.unifiedSocialCreditCode || "",
          pickCaptchaField(captchaMeta, [/captcha[_-]?id/i]),
          pickCaptchaField(captchaMeta, [/risk[_-]?type/i]),
          pickCaptchaField(captchaMeta, [/risk[_-]?id/i]),
          pickCaptchaField(captchaMeta, [/lot[_-]?number/i]),
          pickCaptchaField(captchaMeta, [/process[_-]?token/i]),
          identifiers
        ].map(csvCell).join(","));
      }
      const blob = new Blob([`\ufeff${lines.join("\r\n")}`], { type: "text/csv;charset=utf-8" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `gsxt_flow_log_${new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14)}.csv`;
      document.documentElement.appendChild(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 3000);
      setStatus(`已导出 ${logs.length} 条日志为 CSV，可用 Excel 打开。`);
    });
  }

  function clearLogs() {
    chrome.storage.local.set({ [LOGS_KEY]: [] }, () => setStatus("日志已清空。"));
  }

  function saveTasks() {
    chrome.storage.local.set({ [TASKS_KEY]: getTextarea()?.value || "" });
  }

  function setRunning(value) {
    running = value;
    if (value) {
      const state = readTabState();
      writeTabState({
        running: true,
        startedAt: state.startedAt || Date.now(),
        steps: state.steps || 0,
        submittedKeyword,
        waitStartedAt: state.waitStartedAt || 0
      });
    } else {
      runGeneration += 1;
      clearScheduledTimers();
      clearTabState();
    }
  }

  function autoStateIsValid() {
    const state = readTabState();
    if (!state.running) return false;
    const elapsed = Date.now() - (state.startedAt || Date.now());
    if (elapsed > MAX_AUTO_MS || (state.steps || 0) >= MAX_AUTO_STEPS) {
      setRunning(false);
      setStatus(`自动流程已停止：达到安全上限。\n已运行 ${Math.round(elapsed / 1000)} 秒，执行 ${state.steps || 0} 步。`);
      return false;
    }
    return true;
  }

  function markAutoAction(actionName, pageType = classifyPage()) {
    if (!running) {
      appendLog(actionName, pageType, { mode: "manual" });
      return true;
    }
    if (!autoStateIsValid()) return false;
    const state = readTabState();
    writeTabState({
      steps: (state.steps || 0) + 1,
      lastAction: actionName,
      lastActionAt: Date.now(),
      waitStartedAt: 0
    });
    appendLog(actionName, pageType, { mode: "auto", step: (state.steps || 0) + 1 });
    return true;
  }

  function pauseForManual(message) {
    appendLog("pause_for_manual", classifyPage(), { message });
    setRunning(false);
    setStatus(`${message}\n自动流程已暂停，处理完成后可重新点击“开始”。`);
  }

  function currentTask() {
    return getTasks()[0] || "";
  }

  function shortValue(value) {
    const text = String(value ?? "").trim();
    if (!text) return "";
    return text.length > 80 ? `${text.slice(0, 77)}...` : text;
  }

  function putMeta(meta, key, value) {
    const normalized = String(key || "").trim();
    const text = shortValue(value);
    if (!normalized || !text) return;
    if (!/(captcha|geetest|risk|lot|token|challenge|process|payload)/i.test(normalized)) return;
    if (!meta[normalized]) meta[normalized] = text;
  }

  function collectMetaFromUrl(meta, value) {
    if (!value || typeof value !== "string") return;
    try {
      const url = new URL(value, location.href);
      for (const [key, val] of url.searchParams.entries()) {
        putMeta(meta, key, val);
      }
    } catch {
      // Ignore non-URL strings.
    }
  }

  function collectCaptchaMeta() {
    const meta = {};

    for (const input of Array.from(document.querySelectorAll("input, textarea"))) {
      if (isInPanel(input)) continue;
      putMeta(meta, input.id, input.value);
      putMeta(meta, input.name, input.value);
      for (const attr of Array.from(input.attributes || [])) {
        putMeta(meta, attr.name, attr.value);
      }
    }

    for (const el of Array.from(document.querySelectorAll("iframe, script, img, link, div, span"))) {
      if (isInPanel(el)) continue;
      for (const attr of Array.from(el.attributes || [])) {
        putMeta(meta, attr.name, attr.value);
        collectMetaFromUrl(meta, attr.value);
      }
    }

    for (const script of Array.from(document.scripts || [])) {
      const text = script.textContent || "";
      if (!/(captcha|geetest|risk_id|riskType|risk_type|captcha_id)/i.test(text)) continue;
      for (const match of text.matchAll(/["']?([A-Za-z0-9_]*(?:captcha|geetest|risk|lot|token|challenge|process|payload)[A-Za-z0-9_]*)["']?\s*[:=]\s*["']([^"']{1,200})["']/gi)) {
        putMeta(meta, match[1], match[2]);
      }
    }

    for (const key of Object.getOwnPropertyNames(window).filter((name) => /(captcha|geetest|risk|lot)/i.test(name)).slice(0, 80)) {
      try {
        const value = window[key];
        if (typeof value === "string" || typeof value === "number" || typeof value === "boolean") {
          putMeta(meta, key, value);
        } else if (value && typeof value === "object") {
          for (const [childKey, childValue] of Object.entries(value).slice(0, 40)) {
            if (typeof childValue === "string" || typeof childValue === "number" || typeof childValue === "boolean") {
              putMeta(meta, `${key}.${childKey}`, childValue);
            }
          }
        }
      } catch {
        // Some window properties can throw on access.
      }
    }

    return meta;
  }

  function formatCaptchaMeta(meta) {
    const entries = Object.entries(meta || {})
      .filter(([key]) => /(risk_id|riskId|risk_type|riskType|captcha_id|captchaId|lot_number|lotNumber|gt|challenge|geetest)/i.test(key))
      .slice(0, 8);
    if (!entries.length) return "";
    return entries.map(([key, value]) => `${key}: ${value}`).join("\n");
  }

  function detectCaptcha() {
    const selectors = [
      "iframe[src*='geetest']",
      "iframe[src*='captcha']",
      "[class^='geetest']",
      "[class*=' geetest']",
      "[id^='geetest']",
      "[id*='geetest']",
      "[class^='captcha']",
      "[class*=' captcha']",
      "[id^='captcha']"
    ];
    const nodes = selectors
      .flatMap((selector) => Array.from(document.querySelectorAll(selector)))
      .filter((el) => !isInPanel(el) && isVisible(el));

    const text = pageText();
    const textHit = /请完成验证|智能验证|安全验证|拖动滑块|点击验证|正在验证|验证失败|GeeTest|geetest/i.test(text)
      && /验证码|验证|滑块|captcha|geetest/i.test(text);
    const meta = collectCaptchaMeta();

    return {
      visible: nodes.length > 0 || textHit,
      kind: nodes[0]?.tagName?.toLowerCase() || (textHit ? "text" : ""),
      element: nodes[0] || null,
      meta
    };
  }

  function findCaptchaElement() {
    const selectors = [
      "iframe[src*='geetest']",
      "iframe[src*='captcha']",
      "[class^='geetest']",
      "[class*=' geetest']",
      "[id^='geetest']",
      "[id*='geetest']",
      "[class^='captcha']",
      "[class*=' captcha']",
      "[id^='captcha']"
    ];
    const viewportArea = Math.max(1, window.innerWidth * window.innerHeight);
    const candidates = selectors
      .flatMap((selector) => Array.from(document.querySelectorAll(selector)))
      .filter((el, index, arr) => arr.indexOf(el) === index)
      .filter((el) => !isInPanel(el) && isVisible(el))
      .map((el) => {
        const rect = el.getBoundingClientRect();
        const area = rect.width * rect.height;
        const classId = `${el.id || ""} ${String(el.className || "")}`;
        const isTooLarge = area > viewportArea * 0.45
          || rect.width > window.innerWidth * 0.82
          || rect.height > window.innerHeight * 0.82;
        const looksDialog = /(box|panel|window|widget|wrap|content|container|popup|captcha)/i.test(classId);
        const looksMask = /(mask|overlay|full|shade|bg)/i.test(classId);
        const reasonableSize = rect.width >= 220 && rect.height >= 180
          && rect.width <= Math.max(900, window.innerWidth * 0.7)
          && rect.height <= Math.max(800, window.innerHeight * 0.75);
        const centerX = rect.left + rect.width / 2;
        const centerY = rect.top + rect.height / 2;
        const centered = Math.abs(centerX - window.innerWidth / 2) < window.innerWidth * 0.35
          && Math.abs(centerY - window.innerHeight / 2) < window.innerHeight * 0.35;
        let score = 0;
        if (reasonableSize) score += 100;
        if (looksDialog) score += 40;
        if (centered) score += 25;
        if (rect.width >= 300 && rect.height >= 250) score += 20;
        score += Math.min(area / 10000, 40);
        if (isTooLarge) score -= 300;
        if (looksMask) score -= 80;
        return { el, rect, score, isTooLarge };
      })
      .filter((item) => !item.isTooLarge || item.score > 0);
    if (candidates.length) {
      return candidates
        .sort((a, b) => b.score - a.score)[0].el;
    }
    return null;
  }

  function captureVisibleTab() {
    return new Promise(async (resolve, reject) => {
      await debugStatus("正在请求浏览器截图权限...");
      let settled = false;
      const timer = setTimeout(() => {
        if (settled) return;
        settled = true;
        reject(new Error("captureVisibleTab timeout: background did not respond. 请在 edge://extensions 重新加载扩展，并刷新当前网页。"));
      }, 5000);
      chrome.runtime.sendMessage({ type: "GSXT_CAPTURE_VISIBLE_TAB" }, async (response) => {
        if (settled) return;
        settled = true;
        clearTimeout(timer);
        if (chrome.runtime.lastError) {
          reject(new Error(chrome.runtime.lastError.message));
          return;
        }
        if (!response || !response.ok) {
          reject(new Error(response?.error || "captureVisibleTab failed"));
          return;
        }
        await debugStatus("浏览器截图成功，正在加载截图...");
        resolve(response.dataUrl);
      });
    });
  }

  function loadImage(dataUrl) {
    return new Promise((resolve, reject) => {
      const img = new Image();
      img.onload = () => resolve(img);
      img.onerror = () => reject(new Error("failed to load captured image"));
      img.src = dataUrl;
    });
  }

  function captchaDebugId() {
    const stamp = new Date().toISOString().replace(/[-:.TZ]/g, "").slice(0, 14);
    return `captcha_${stamp}_${Math.random().toString(36).slice(2, 7)}`;
  }

  function rectSnapshot(el) {
    if (!el) return null;
    const rect = el.getBoundingClientRect();
    return {
      left: Math.round(rect.left),
      top: Math.round(rect.top),
      width: Math.round(rect.width),
      height: Math.round(rect.height)
    };
  }

  function sameRect(a, b, tolerance = 2) {
    if (!a || !b) return false;
    return Math.abs(a.left - b.left) <= tolerance
      && Math.abs(a.top - b.top) <= tolerance
      && Math.abs(a.width - b.width) <= tolerance
      && Math.abs(a.height - b.height) <= tolerance;
  }

  async function waitForStableCaptchaElement(el, timeoutMs = 3500) {
    if (!el) throw new Error("captcha element not found");
    const deadline = Date.now() + timeoutMs;
    let last = null;
    let best = null;
    let stableCount = 0;
    await debugStatus("已找到验证码区域，等待渲染稳定...");

    while (Date.now() < deadline) {
      const current = rectSnapshot(el);
      if (current && current.width >= 80 && current.height >= 60) {
        best = current;
      }
      if (current && current.width >= 80 && current.height >= 60 && sameRect(last, current)) {
        stableCount += 1;
        if (stableCount >= 3) {
          await sleep(500);
          await debugStatus(`验证码区域稳定：${current.width}x${current.height}，准备截图...`);
          return current;
        }
      } else {
        stableCount = 0;
      }
      last = current;
      await sleep(180);
    }

    const finalRect = best || rectSnapshot(el);
    if (!finalRect || finalRect.width < 80 || finalRect.height < 60) {
      throw new Error(`captcha element is not stable or too small: ${JSON.stringify(finalRect)}`);
    }
    await sleep(500);
    await debugStatus(`验证码区域未完全稳定，但尺寸可用：${finalRect.width}x${finalRect.height}，继续截图...`);
    return finalRect;
  }

  async function cropElementScreenshot(el) {
    if (!el) throw new Error("captcha element not found");
    await waitForStableCaptchaElement(el);
    const panelRect = el.getBoundingClientRect();
    const pad = 6;
    const confirmButton = findCaptchaConfirmButton(el);
    const confirmRect = confirmButton ? confirmButton.getBoundingClientRect() : null;
    let cropLeft = panelRect.left - pad;
    let cropTop = panelRect.top - pad;
    let cropRight = panelRect.right + pad;
    let cropBottom = panelRect.bottom + pad;
    let excludedControl = null;

    if (
      confirmRect
      && confirmRect.top > panelRect.top + panelRect.height * 0.35
      && confirmRect.top < panelRect.bottom
      && confirmRect.left < panelRect.right
      && confirmRect.right > panelRect.left
    ) {
      cropBottom = Math.max(
        panelRect.top + Math.min(180, panelRect.height * 0.45),
        confirmRect.top - 8
      );
      excludedControl = {
        reason: "confirm_button",
        target: describeElement(confirmButton),
        text: textOf(confirmButton) || String(confirmButton.value || ""),
        viewportRect: {
          left: Math.round(confirmRect.left),
          top: Math.round(confirmRect.top),
          width: Math.round(confirmRect.width),
          height: Math.round(confirmRect.height)
        }
      };
    } else {
      const fallbackBottom = panelRect.top + panelRect.height * 0.70;
      if (fallbackBottom > panelRect.top + Math.min(180, panelRect.height * 0.45)) {
        cropBottom = fallbackBottom;
        excludedControl = {
          reason: "ratio_fallback",
          target: "<not found>",
          text: "",
          viewportRect: null
        };
      }
    }

    const cropCssWidth = Math.max(1, cropRight - cropLeft);
    const cropCssHeight = Math.max(1, cropBottom - cropTop);
    const dataUrl = await captureVisibleTab();
    if (!dataUrl || !dataUrl.startsWith("data:image/")) {
      throw new Error("captureVisibleTab returned invalid image data");
    }
    const image = await loadImage(dataUrl);
    await debugStatus(`截图加载成功：${image.naturalWidth}x${image.naturalHeight}，正在裁剪验证码...`);
    const scaleX = image.naturalWidth / Math.max(1, window.innerWidth);
    const scaleY = image.naturalHeight / Math.max(1, window.innerHeight);
    const sx = Math.max(0, Math.floor(cropLeft * scaleX));
    const sy = Math.max(0, Math.floor(cropTop * scaleY));
    const sw = Math.min(image.naturalWidth - sx, Math.ceil(cropCssWidth * scaleX));
    const sh = Math.min(image.naturalHeight - sy, Math.ceil(cropCssHeight * scaleY));
    if (sw <= 20 || sh <= 20) throw new Error(`captcha crop too small: ${sw}x${sh}`);

    const canvas = document.createElement("canvas");
    canvas.width = sw;
    canvas.height = sh;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(image, sx, sy, sw, sh, 0, 0, sw, sh);
    await debugStatus(`验证码裁剪成功：${sw}x${sh}，正在保存调试图...`);
    return {
      imageBase64: canvas.toDataURL("image/png"),
      rect: { left: cropLeft, top: cropTop, width: sw / scaleX, height: sh / scaleY },
      cropInfo: {
        sourceElement: describeElement(el),
        solverCropMode: "challenge_only",
        viewportRect: {
          left: Math.round(panelRect.left),
          top: Math.round(panelRect.top),
          width: Math.round(panelRect.width),
          height: Math.round(panelRect.height)
        },
        solverViewportRect: {
          left: Math.round(cropLeft),
          top: Math.round(cropTop),
          width: Math.round(sw / scaleX),
          height: Math.round(sh / scaleY)
        },
        excludedControl,
        cropPixels: { x: sx, y: sy, width: sw, height: sh },
        screenshotPixels: { width: image.naturalWidth, height: image.naturalHeight },
        scale: { x: scaleX, y: scaleY }
      },
      scaleX,
      scaleY
    };
  }

  function describeElement(el) {
    if (!el) return "<none>";
    const tag = (el.tagName || "").toLowerCase();
    const id = el.id ? `#${el.id}` : "";
    const cls = String(el.className || "")
      .replace(/\s+/g, ".")
      .slice(0, 80);
    return `${tag}${id}${cls ? `.${cls}` : ""}`.slice(0, 140);
  }

  function clickViewportPoint(x, y) {
    const target = document.elementFromPoint(x, y) || document.body;
    const common = {
      bubbles: true,
      cancelable: true,
      view: window,
      clientX: x,
      clientY: y
    };
    for (const type of ["pointerover", "pointerenter", "pointermove", "pointerdown", "pointerup"]) {
      try {
        target.dispatchEvent(new PointerEvent(type, {
          ...common,
          pointerId: 1,
          pointerType: "mouse",
          isPrimary: true,
          button: 0,
          buttons: type === "pointerdown" ? 1 : 0
        }));
      } catch {
        // Some older browser contexts may not allow PointerEvent construction.
      }
    }
    for (const type of ["mouseover", "mousemove", "mousedown", "mouseup", "click"]) {
      target.dispatchEvent(new MouseEvent(type, {
        ...common,
        button: 0,
        buttons: type === "mousedown" ? 1 : 0
      }));
    }
    return describeElement(target);
  }

  function findCaptchaConfirmButton(root) {
    const scope = root || document;
    const nodes = Array.from(scope.querySelectorAll("button, div, span, a, input"))
      .filter((el) => !isInPanel(el) && isVisible(el));
    return nodes.find((el) => textOf(el) === "确定")
      || nodes.find((el) => textOf(el).includes("确定") && textOf(el).length <= 8)
      || nodes.find((el) => String(el.value || "").includes("确定"))
      || nodes.find((el) => /(submit|confirm|commit|btn)/i.test(`${el.id || ""} ${String(el.className || "")}`) && textOf(el).length <= 12);
  }

  function clickCaptchaConfirm(root) {
    const confirmButton = findCaptchaConfirmButton(root);
    if (confirmButton) {
      clickElement(confirmButton, false);
      return {
        method: "text_or_selector",
        target: describeElement(confirmButton),
        text: textOf(confirmButton) || String(confirmButton.value || "")
      };
    }

    if (!root) return null;
    const rect = root.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height * 0.74;
    const target = document.elementFromPoint(x, y) || root;
    const targetDesc = clickViewportPoint(x, y);
    return {
      method: "geometry_fallback",
      target: targetDesc,
      text: textOf(target),
      viewportPoint: { x: Math.round(x), y: Math.round(y) }
    };
  }

  function findCaptchaRefreshButton(root) {
    const scope = root || document;
    const selectors = [
      "[class*='refresh']",
      "[class*='reload']",
      "[class*='change']",
      "[class*='switch']",
      "[aria-label*='刷新']",
      "[aria-label*='换']",
      "[title*='刷新']",
      "[title*='换']"
    ];
    for (const selector of selectors) {
      const hit = Array.from(scope.querySelectorAll(selector))
        .find((el) => !isInPanel(el) && isVisible(el));
      if (hit) return hit;
    }
    const nodes = Array.from(scope.querySelectorAll("button, div, span, a, i, svg"))
      .filter((el) => !isInPanel(el) && isVisible(el));
    return nodes.find((el) => /refresh|reload|change|switch|换|刷新/i.test(
      `${el.id || ""} ${String(el.className || "")} ${textOf(el)} ${el.getAttribute?.("aria-label") || ""} ${el.getAttribute?.("title") || ""}`
    )) || null;
  }

  async function clickCaptchaRefresh(root, reason = "solver_result_not_three") {
    const refreshButton = findCaptchaRefreshButton(root);
    if (refreshButton) {
      clickElement(refreshButton, false);
      return {
        method: "selector_or_text",
        target: describeElement(refreshButton),
        text: textOf(refreshButton) || String(refreshButton.value || ""),
        reason
      };
    }

    if (!root) return null;
    const rect = root.getBoundingClientRect();
    const x = rect.left + rect.width * 0.20;
    const y = rect.top + rect.height * 0.90;
    const targetDesc = clickViewportPoint(x, y);
    return {
      method: "geometry_left_bottom_second",
      target: targetDesc,
      text: "",
      reason,
      viewportPoint: { x: Math.round(x), y: Math.round(y) }
    };
  }

  async function postClickReport(payload) {
    try {
      const response = await fetch("http://127.0.0.1:7755/click-report", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        setStatus(`点击诊断上报失败：HTTP ${response.status}\n请确认 server.py 已重启到最新版。`);
      }
    } catch (err) {
      setStatus(`点击诊断上报失败：${err?.message || err}`);
    }
  }

  async function solveCaptchaWithGsxt(captcha, runToken = null) {
    assertRunNotStopped(runToken);
    const el = findCaptchaElement() || captcha.element;
    if (!el) throw new Error("captcha element not found before screenshot");
    const crop = await cropElementScreenshot(el);
    assertRunNotStopped(runToken);
    const debugId = captchaDebugId();
    const captureResponse = await fetch("http://127.0.0.1:7755/debug-capture", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        image_base64: crop.imageBase64,
        debug_id: debugId,
        reason: "before_solve",
        crop_info: crop.cropInfo
      })
    });
    let captureResult = null;
    try {
      captureResult = await captureResponse.json();
    } catch {
      captureResult = null;
    }
    if (!captureResponse.ok || !captureResult?.success) {
      throw new Error(`截图已裁剪，但保存调试图失败：HTTP ${captureResponse.status} ${captureResult?.error || "no JSON body"}`);
    }
    await debugStatus(`截图保存成功：${captureResult.debug_image}\n正在调用模型识别...`);
    assertRunNotStopped(runToken);

    const response = await fetch("http://127.0.0.1:7755/solve", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ image_base64: crop.imageBase64, timeout: 300, debug_id: debugId, crop_info: crop.cropInfo })
    });
    let result = null;
    try {
      result = await response.json();
    } catch {
      result = null;
    }
    if (!response.ok) {
      throw new Error(`GSXT solver HTTP ${response.status}: ${result?.error || "no JSON body"}${result?.debug_image ? `\ndebug_image: ${result.debug_image}` : ""}`);
    }
    if (!result.success) {
      throw new Error(`${result.error || "GSXT solver failed"}${result.debug_image ? `\ndebug_image: ${result.debug_image}` : ""}`);
    }
    const points = Array.isArray(result.points) ? result.points : [];
    if (points.length !== CAPTCHA_EXPECTED_POINTS) {
      throw new CaptchaRefreshRequested(
        `GSXT solver returned ${points.length} points, expected ${CAPTCHA_EXPECTED_POINTS}${result.debug_image ? `\ndebug_image: ${result.debug_image}` : ""}`,
        {
          debug_id: result.debug_id,
          debug_image: result.debug_image,
          task: result.task,
          sequence: result.sequence,
          points,
          crop_info: crop.cropInfo
        }
      );
    }

    await debugStatus(
      `模型识别成功\n`
      + `类型：${result.task?.type || "-"} / ${result.task?.action || "-"}\n`
      + `顺序：${(result.sequence || []).join(" → ") || "-"}\n`
      + `图片点位：${points.map((p) => `(${p.x},${p.y})`).join(" → ")}`
    );
    assertRunNotStopped(runToken);

    const clickResults = [];
    for (let i = 0; i < points.length; i += 1) {
      assertRunNotStopped(runToken);
      const point = points[i];
      const cssPointX = Number(point.x) / Math.max(0.0001, crop.scaleX);
      const cssPointY = Number(point.y) / Math.max(0.0001, crop.scaleY);
      const x = crop.rect.left + cssPointX;
      const y = crop.rect.top + cssPointY;
      const targetDesc = clickViewportPoint(x, y);
      const row = {
        index: i + 1,
        label: result.sequence?.[i] || "",
        imagePoint: { x: Number(point.x), y: Number(point.y) },
        cssPoint: { x: Math.round(cssPointX), y: Math.round(cssPointY) },
        viewportPoint: { x: Math.round(x), y: Math.round(y) },
        target: targetDesc
      };
      clickResults.push(row);
      if (/^(iframe|body|html)/i.test(row.target)) {
        await postClickReport({
          debug_id: result.debug_id,
          stage: "suspicious_click_target",
          warning: "click target may not receive captcha events",
          click: row,
          crop_info: crop.cropInfo
        });
      }
      setStatus(
        `点击 ${row.index}/${points.length}: ${row.label || "-"}\n`
        + `图片坐标：(${row.imagePoint.x}, ${row.imagePoint.y})\n`
        + `CSS偏移：(${row.cssPoint.x}, ${row.cssPoint.y})\n`
        + `网页坐标：(${row.viewportPoint.x}, ${row.viewportPoint.y})\n`
        + `命中元素：${row.target}`
      );
      await postClickReport({
        debug_id: result.debug_id,
        stage: "target_click",
        click: row,
        crop_info: crop.cropInfo
      });
      await sleep(900);
    }
    assertRunNotStopped(runToken);
    result.click_results = clickResults;
    await postClickReport({
      debug_id: result.debug_id,
      stage: "before_confirm",
      message: "target clicks finished, about to click confirm",
      click_results: clickResults,
      crop_info: crop.cropInfo
    });
    assertRunNotStopped(runToken);
    const confirmRow = clickCaptchaConfirm(el);
    if (confirmRow) {
      result.confirm_click = { index: "confirm", label: "确定", ...confirmRow };
      setStatus(
        `已点击验证码确认按钮\n`
        + `方式：${result.confirm_click.method}\n`
        + `命中元素：${result.confirm_click.target}`
      );
      await postClickReport({
        debug_id: result.debug_id,
        stage: "confirm_click",
        confirm_click: result.confirm_click,
        crop_info: crop.cropInfo
      });
      await sleep(1000);
    } else {
      result.confirm_click = null;
      setStatus("已发送目标点击事件，但未找到验证码“确定”按钮。");
      await sleep(600);
    }
    await postClickReport({
      debug_id: result.debug_id,
      stage: "final_click_summary",
      task: result.task,
      sequence: result.sequence,
      points: result.points,
      click_results: clickResults,
      confirm_click: result.confirm_click,
      crop_info: crop.cropInfo
    });
    await debugStatus(
      `验证码点击事件已发送\n`
      + clickResults.map((row) => `${row.index}. ${row.label || "-"} image=(${row.imagePoint.x},${row.imagePoint.y}) css=(${row.cssPoint.x},${row.cssPoint.y}) page=(${row.viewportPoint.x},${row.viewportPoint.y}) target=${row.target}`).join("\n"),
      1500
    );
    return result;
  }

  function pageText() {
    if (!document.body) return "";
    const clone = document.body.cloneNode(true);
    clone.querySelector(`#${PANEL_ID}`)?.remove();
    return clone.innerText || "";
  }

  function isDetailPage() {
    const text = pageText();
    return text.includes("统一社会信用代码") && (text.includes("营业执照信息") || text.includes("基础信息"));
  }

  function isResultPage() {
    return Array.from(document.querySelectorAll("a.search_list_item, .search_list_item"))
      .some((el) => !isInPanel(el) && isVisible(el));
  }

  function isSearchPage() {
    return Boolean(document.querySelector("#keyword") && document.querySelector("#btn_query"));
  }

  function findFirstSearchResult() {
    const searchItems = Array.from(document.querySelectorAll("a.search_list_item, .search_list_item"))
      .filter((el) => !isInPanel(el) && isVisible(el));
    if (searchItems[0]) return searchItems[0];

    const encryptedLinks = Array.from(document.querySelectorAll("a"))
      .filter((el) => !isInPanel(el) && isVisible(el) && /gsxt\.gov\.cn\/(%7B|\{)/i.test(el.href || ""));
    return encryptedLinks[0] || null;
  }

  function findSendDialogButton() {
    const text = pageText();
    if (!text.includes("发送报告") && !text.includes("报告将发送")) return null;
    const nodes = Array.from(document.querySelectorAll("button, a, div, span"))
      .filter((el) => !isInPanel(el) && isVisible(el));
    return nodes.find((el) => textOf(el) === "发送")
      || nodes.find((el) => textOf(el).includes("发送") && textOf(el).length <= 6);
  }

  function findSendReportMenu(includeHidden = false) {
    const sendPdf = document.querySelector("#btn_send_pdf");
    if (sendPdf && !isInPanel(sendPdf) && (includeHidden || isVisible(sendPdf))) return sendPdf;

    const menu = document.querySelector("#moreActionsMenu");
    if (menu && !isInPanel(menu)) {
      const nodes = Array.from(menu.querySelectorAll("button, a, div, span, li"))
        .filter((el) => !isInPanel(el) && (includeHidden || isVisible(el)));
      return nodes.find((el) => textOf(el) === "发送报告")
        || nodes.find((el) => textOf(el).includes("发送报告"));
    }
    return byText("发送报告");
  }

  function findMoreButton() {
    const moreById = document.querySelector("#moreActionsToggle");
    if (moreById && !isInPanel(moreById) && isVisible(moreById)) return moreById;

    const nodes = Array.from(document.querySelectorAll("button, a, div, span, li"))
      .filter((el) => !isInPanel(el) && isVisible(el));
    return nodes.find((el) => textOf(el) === "更多")
      || nodes.find((el) => textOf(el).replace(/\s+/g, "").startsWith("更多"))
      || byText("更多");
  }

  async function clickMoreButton() {
    const more = findMoreButton();
    if (!more) throw new Error("未找到“更多”");

    const menu = document.querySelector("#moreActionsMenu");
    if (menu && !isInPanel(menu)) {
      menu.classList.remove("hidden");
      menu.style.display = "block";
      menu.style.visibility = "visible";
      menu.style.opacity = "1";
      menu.style.pointerEvents = "auto";
      await sleep(200);
      return Boolean(findSendReportMenu(true));
    }

    if (more.id === "moreActionsToggle") {
      clickElement(more, false);
      await sleep(500);
      return Boolean(findSendReportMenu());
    }

    const targets = [];
    let current = more;
    for (let depth = 0; current && current !== document.body && depth < 6; depth += 1) {
      if (!isInPanel(current) && isVisible(current) && !targets.includes(current)) {
        targets.push(current);
      }
      current = current.parentElement;
    }

    const preferred = clickableTarget(more);
    if (preferred && !targets.includes(preferred)) targets.unshift(preferred);

    for (const target of targets) {
      clickElement(target, false);
      await sleep(250);
      if (findSendReportMenu()) return true;
    }
    return false;
  }

  function classifyPage() {
    if (findSendDialogButton()) return "send_dialog";
    if (isResultPage()) return "result";
    if (isDetailPage()) return "detail";
    if (isSearchPage()) return "search";
    return "unknown";
  }

  async function finishCurrentTask() {
    const tasks = getTasks();
    const done = tasks.shift();
    appendLog("task_sent", classifyPage(), { doneTask: done || "", remaining: Math.max(tasks.length, 0) });
    setTasks(tasks);
    writeTabState({ pageRecoveryRefreshes: 0, captchaImageRefreshes: 0 });
    if (!tasks.length) {
      setRunning(false);
    }
    setStatus(done ? `已发送：${done}\n剩余 ${tasks.length} 条。` : "已点击发送。");
    await sleep(1500);
    if (tasks.length && running) {
      window.location.assign("https://www.gsxt.gov.cn/index.html");
    }
  }

  async function actOnce() {
    if (busy) return;
    if (running && !autoStateIsValid()) return;
    busy = true;
    const runToken = running ? currentRunToken() : null;
    try {
      const tasks = getTasks();
      const keyword = tasks[0];
      const tabState = readTabState();
      submittedKeyword = tabState.submittedKeyword || submittedKeyword;
      const pageType = classifyPage();
      const captcha = detectCaptcha();

      if (captcha.visible) {
        const searchInputValue = document.querySelector("#keyword")?.value?.trim() || "";
        const isPreSearchFalsePositive = pageType === "search" && !submittedKeyword && !searchInputValue;
        if (isPreSearchFalsePositive) {
          // Some normal GSXT pages contain generic verification-related DOM.
          // Do not block the first search until a query has actually been submitted.
        } else {
          const metaText = formatCaptchaMeta(captcha.meta);
          const state = readTabState();
          if (!running || Date.now() - (state.lastCaptchaLogAt || 0) > 15000) {
            appendLog("captcha_detected", pageType, {
              mode: running ? "auto" : "manual",
              captchaKind: captcha.kind || "captcha",
              captchaMeta: captcha.meta
            });
            if (running) writeTabState({ lastCaptchaLogAt: Date.now() });
          }

          const gsxtElement = findCaptchaElement() || captcha.element;
          const gsxtRect = gsxtElement?.getBoundingClientRect?.();
          const gsxtCaptchaKey = [
            location.href,
            captcha.kind || "captcha",
            Math.round(gsxtRect?.left || 0),
            Math.round(gsxtRect?.top || 0),
            Math.round(gsxtRect?.width || 0),
            Math.round(gsxtRect?.height || 0)
          ].join("|");
          const gsxtTabState = readTabState();
          const gsxtAlreadyTried = gsxtTabState.captchaTriedId === gsxtCaptchaKey
            && gsxtTabState.captchaTriedAt
            && Date.now() - gsxtTabState.captchaTriedAt < CAPTCHA_RETRY_COOLDOWN_MS;

          if (gsxtTabState.captchaSolvedAt) {
            const gsxtElapsed = Date.now() - gsxtTabState.captchaSolvedAt;
            if (gsxtElapsed < 8000) {
              setStatus(`验证码已点击，等待页面响应...\n${Math.ceil((8000 - gsxtElapsed) / 1000)} 秒后继续`);
              if (running) schedule(1500);
              return;
            }
            writeTabState({ captchaSolvedAt: 0 });
          }

          if (gsxtAlreadyTried) {
            if (running) setRunning(false);
            writeTabState({ captchaTriedId: "", captchaTriedAt: 0 });
            setStatus("自动识别刚刚失败过，已停止自动流程以避免页面循环卡住。\n请手动完成验证码，完成后再点击“开始”继续。");
            return;
          }

          writeTabState({ captchaTriedId: gsxtCaptchaKey, captchaTriedAt: Date.now() });
          try {
            setStatus("正在截图并调用本地 GSXT Solver...");
            const gsxtResult = await solveCaptchaWithGsxt({ ...captcha, element: gsxtElement }, runToken);
            appendLog("captcha_auto_solved", pageType, {
              mode: running ? "auto" : "manual",
              solver: "gsxt_solver",
              result: gsxtResult
            });
            writeTabState({ captchaSolvedAt: Date.now(), captchaTriedId: "", captchaTriedAt: 0, captchaImageRefreshes: 0, pageRecoveryRefreshes: 0 });
            setStatus(`验证码点击完成，等待页面响应...\n${(gsxtResult.sequence || []).join(" → ")}`);
            if (running) schedule(2500);
            return;
          } catch (err) {
            const gsxtError = err?.message || String(err);
            if (runToken != null && (!running || runToken !== currentRunToken())) {
              return;
            }
            appendLog("captcha_auto_failed", pageType, {
              mode: running ? "auto" : "manual",
              solver: "gsxt_solver",
              error: gsxtError,
              details: err?.details || null
            });
            writeTabState({ captchaTriedId: "", captchaTriedAt: 0, captchaSolvedAt: 0 });
            if (err instanceof CaptchaRefreshRequested) {
              const stateAfterSolve = readTabState();
              const refreshCount = Number(stateAfterSolve.captchaImageRefreshes || 0) + 1;
              if (!running) {
                setStatus(`自动识别结果不是 ${CAPTCHA_EXPECTED_POINTS} 个目标。\n原因：${gsxtError}`);
                return;
              }
              if (refreshCount > MAX_CAPTCHA_IMAGE_REFRESHES) {
                writeTabState({ captchaImageRefreshes: 0 });
                schedulePageRecoveryRefresh(
                  `验证码连续 ${MAX_CAPTCHA_IMAGE_REFRESHES} 次未识别出 ${CAPTCHA_EXPECTED_POINTS} 个目标，改为刷新页面恢复。`
                );
                return;
              }
              const refreshRow = await clickCaptchaRefresh(gsxtElement, "solver_result_not_three");
              if (!refreshRow) {
                writeTabState({ captchaImageRefreshes: 0 });
                schedulePageRecoveryRefresh("未找到验证码更换图片按钮，改为刷新页面恢复。");
                return;
              }
              writeTabState({ captchaImageRefreshes: refreshCount });
              await postClickReport({
                debug_id: err.details?.debug_id || "-",
                stage: "captcha_image_refresh",
                message: `solver returned ${err.details?.points?.length ?? 0} points, expected ${CAPTCHA_EXPECTED_POINTS}`,
                refresh_click: refreshRow,
                task: err.details?.task,
                sequence: err.details?.sequence,
                points: err.details?.points,
                crop_info: err.details?.crop_info
              });
              setStatus(
                `识别到 ${err.details?.points?.length ?? 0} 个目标，不是 ${CAPTCHA_EXPECTED_POINTS} 个，已点击更换图片。\n`
                + `第 ${refreshCount}/${MAX_CAPTCHA_IMAGE_REFRESHES} 次刷新验证码，稍后重新识别。`
              );
              if (running) schedule(2500);
              return;
            }
            if (running) {
              schedulePageRecoveryRefresh(`自动识别异常，将刷新页面恢复。\n原因：${gsxtError}`);
              return;
            }
            setStatus(`自动识别失败。\n原因：${gsxtError}`);
            return;
          }

          // 尝试通过本地 GeekedTest 服务自动识别验证码
          const captchaId = pickCaptchaField(captcha.meta, [/captcha[_-]?id/i]);
          let riskType = pickCaptchaField(captcha.meta, [/risk[_-]?type/i]);
          const RISK_TYPES_ALL = ["icon", "slide", "gobang", "ai"];
          const validRiskType = ["slide", "gobang", "icon", "ai"].includes(riskType);

          // 没有拿到 captcha_id 则直接回退手动
          if (!captchaId) {
            setStatus(`检测到验证码，未能获取 captcha_id，请手动完成。\n类型线索：${captcha.kind || "captcha"}${metaText ? `\n${metaText}` : ""}`);
            if (running) schedule(2000);
            return;
          }

          // 确定要尝试的 risk_type 列表
          const riskTypesToTry = validRiskType ? [riskType] : RISK_TYPES_ALL;

          async function trySolve(rid) {
            setStatus(`正在尝试自动识别验证码...\ncaptcha_id: ${captchaId}\nrisk_type: ${rid}${!validRiskType ? `\n（未获取到 risk_type，轮流尝试）` : ""}`);
            const resp = await fetch("http://127.0.0.1:7755/solve", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ captcha_id: captchaId, risk_type: rid })
            });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const result = await resp.json();
            if (result.error) throw new Error(result.error);
            return result;
          }

          function applyResult(result, rid) {
            const fieldMap = {
              lot_number: ["lot_number", "lotNumber"],
              pass_token: ["pass_token", "passToken"],
              gen_time: ["gen_time", "genTime"],
              captcha_output: ["captcha_output", "captchaOutput"],
              captcha_id: ["captcha_id", "captchaId"]
            };
            for (const [resKey, candidates] of Object.entries(fieldMap)) {
              const val = result[resKey];
              if (!val) continue;
              for (const name of candidates) {
                const el = document.querySelector(`input[name="${name}"], input[id="${name}"]`);
                if (el && !isInPanel(el)) { setInputValue(el, val); break; }
              }
            }
            const geetestBox = document.querySelector("[class*='geetest'], [id*='geetest'], [class*='captcha'], [id*='captcha']");
            const submitBtn = geetestBox
              ? geetestBox.closest("form")?.querySelector("button[type=submit], input[type=submit]")
              : null;
            if (submitBtn && !isInPanel(submitBtn)) clickElement(submitBtn);

            appendLog("captcha_auto_solved", pageType, {
              mode: running ? "auto" : "manual",
              captchaId,
              riskType: rid,
              result
            });
            setStatus(`验证码自动识别成功！\nrisk_type: ${rid}\nlot_number: ${result.lot_number || "-"}\npass_token: ${(result.pass_token || "").slice(0, 16)}...`);
          }

          // 检查本次验证码是否已尝试过一轮（用 captchaTriedAt + captchaTriedId 标记）
          const tabStatePre = readTabState();
          const alreadyTried = tabStatePre.captchaTriedId === captchaId
            && tabStatePre.captchaTriedAt
            && Date.now() - tabStatePre.captchaTriedAt < 5 * 60 * 1000; // 5分钟内同一captchaId不重复尝试

          // 若已提交成功，等待页面跳转
          if (tabStatePre.captchaSolvedAt) {
            const elapsed = Date.now() - tabStatePre.captchaSolvedAt;
            if (elapsed < 8000) {
              setStatus(`验证码已提交，等待页面响应...\n（${Math.ceil((8000 - elapsed) / 1000)} 秒后继续）`);
              if (running) schedule(1500);
              return;
            }
            writeTabState({ captchaSolvedAt: 0 });
          }

          if (alreadyTried) {
            // 已尝试过一轮且失败，只提示手动，不再自动重试
            setStatus(
              `自动识别已尝试过，请手动完成验证码。\n完成后流程将自动继续。\n`
              + `captcha_id: ${captchaId}`
            );
            if (running) schedule(3000);
            return;
          }

          // 标记本次开始尝试
          writeTabState({ captchaTriedId: captchaId, captchaTriedAt: Date.now() });

          let solved = false;
          const errors = [];
          for (let i = 0; i < riskTypesToTry.length; i++) {
            const rid = riskTypesToTry[i];
            const progress = riskTypesToTry.length > 1 ? `（${i + 1}/${riskTypesToTry.length}）` : "";
            setStatus(`正在尝试 risk_type: ${rid} ${progress}\ncaptcha_id: ${captchaId}`);
            try {
              const result = await trySolve(rid);
              applyResult(result, rid);
              solved = true;
              writeTabState({ captchaSolvedAt: Date.now(), captchaTriedId: "", captchaTriedAt: 0 });
              setStatus(`验证码自动识别成功！\nrisk_type: ${rid}\n等待页面响应后继续...`);
              if (running) schedule(2500);
              break;
            } catch (err) {
              const msg = err.message || String(err);
              errors.push(`${rid}: ${msg}`);
              setStatus(`risk_type: ${rid} 失败 ${progress}\n原因：${msg}${i < riskTypesToTry.length - 1 ? `\n正在尝试下一个...` : ""}`);
              if (i < riskTypesToTry.length - 1) await sleep(800);
            }
          }

          if (!solved) {
            setStatus(
              `自动识别失败，请手动完成验证码。\n完成后流程将自动继续，无需重新点"开始"。\n`
              + `captcha_id: ${captchaId}\n`
              + `失败详情：\n${errors.join("\n")}`
            );
            // 轮询等待用户手动过验证码，页面跳转后自动继续
            if (running) schedule(3000);
          }
          return;
        }
      }

      if (pageType === "send_dialog") {
        submittedKeyword = "";
        if (!markAutoAction("send_dialog:send")) return;
        writeTabState({ pageRecoveryRefreshes: 0 });
        setStatus("当前页面：发送确认弹窗\n点击“发送”。");
        clickElement(findSendDialogButton());
        await finishCurrentTask();
        return;
      }

      if (pageType === "detail") {
        submittedKeyword = "";
        const state = readTabState();
        if (running && state.lastAction === "detail:send_report") {
          const waited = Date.now() - (state.lastActionAt || Date.now());
          if (waited > MAX_WAIT_MS) {
            schedulePageRecoveryRefresh("已点击“发送报告”，但长时间未进入发送确认弹窗，刷新页面恢复。");
            return;
          }
          setStatus("已点击“发送报告”。\n未检测到验证码，等待发送确认弹窗出现，不会重复点击。");
          schedule(2000);
          return;
        }
        const report = findSendReportMenu();
        if (report) {
          if (!markAutoAction("detail:send_report")) return;
          writeTabState({ pageRecoveryRefreshes: 0 });
          setStatus("当前页面：详情页，菜单已展开\n点击“发送报告”。如有验证码，将尝试自动截图识别。");
          clickElement(report);
          schedule(1200);
          return;
        }
        const more = findMoreButton();
        if (more) {
          if (!markAutoAction("detail:open_more")) return;
          writeTabState({ pageRecoveryRefreshes: 0 });
          setStatus("当前页面：详情页\n展开“更多”菜单并点击“发送报告”。");
          const opened = await clickMoreButton();
          const reportAfterOpen = findSendReportMenu(true);
          if (opened && reportAfterOpen) {
            clickElement(reportAfterOpen);
            schedule(1200);
          } else {
            if (running) {
              schedulePageRecoveryRefresh("当前页面：详情页\n已尝试展开“更多”，但没有识别到“发送报告”。");
            } else {
              setStatus("当前页面：详情页\n已尝试展开“更多”，但没有识别到“发送报告”。");
            }
          }
          return;
        }
        if (running) {
          schedulePageRecoveryRefresh("当前页面：详情页\n未找到“更多”或“发送报告”。");
        } else {
          setStatus("当前页面：详情页\n未找到“更多”或“发送报告”。");
        }
        return;
      }

      if (pageType === "result") {
        submittedKeyword = "";
        if (!markAutoAction("result:first_result")) return;
        writeTabState({ pageRecoveryRefreshes: 0 });
        setStatus("当前页面：搜索结果页\n点击第一条搜索结果。");
        const firstResult = findFirstSearchResult();
        if (!firstResult) throw new Error("未找到第一条搜索结果");
        if (firstResult.href) {
          firstResult.removeAttribute("target");
          window.location.assign(firstResult.href);
        } else {
          clickElement(firstResult);
        }
        schedule(1500);
        return;
      }

      if (pageType === "search") {
        if (!keyword) {
          setStatus("当前页面：首页/搜索页\n任务框为空，请先填写企业名称或统一社会信用代码。");
          return;
        }
        const input = document.querySelector("#keyword");
        if (running && submittedKeyword === keyword && input && input.value.trim() === keyword) {
          const state = readTabState();
          const waitStartedAt = state.waitStartedAt || Date.now();
          writeTabState({ waitStartedAt });
          if (Date.now() - waitStartedAt > MAX_WAIT_MS) {
            schedulePageRecoveryRefresh(`已提交查询：${keyword}\n等待结果超过 ${Math.round(MAX_WAIT_MS / 1000)} 秒，刷新页面恢复。`);
            return;
          }
          setStatus(`当前页面：首页/搜索页\n已提交查询：${keyword}\n未检测到验证码，等待结果页。`);
          schedule(2000);
          return;
        }
        if (!markAutoAction(`search:${keyword}`)) return;
        writeTabState({ pageRecoveryRefreshes: 0 });
        setStatus(`当前页面：首页/搜索页\n查询：${keyword}`);
        const button = document.querySelector("#btn_query");
        setInputValue(input, keyword);
        await sleep(300);
        clickElement(button);
        submittedKeyword = keyword;
        writeTabState({ submittedKeyword: keyword, waitStartedAt: Date.now() });
        setStatus(`已点击查询：${keyword}\n等待结果页；如出现验证码会自动截图识别。`);
        schedule(2000);
        return;
      }

      if (running) {
        appendLog("unknown_page_refresh", pageType, { url: location.href, keyword });
        schedulePageRecoveryRefresh(`当前页面：未知\n当前任务：${keyword || "无"}`);
      } else {
        setStatus(`当前页面：未知\n请手动停在首页、结果页、详情页或发送弹窗。\n当前任务：${keyword || "无"}`);
      }
    } catch (err) {
      console.error("[GSXT assistant]", err);
      if (runToken != null && (!running || runToken !== currentRunToken())) {
        return;
      }
      const message = err?.message || String(err);
      appendLog("auto_error_refresh", classifyPage(), { message, url: location.href });
      if (running) {
        schedulePageRecoveryRefresh(`流程出错，将刷新页面恢复。\n原因：${message}`);
      } else {
        setStatus(`出错：${message}`);
      }
    } finally {
      busy = false;
    }
  }

  function schedule(delay) {
    if (!running || !autoStateIsValid()) return;
    const token = currentRunToken();
    const timer = setTimeout(() => {
      scheduledTimers.delete(timer);
      if (!isRunTokenActive(token)) return;
      actOnce();
    }, delay);
    scheduledTimers.add(timer);
  }

  function schedulePageRecoveryRefresh(message, delay = PAGE_RECOVERY_REFRESH_MS) {
    if (!running || !autoStateIsValid()) return;
    const state = readTabState();
    const refreshCount = Number(state.pageRecoveryRefreshes || 0) + 1;
    if (refreshCount > MAX_PAGE_RECOVERY_REFRESHES) {
      appendLog("page_recovery_refresh_limit", classifyPage(), {
        message,
        refreshCount: refreshCount - 1,
        maxRefreshes: MAX_PAGE_RECOVERY_REFRESHES,
        url: location.href
      });
      setRunning(false);
      setStatus(
        `自动恢复刷新已达到上限：${MAX_PAGE_RECOVERY_REFRESHES} 次。\n`
        + `${message}\n`
        + `流程已安全停止，请手动检查页面后再点击“开始”。`
      );
      return;
    }
    writeTabState({ pageRecoveryRefreshes: refreshCount });
    const token = currentRunToken();
    setStatus(`${message}\n${Math.ceil(delay / 1000)} 秒后刷新页面重试（${refreshCount}/${MAX_PAGE_RECOVERY_REFRESHES}）。点击“停止”可取消。`);
    const timer = setTimeout(() => {
      scheduledTimers.delete(timer);
      if (!isRunTokenActive(token)) return;
      window.location.reload();
    }, delay);
    scheduledTimers.add(timer);
  }

  function startRun() {
    saveTasks();
    clearScheduledTimers();
    runGeneration += 1;
    running = true;
    submittedKeyword = "";
    const runId = makeRunId("run");
    writeTabState({
      running: true,
      runId,
      startedAt: Date.now(),
      steps: 0,
      submittedKeyword: "",
      waitStartedAt: 0
    });
    appendLog("start", classifyPage(), { mode: "auto", runId });
    setStatus("已开始。扩展会按当前页面类型决定下一步。");
    actOnce();
  }

  function stopRun() {
    appendLog("stop", classifyPage(), { mode: running ? "auto" : "manual" });
    setRunning(false);
    setStatus("已停止。");
  }

  function highlightFirstSearchResult() {
    const el = findFirstSearchResult();
    if (!el) {
      setStatus("未识别到第一条搜索结果。");
      return;
    }
    el.style.outline = "4px solid #ff3b30";
    el.style.backgroundColor = "rgba(255, 59, 48, 0.12)";
    el.scrollIntoView({ block: "center", inline: "center" });
    setStatus(`已高亮第一条结果：${textOf(el).slice(0, 80)}`);
  }

  function diagnose() {
    const captcha = detectCaptcha();
    const metaText = formatCaptchaMeta(captcha.meta);
    setStatus(
      `当前页面：${classifyPage()}\n`
      + `验证码：${captcha.visible ? "已检测到" : "未检测到"}\n`
      + `${captcha.visible ? `类型线索：${captcha.kind || "captcha"}\n` : ""}`
      + `${metaText ? `${metaText}\n` : ""}`
      + `当前任务：${currentTask() || "无"}\n`
      + `URL：${location.href}`
    );
  }

  function injectPanel(savedTasks) {
    if (document.getElementById(PANEL_ID)) return;
    const panel = document.createElement("div");
    panel.id = PANEL_ID;
    panel.innerHTML = `
      <div class="ga-header">
        <span>GSXT 发送报告助手</span>
        <button type="button" class="ga-secondary" data-ga-min>收起</button>
      </div>
      <div class="ga-body">
        <textarea data-ga-tasks placeholder="一行一个企业名称或统一社会信用代码">${savedTasks || ""}</textarea>
        <div class="ga-row">
          <button type="button" data-ga-start>开始</button>
          <button type="button" class="ga-danger" data-ga-stop>停止</button>
          <button type="button" class="ga-secondary" data-ga-save>保存任务</button>
        </div>
        <div class="ga-row">
          <button type="button" class="ga-secondary" data-ga-step>识别并执行一步</button>
          <button type="button" class="ga-secondary" data-ga-diag>识别当前页</button>
          <button type="button" class="ga-secondary" data-ga-highlight>高亮第一条</button>
        </div>
        <div class="ga-row">
          <button type="button" class="ga-secondary" data-ga-export>导出日志</button>
          <button type="button" class="ga-secondary" data-ga-clear-log>清空日志</button>
        </div>
        <div class="ga-status">正常登录并停在首页/结果页/详情页/发送弹窗均可。只有点击“开始”或“识别并执行一步”后才会操作页面。</div>
        <div class="ga-note">每次都识别当前页面，再决定下一步；流程日志可导出为 CSV，用 Excel 打开。</div>
      </div>
    `;
    document.documentElement.appendChild(panel);

    panel.querySelector("[data-ga-start]").addEventListener("click", startRun);
    panel.querySelector("[data-ga-stop]").addEventListener("click", stopRun);
    panel.querySelector("[data-ga-save]").addEventListener("click", () => {
      saveTasks();
      setStatus("任务已保存。");
    });
    panel.querySelector("[data-ga-step]").addEventListener("click", actOnce);
    panel.querySelector("[data-ga-diag]").addEventListener("click", diagnose);
    panel.querySelector("[data-ga-highlight]").addEventListener("click", highlightFirstSearchResult);
    panel.querySelector("[data-ga-export]").addEventListener("click", exportLogs);
    panel.querySelector("[data-ga-clear-log]").addEventListener("click", clearLogs);
    panel.querySelector("[data-ga-min]").addEventListener("click", (event) => {
      panel.classList.toggle("gsxt-minimized");
      event.target.textContent = panel.classList.contains("gsxt-minimized") ? "展开" : "收起";
    });
  }

  chrome.storage.local.get([TASKS_KEY], (data) => {
    const state = readTabState();
    running = Boolean(state.running);
    submittedKeyword = state.submittedKeyword || "";
    injectPanel(data[TASKS_KEY] || "");
    if (running && autoStateIsValid()) {
      setStatus("自动流程继续中，正在识别当前页面...");
      schedule(800);
    }
  });
})();
